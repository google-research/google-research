# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for camera optimization."""

import abc
from collections.abc import Callable, Mapping
import functools
from typing import Any

import chex
from flax import linen as nn
import gin
import immutabledict
from internal import coord
from internal import geometry
from internal import math as mipnerf360_math
from internal import rigid_body
from internal import spin_math
import jax
from jax import random
import jax.numpy as jnp
import jaxcam


def _v_ravel_pytree(pytree):
  """Ravels a batched pytree for each batch separately.

  Unfortunately `ravel_pytree` cannot be directly used with `jax.vmap` because
  it returns a function (`unflatten_fn`). We therefore apply vmap to just the
  first return value, which is the flattened params, and fetch the unflatten
  function separately.

  Example:

      flat_params, unflatten_fn = _v_ravel_pytree(pytree)
      pytree = jax.vmap(unflatten_fn)(flat_params)


  Args:
    pytree: The pytree to flatten.

  Returns:
    A tuple containing the flattened pytree, with each batch item flattened
    separately, and the unbatched unflatten function. The unflatten function
    must be vmapped.
  """
  flat_params = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])(pytree)
  _, unflatten_fn = jax.flatten_util.ravel_pytree(
      jax.tree_util.tree_map(lambda x: x[0], pytree)
  )

  return flat_params, unflatten_fn


@gin.configurable
class CameraDelta(abc.ABC, nn.Module):
  """An abstract module for optimizing cameras.

  A CameraDelta is a module that is applied to a camera for the purpose of
  optimization. This abstraction allows for different parameterizations of
  camera transforms to be easily swapped in and out.

  Example:

      camera: jaxcam.Camera = ...
      camera_delta = FocalPoseCameraDelta()
      params = camera_delta.init({'params': None}, camera)
      optimized_params = optimize_camera(camera, params)
      optimized_camera = camera_delta.apply(optimized_params, camera)
  """

  # The bounding box of the scene. It is currently only used to generate
  # points used for preconditioning.
  bbox: None | jnp.ndarray | tuple[
      tuple[float, float, float], tuple[float, float, float]
  ] = None

  # If True, use log scale in place if biases for length parameters such as
  # focal length or the z-translation.
  use_log_scales: bool = False

  # If True will enable preconditioning. See `_compute_precondition_matrix()`.
  use_precondition: bool = False

  # A padding to add to the diagonal before taking the square root when
  # computing the preconditioning matrix. `M' = M + λI` where M is the
  # preconditioning matrix and λ is this padding. This is similar to the damping
  # parameter used in the Levenberg-Marquadt algorithm. If λ is large it is
  # similar to having no preconditioning.
  # The maximum of the absolute padding and the relative padding will be added.
  precondition_diagonal_absolute_padding: float = 1e-8
  # The relative padding scale, which multiplied by the diagonal of J^T J.
  precondition_diagonal_relative_padding_scale: float = 1e-2

  # If True, use the full preconditioning matrix. Otherwise, take the diagonal.
  # The full preconditioning matrix will "decorrelate" different parameters
  # while the diagonal will only scale them.
  precondition_use_full_matrix: bool = True
  # If True, track a running estimate of the preconditioning matrix rather than
  # keeping it fixed.
  precondition_running_estimate: bool = True
  # The momentum for the preconditioning matrix.
  precondition_momentum: float = 1.0
  # Set to True if this is an instance used for training.
  is_training: bool = False

  # The number of points used to compute the preconditioning matrix.
  precondition_num_points: int = 1000
  # The method to use to sample the points for preconditioning.
  # Possible values are:
  #   'bbox': Sample points uniformly within a bounding bo.
  #   'frustum': Sample points uniformly within the camera frustum.
  #   'frustum_contracted': Sample points uniformly in a contracted frustum.
  precondition_point_sample_method: str = 'bbox'

  # The ray distance function to be used with the # 'frustum_raydist_fn` option
  # for point sampling. The default is set to the Zip-NeRF parameters.
  precondition_raydist_fn: Callable[..., Any] = mipnerf360_math.power_ladder
  precondition_raydist_inv_fn: Callable[..., Any] = (
      mipnerf360_math.inv_power_ladder
  )
  precondition_raydist_fn_kwargs: Mapping[str, Any] = (
      immutabledict.immutabledict({'p': -1.5, 'premult': 1})
  )
  # The near plane depth of the point sampling frustum.
  precondition_near: float = 0.1
  # The far plane depth of the point sampling frustum.
  precondition_far: float = 1000
  # If True, normalize the eigenvalues of the preconditionoing matrix.
  precondition_normalize_eigvals: bool = False

  # If True, scale any parameters that are in pixels to metric units.
  # For example, `focal_bias` in `FocalPoseCameraDelta` can be scaled to world
  # coordinates by multiplying the bias by the current focal length.
  scale_pixel_units_to_metric_units: bool = False

  def _compute_approximate_hessian(
      self,
      camera_params: chex.ArrayTree,
      points: jnp.ndarray,
      camera: jaxcam.Camera,
  ) -> jnp.ndarray:
    flat_camera_params, unflatten_fn = jax.flatten_util.ravel_pytree(
        camera_params
    )

    def _project_points(flat_camera_params, points, camera):
      """Computes the 3D to 2D projection."""
      camera_params = unflatten_fn(flat_camera_params)
      camera = self.transform_camera(camera_params, camera)
      pixels = jaxcam.project(camera, points)
      # Scale pixels by the size of the image to make it resolution agnostic.
      max_image_size = jnp.maximum(camera.image_size_x, camera.image_size_y)
      return pixels / max_image_size, pixels

    # Compute the Jacobian of the camera projection function with respect to
    # the camera delta parameters. See the method docstring for a breakdown of
    # the math.
    jac_fn = jax.jacfwd(_project_points, has_aux=True)
    jacs, pixels = jax.vmap(jac_fn, in_axes=(None, 0, None))(
        flat_camera_params, points, camera
    )
    # Ignore points that are outside of the camera viewport. This could happen
    # under extreme distortion parameters.
    pixels_in_bounds = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < camera.image_size_x)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < camera.image_size_y)
    )
    jacs = jnp.where(
        pixels_in_bounds[..., None, None], jacs, jnp.zeros_like(jacs)
    )
    jtj = jax.vmap(lambda x: spin_math.matmul(x.T, x))(jacs)
    # Take the mean across the points.
    jtj = jnp.sum(jtj, axis=0) / pixels_in_bounds.sum(axis=0)

    return jtj

  def precondition_matrix_from_jtj(self, jtj: jnp.ndarray) -> jnp.ndarray:
    """Computes the preconditioning matrix.

    This function computes a matrix that when left-multiplied to a vector
    corresponding to the flattened camera delta parameters decorrelates them.

    The preconditioning matrix $M$ is computed by taking the Jacobian $J$ of
    the pixel projection function $p$ w.r.t. the camera delta parameters. If
    you compute $J^T J$, the diagonals will look like

      (dp_x/dw_i)^2 + (dp_y/dw_i)^2,

    while the off-diagonals will look like

      (dp_x/dw_i)(dp_x/dw_j) + (dp_y/dw_i)(dp_y/dw_j).

    If you take $M = sqrtm(J^T J)$, the diagonal of $M$ essentially encodes the
    Euclidean magnitude of the change in pixel space w.r.t. the change in
    parameter space while the off-diagonals encode correlation between different
    parameters. Multiplying parameters by the inverse of $M$ can be thought of
    as the parameters from a decorrelated space to a metric space where the
    cameras actually live.

    A useful breakdown of $M$ is to consider its Eigendecomposition, which can
    be computed since $J^T J$ is symmetric. Note that the matrix square root
    is implemented using this:

      sqrtm(J^T J) = P diag(sqrt(diag(D))) P^-1.

    Given the Eigendecomposition, it's easy to see that $M$ is rotating the
    parameters to the Eigenbasis defined by $M$, rescaling things, and then
    rotating it back.

    We compute the expected matrix $M$ over uniformly sample points within the
    scene bounding box.

    Args:
      jtj: The approximate Hessian to compute the preconditioning matrix with.

    Returns:
      A (num_params, num_params) matrix that transforms the parameters into a
      decorrelated space.
    """
    # Add a diagonal padding. See `precondition_diagonal_padding`.
    diagonal_absolute_padding = (
        self.precondition_diagonal_absolute_padding * jnp.ones(jtj.shape[-1])
    )
    diagonal_relative_padding = (
        self.precondition_diagonal_relative_padding_scale * jnp.diag(jtj)
    )
    diagonal_padding = jnp.diag(
        jnp.maximum(diagonal_absolute_padding, diagonal_relative_padding)
    )
    if self.precondition_use_full_matrix:
      matrix, _ = spin_math.inv_sqrtm(
          jtj + diagonal_padding,
          normalize_eigvals=self.precondition_normalize_eigvals,
      )
    else:
      # TODO(keunhong): Consider optimizing this code path by only tracking the
      # diagonal.
      jtj_diag = jnp.diag(jtj + diagonal_padding)
      if self.precondition_normalize_eigvals:
        log_jtj_diag = jnp.log(jtj_diag)
        jtj_diag = jnp.exp(
            log_jtj_diag - jnp.mean(log_jtj_diag, axis=-1, keepdims=True)
        )
      matrix = jnp.diag(1 / jnp.sqrt(jtj_diag))

    return matrix

  @abc.abstractmethod
  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    """Creates the camera delta parameters given the cameras.

    If preconditioning is enabled, this generates "latent" parameters.
    Otherwise, this generates the actual camera parameters directly.

    Args:
      cameras: A batch of cameras to create parameters for.

    Returns:
      A dictionary containing the camera delta parameters for each camera.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    """Implements the camera delta transform on a single camera.

    Args:
      camera_params: The camera delta parameters for a single camera.
      camera: The camera to transform.

    Returns:
      The transformed camera.
    """
    raise NotImplementedError()

  def _create_points_from_frustum(
      self, camera: jaxcam.Camera, rng: chex.PRNGKey
  ) -> jnp.ndarray:
    rng, key1, key2 = random.split(rng, 3)
    pixels = (
        random.uniform(key1, (self.precondition_num_points, 2))
        * jnp.array([camera.image_size_x - 1, camera.image_size_y - 1])
        + 0.5
    )
    depths = random.uniform(
        key2,
        (self.precondition_num_points,),
        minval=self.precondition_near,
        maxval=self.precondition_far,
    )
    points = jaxcam.pixels_to_points(camera, pixels, depths)[..., :3]
    return points

  def _create_points_from_contracted_frustum(
      self,
      camera: jaxcam.Camera,
      rng: chex.PRNGKey,
      sample_depth_contracted: bool = True,
  ) -> jnp.ndarray:
    """Samples points uniformly in the contracted frustum.

    We first compute the contracted camera frustum by intersecting camera rays
    with the bounding sphere (which has radius 2). This defines a frustum from
    the near plane to infinity. We can then apply the inverse of the contraction
    to the points to get the metric point samples.

    Args:
      camera: The camera used to compute the frustum.
      rng: A PRNGKey used to sample points.
      sample_depth_contracted: If True, sample the depth in the contracted
        space. Otherwise, sample linearly in metric space.

    Returns:
      Points sampled uniformly in the contracted frustum.
    """
    if self.precondition_far >= 2.0:
      raise ValueError('Far plane must be <2 when using contracted planes.')

    rng, key1, key2 = random.split(rng, 3)
    pixels = (
        random.uniform(key1, (self.precondition_num_points, 2))
        * jnp.array([camera.image_size_x - 1, camera.image_size_y - 1])
        + 0.5
    )
    rays = jaxcam.pixels_to_rays(camera, pixels)
    near_points = camera.position + rays * self.precondition_near
    far_points = geometry.ray_sphere_intersection(
        camera.position, rays, radius=self.precondition_far
    )
    s_dist = random.uniform(key2, (self.precondition_num_points, 1))
    if sample_depth_contracted:
      # Lerp between contracted near and far plane.
      points = s_dist * far_points + (1 - s_dist) * near_points
      points = coord.inv_contract(points)
    else:
      # Lerp between uncontracted near and far plane.
      near_points = coord.inv_contract(near_points)
      far_points = coord.inv_contract(far_points)
      points = s_dist * far_points + (1 - s_dist) * near_points
    return points

  def _create_points_from_raydist_fn(
      self,
      camera: jaxcam.Camera,
      rng: chex.PRNGKey,
  ) -> jnp.ndarray:
    """Samples points using a ray distance function."""
    rng, key1, key2 = random.split(rng, 3)
    kwargs = self.precondition_raydist_fn_kwargs
    _, s_to_t = coord.construct_ray_warps(
        t_near=self.precondition_near,
        t_far=self.precondition_far,
        fn=functools.partial(self.precondition_raydist_fn, **kwargs),
        fn_inv=functools.partial(self.precondition_raydist_inv_fn, **kwargs),
    )
    # Sample normalized distances, then map to metric.
    s_dist = random.uniform(key2, (self.precondition_num_points, 1))
    t_dist = s_to_t(s_dist)
    # Sample random pixel positions.
    pixels = random.uniform(
        key1,
        (self.precondition_num_points, 2),
    )
    pixels *= (
        jnp.array([camera.image_size_x - 1, camera.image_size_y - 1]) + 0.5
    )
    rays = jaxcam.pixels_to_rays(camera, pixels)
    return camera.position + rays * t_dist

  def _create_points_from_bbox(
      self, camera: jaxcam.Camera, rng: chex.PRNGKey
  ) -> jnp.ndarray:
    rng, key = random.split(rng)
    # Generate points within the bounding box to compute the average
    # preconditioning metric over.
    bbox = jnp.array(self.bbox)
    return (
        random.uniform(
            key, (self.precondition_num_points, 3), minval=0, maxval=1
        )
        * (bbox[1] - bbox[0])
        + bbox[0]
    )

  def create_points(
      self, camera: jaxcam.Camera, rng: chex.PRNGKey
  ) -> jnp.ndarray:
    match self.precondition_point_sample_method:
      case 'frustum':
        return self._create_points_from_frustum(camera, rng=rng)
      case 'frustum_unbounded':
        return self._create_points_from_contracted_frustum(
            camera, rng=rng, sample_depth_contracted=False
        )
      case 'frustum_contracted':
        return self._create_points_from_contracted_frustum(camera, rng=rng)
      case 'frustum_raydist_fn':
        return self._create_points_from_raydist_fn(camera, rng=rng)
      case 'bbox':
        return self._create_points_from_bbox(camera, rng=rng)
      case _:
        raise ValueError(
            f'Unknown sample method {self.precondition_point_sample_method}'
        )

  def compute_approximate_hessian(
      self,
      camera_params: chex.ArrayTree,
      cameras: jaxcam.Camera,
      rng: chex.PRNGKey,
  ) -> jnp.ndarray:
    """Computes the approximate Hessian matrix for the given cameras.

    Args:
      camera_params: The camera parameters to compute the matrix at.
      cameras: The cameras to compute the matrix for. Note that this should be
        the _raw_ cameras without the camera delta applied, since the model will
        internally apply the camera delta.
      rng: A PRNGKey for used to generate the preconditioning points.

    Returns:
      The approximate Hessian matrix for each camera.
    """
    rng, key = random.split(rng)
    keys = random.split(key, len(cameras))
    transformed_cameras = jax.vmap(self.transform_camera)(
        camera_params, cameras
    )
    points = jax.vmap(self.create_points)(transformed_cameras, keys)
    v_compute_jtj = jax.vmap(self._compute_approximate_hessian)
    return v_compute_jtj(camera_params, points, cameras)

  def apply_precondition_matrix(
      self,
      target_params: chex.ArrayTree,
      matrix: jnp.ndarray,
  ) -> chex.ArrayTree:
    """Applies the preconditioning matrix to the given parameters."""

    def _apply_precondition_matrix(target, matrix):
      flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(target)
      flat_params = spin_math.matmul(matrix, flat_params)
      return unflatten_fn(flat_params)

    return jax.vmap(_apply_precondition_matrix)(target_params, matrix)

  def _compute_camera_params_from_latent(
      self, latent_params: chex.ArrayTree, jtj: jnp.ndarray
  ) -> chex.ArrayTree:
    """Converts params to metric params given JTJ."""
    # Compute and apply preconditioning matrix.
    precondition_matrix = jax.vmap(self.precondition_matrix_from_jtj)(jtj)
    return self.apply_precondition_matrix(
        latent_params, jax.lax.stop_gradient(precondition_matrix)
    )

  @nn.compact
  def get_camera_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    latent_params = self.create_params(cameras)
    camera_params = latent_params
    if self.use_precondition:
      def init_jtj(_):
        return self.compute_approximate_hessian(
            camera_params, cameras, random.PRNGKey(0)
        )

      flat_params, _ = _v_ravel_pytree(latent_params)
      jtj = self.variable(
          'precondition', 'jtj', init_jtj, (len(cameras), flat_params.shape[1])
      )
      # Compute current estimate of metric params using previous JTJ.
      camera_params = self._compute_camera_params_from_latent(
          latent_params, jtj.value
      )

      if self.is_initializing() or (
          self.precondition_running_estimate and self.is_training
      ):
        # Update JTJ.
        rng = self.make_rng('params')
        rng, key = random.split(rng)
        prev_jtj = jtj.value
        next_jtj = self.compute_approximate_hessian(camera_params, cameras, key)
        if self.is_initializing() or self.precondition_momentum == 1.0:
          jtj.value = prev_jtj
        else:
          jtj.value = (
              self.precondition_momentum * prev_jtj
              + (1 - self.precondition_momentum) * next_jtj
          )
        # Compute metric params using updated JTJ.
        camera_params = self._compute_camera_params_from_latent(
            latent_params, jtj.value
        )

    return camera_params

  def __call__(
      self, cameras: jaxcam.Camera, return_params: bool = False
  ) -> tuple[jaxcam.Camera, chex.ArrayTree] | jaxcam.Camera:
    camera_params = self.get_camera_params(cameras)
    transformed_cameras = jax.vmap(self.transform_camera)(
        camera_params, cameras
    )

    if return_params:
      return transformed_cameras, camera_params
    return transformed_cameras


@gin.configurable
class SE3CameraDelta(CameraDelta):
  """A naive camera delta using a log SE3 formulation."""

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    return {
        'screw_axis_bias': self.param(
            'screw_axis_bias', jax.nn.initializers.zeros, (*cameras.shape, 6)
        ),
    }

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    # Convert camera to screw axis representation.
    translation = spin_math.matmul(-camera.orientation, camera.position)
    transform = rigid_body.rp_to_se3(camera.orientation, translation)
    screw_axis = rigid_body.log_se3(transform)
    new_screw_axis = screw_axis + camera_params['screw_axis_bias']

    new_transform = rigid_body.exp_se3(new_screw_axis)
    new_orientation, new_translation = rigid_body.se3_to_rp(new_transform)
    new_position = spin_math.matmul(-new_orientation.T, new_translation)
    return camera.replace(
        orientation=new_orientation,
        position=new_position,
    )


@gin.configurable
class SE3WithFocalCameraDelta(SE3CameraDelta):
  """The SE3 camera formulation with a focal length parameter.."""

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    params = {
        **SE3CameraDelta.create_params(self, cameras),
    }
    if self.use_log_scales:
      params['log_focal_scale'] = self.param(
          'log_focal_scale', jax.nn.initializers.zeros, cameras.shape
      )
    else:
      params['focal_bias'] = self.param(
          'focal_bias', jax.nn.initializers.zeros, cameras.shape
      )

    return params

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    # Convert camera to screw axis representation.
    if self.use_log_scales:
      new_focal_length = camera.focal_length * jnp.exp(
          camera_params['log_focal_scale']
      )
    else:
      focal_bias = camera_params['focal_bias']
      if self.scale_pixel_units_to_metric_units:
        focal_bias *= jax.lax.stop_gradient(camera.focal_length)
      new_focal_length = focal_bias + camera.focal_length
    return SE3CameraDelta.transform_camera(
        self, camera_params, camera.replace(focal_length=new_focal_length)
    )


@gin.configurable
class FocalPoseCameraDelta(CameraDelta):
  """Camera delta using the focal pose formulation.

  This allows the focal length to be easily adjusted during optimization and
  results in the cameras moving much more freely.
  See:
    https://arxiv.org/abs/2204.05145, Sec. 3.2.
  """

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    params = {
        'x_bias': self.param(
            'x_bias', jax.nn.initializers.zeros, cameras.shape
        ),
        'y_bias': self.param(
            'y_bias', jax.nn.initializers.zeros, cameras.shape
        ),
        'axis_angle_bias': self.param(
            'axis_angle_bias', jax.nn.initializers.zeros, (*cameras.shape, 3)
        ),
    }
    if self.use_log_scales:
      params.update({
          'log_focal_scale': self.param(
              'log_focal_scale', jax.nn.initializers.zeros, cameras.shape
          ),
          'log_z_scale': self.param(
              'log_z_scale', jax.nn.initializers.zeros, cameras.shape
          ),
      })
    else:
      params.update({
          'focal_bias': self.param(
              'focal_bias', jax.nn.initializers.zeros, cameras.shape
          ),
          'z_bias': self.param(
              'z_bias', jax.nn.initializers.zeros, cameras.shape
          ),
      })
    return params

  def update_focal_pose(
      self,
      params: chex.ArrayTree,
      x: jnp.ndarray,
      y: jnp.ndarray,
      z: jnp.ndarray,
      focal_length: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Applies the focal pose update for the translation and focal length."""
    eps = jnp.finfo(jnp.float32).eps

    x_bias = params['x_bias']
    y_bias = params['y_bias']
    if self.scale_pixel_units_to_metric_units:
      x_bias = focal_length * x_bias
      y_bias = focal_length * y_bias

    if self.use_log_scales:
      new_z = z * jnp.exp(params['log_z_scale'])
      new_focal_length = focal_length * jnp.exp(params['log_focal_scale'])
    else:
      new_z = z + params['z_bias']
      focal_bias = params['focal_bias']
      if self.scale_pixel_units_to_metric_units:
        focal_bias = focal_length * focal_bias
      new_focal_length = focal_length + focal_bias

    new_x = (
        x_bias / new_focal_length
        + jnp.sign(z) * jnp.divide(x, jnp.abs(z) + eps)
    ) * new_z
    new_y = (
        y_bias / new_focal_length
        + jnp.sign(z) * jnp.divide(y, jnp.abs(z) + eps)
    ) * new_z
    return new_x, new_y, new_z, new_focal_length

  def update_orientation(
      self, params: chex.ArrayTree, orientation: jnp.ndarray
  ) -> jnp.ndarray:
    """Updates the orientation based on a 3-DoF axis-angle bias."""
    # The orientation is the world-to-object rotation matrix which is equivalent
    # to an object centric rotation as described in the paper.
    orientation_delta = rigid_body.exp_so3(params['axis_angle_bias'])
    new_orientation = spin_math.matmul(orientation, orientation_delta)
    return new_orientation

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    translation = spin_math.matmul(-camera.orientation, camera.position)
    x, y, z = jnp.split(translation, 3, -1)
    new_x, new_y, new_z, new_focal_length = self.update_focal_pose(
        camera_params, x, y, z, camera.focal_length
    )
    new_translation = jnp.concatenate([new_x, new_y, new_z], axis=-1)
    new_orientation = self.update_orientation(camera_params, camera.orientation)
    new_position = spin_math.matmul(-new_orientation.T, new_translation)

    return camera.replace(
        orientation=new_orientation,
        position=new_position,
        focal_length=new_focal_length,
    )


class IntrinsicCameraDelta(CameraDelta):
  """A camera delta that modifies the intrinsics."""

  num_radial_distortion_coeffs: int = 2
  use_principal_point: bool = True
  use_radial_distortion: bool = True

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    params = {}
    if self.use_principal_point:
      params['principal_point_bias'] = self.param(
          'principal_point_bias',
          jax.nn.initializers.zeros,
          (*cameras.shape, 2),
      )
    if self.use_radial_distortion:
      params['radial_distortion_bias'] = self.param(
          'radial_distortion_bias',
          jax.nn.initializers.zeros,
          (*cameras.shape, self.num_radial_distortion_coeffs),
      )
    return params

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    # Insert radial distortion if it doesn't exist.
    if self.use_radial_distortion:
      if camera.radial_distortion is None:
        camera = camera.replace(radial_distortion=jnp.zeros(*camera.shape, 4))
      radial_distortion_bias = camera_params['radial_distortion_bias']
      radial_distortion_bias = jnp.pad(
          radial_distortion_bias,
          pad_width=(0, 4 - jnp.shape(radial_distortion_bias)[-1]),
      )
      camera = camera.replace(
          radial_distortion=camera.radial_distortion + radial_distortion_bias
      )
    if self.use_principal_point:
      principal_point_bias = camera_params['principal_point_bias']
      if self.scale_pixel_units_to_metric_units:
        principal_point_bias *= jax.lax.stop_gradient(camera.focal_length)
      camera = camera.replace(
          principal_point=camera.principal_point + principal_point_bias,
      )

    return camera


@gin.configurable
class IntrinsicSE3WithFocalCameraDelta(
    IntrinsicCameraDelta, SE3WithFocalCameraDelta
):
  """SE3 with intrinsics."""

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    return {
        **IntrinsicCameraDelta.create_params(self, cameras),
        **SE3WithFocalCameraDelta.create_params(self, cameras),
    }

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    camera = SE3WithFocalCameraDelta.transform_camera(
        self, camera_params, camera
    )
    camera = IntrinsicCameraDelta.transform_camera(self, camera_params, camera)
    return camera


@gin.configurable
class IntrinsicFocalPoseCameraDelta(IntrinsicCameraDelta, FocalPoseCameraDelta):
  """FocalPose with intrinsics."""

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    return {
        **IntrinsicCameraDelta.create_params(self, cameras),
        **FocalPoseCameraDelta.create_params(self, cameras),
    }

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    camera = FocalPoseCameraDelta.transform_camera(self, camera_params, camera)
    camera = IntrinsicCameraDelta.transform_camera(self, camera_params, camera)
    return camera


@gin.configurable
class DollyCameraDelta(CameraDelta):
  """Camera delta using the focal pose formulation and a dolly zoom.

  This extends the focal pose formatulation with an additional "dolly" update.
  The dolly update is an additional transform on the distance to the object
  (z-translation). The difference is that this transform is also propagated
  to the focal length so that the projected size of the object does not change
  (i.e., a dolly zoom). This allows the model to more easily change the
  persepctive.
  """
  use_se3: bool = False

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    params = {}
    if self.use_se3:
      params.update(SE3WithFocalCameraDelta.create_params(self, cameras))
    else:
      params.update(FocalPoseCameraDelta.create_params(self, cameras))

    if self.use_log_scales:
      params['log_dolly_scale'] = self.param(
          'log_dolly_scale', jax.nn.initializers.zeros, cameras.shape
      )
    else:
      params['dolly_bias'] = self.param(
          'dolly_bias', jax.nn.initializers.zeros, cameras.shape
      )
    return params

  def update_dolly_zoom(
      self,
      camera_params: chex.ArrayTree,
      z: jnp.ndarray,
      focal_length: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies a dolly zoom to the given z-translation and focal length."""
    # Allow moving the camera by a scale and bias.
    if self.use_log_scales:
      new_z = z * jnp.exp(camera_params['log_dolly_scale'])
    else:
      new_z = z + camera_params['dolly_bias']

    # Use the ratio of the updated and original z-translation to preserve the
    # size of the image.
    eps = jnp.finfo(jnp.float32).eps
    new_focal_length = focal_length * (
        new_z.clip(min=eps) / z.clip(min=eps)
    ).squeeze(-1)
    return new_z, new_focal_length

  def transform_camera_focal_pose(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    translation = spin_math.matmul(-camera.orientation, camera.position)
    x, y, z = jnp.split(translation, 3, -1)
    new_x, new_y, new_z, new_focal_length = (
        FocalPoseCameraDelta.update_focal_pose(
            self, camera_params, x, y, z, camera.focal_length
        )
    )
    new_z, new_focal_length = self.update_dolly_zoom(
        camera_params, new_z, new_focal_length
    )
    new_translation = jnp.concatenate([new_x, new_y, new_z], axis=-1)
    new_orientation = FocalPoseCameraDelta.update_orientation(
        self, camera_params, camera.orientation
    )
    new_position = spin_math.matmul(-new_orientation.T, new_translation)

    return camera.replace(
        orientation=new_orientation,
        position=new_position,
        focal_length=new_focal_length,
    )

  def transform_camera_se3(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    camera = SE3WithFocalCameraDelta.transform_camera(
        self, camera_params, camera
    )
    translation = spin_math.matmul(-camera.orientation, camera.position)
    x, y, z = jnp.split(translation, 3, -1)
    new_z, new_focal_length = self.update_dolly_zoom(
        camera_params, z, camera.focal_length
    )
    new_translation = jnp.concatenate([x, y, new_z], axis=-1)
    new_position = spin_math.matmul(-camera.orientation.T, new_translation)
    return camera.replace(
        position=new_position,
        focal_length=new_focal_length,
    )

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    if self.use_se3:
      return self.transform_camera_se3(camera_params, camera)
    return self.transform_camera_focal_pose(camera_params, camera)


@gin.configurable
class IntrinsicDollyCameraDelta(IntrinsicCameraDelta, DollyCameraDelta):
  """Camera delta that also provides principal point variation."""

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    return {
        **IntrinsicCameraDelta.create_params(self, cameras),
        **DollyCameraDelta.create_params(self, cameras),
    }

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    camera = DollyCameraDelta.transform_camera(self, camera_params, camera)
    camera = IntrinsicCameraDelta.transform_camera(self, camera_params, camera)
    return camera


class SCNeRFCameraDelta(CameraDelta):
  """An implementation of the SC-NeRF camera."""

  use_principal_point: bool = True
  use_radial_distortion: bool = True

  def create_params(self, cameras: jaxcam.Camera) -> chex.ArrayTree:
    params = {
        'focal_x_delta': self.param(
            'focal_x_delta', jax.nn.initializers.zeros, cameras.shape
        ),
        'focal_y_delta': self.param(
            'focal_y_delta', jax.nn.initializers.zeros, cameras.shape
        ),
        'rotation_6d_delta': self.param(
            'rotation_6d_delta', jax.nn.initializers.zeros, (*cameras.shape, 6)
        ),
        'translation_delta': self.param(
            'translation_delta', jax.nn.initializers.zeros, (*cameras.shape, 3)
        ),
    }
    if self.use_principal_point:
      params['principal_point_bias'] = self.param(
          'principal_point_bias',
          jax.nn.initializers.zeros,
          (*cameras.shape, 2),
      )
    if self.use_radial_distortion:
      params['radial_distortion_bias'] = self.param(
          'radial_distortion_bias',
          jax.nn.initializers.zeros,
          (*cameras.shape, 2),
      )
    return params

  def transform_camera(
      self, camera_params: chex.ArrayTree, camera: jaxcam.Camera
  ) -> jaxcam.Camera:
    # Insert radial distortion if it doesn't exist.
    if self.use_radial_distortion:
      radial_distortion_bias = camera_params['radial_distortion_bias']
      radial_distortion_bias = jnp.pad(
          radial_distortion_bias,
          pad_width=(0, 4 - jnp.shape(radial_distortion_bias)[-1]),
      )
      if camera.radial_distortion is None:
        camera = camera.replace(radial_distortion=jnp.zeros(*camera.shape, 4))
      camera = camera.replace(
          radial_distortion=camera.radial_distortion + radial_distortion_bias
      )
    if self.use_principal_point:
      principal_point_bias = camera_params['principal_point_bias']
      camera = camera.replace(
          principal_point=camera.principal_point + principal_point_bias,
      )

    new_focal_length_x = camera.scale_factor_x + camera_params['focal_x_delta']
    new_focal_length_y = camera.scale_factor_y + camera_params['focal_y_delta']
    new_pixel_aspect_ratio = new_focal_length_x / new_focal_length_y
    camera = camera.replace(
        focal_length=new_focal_length_x,
        pixel_aspect_ratio=new_pixel_aspect_ratio,
    )

    rotation_6d = rigid_body.ortho6d_from_rotation_matrix(camera.orientation)
    new_rotation_6d = rotation_6d + camera_params['rotation_6d_delta']
    new_orientation = rigid_body.rotation_matrix_from_ortho6d(new_rotation_6d)
    new_translation = camera.translation + camera_params['translation_delta']
    new_position = spin_math.matmul(-new_orientation.T, new_translation)
    return camera.replace(
        orientation=new_orientation,
        position=new_position,
    )


# List here all the camera delta classes so we can list them in
# config_utils.ensure_configurables_registered.
CAMERA_DELTA_CLASSES = (
    SE3CameraDelta,
    SE3WithFocalCameraDelta,
    FocalPoseCameraDelta,
    DollyCameraDelta,
    IntrinsicDollyCameraDelta,
    IntrinsicCameraDelta,
    IntrinsicSE3WithFocalCameraDelta,
    IntrinsicFocalPoseCameraDelta,
    SCNeRFCameraDelta,
)
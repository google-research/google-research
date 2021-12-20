# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Ray Projection."""

from functools import partial

import chex
from einshape import jax_einshape as einshape
import jax
import jax.numpy as jnp

from light_field_neural_rendering.src.utils import data_types


class RayProjector:
  """Class for ray projection."""

  def __init__(self, config):

    self.precision = config.precision
    self.use_pixel_centers = config.use_pixel_centers
    self.min_depth = config.min_depth
    self.max_depth = config.max_depth
    self.image_height = config.image_height
    self.image_width = config.image_width

    self.num_samples = config.num_projections
    self.interpolation_type = config.interpolation_type
    interpolation_fn = lambda x, idx: x[idx[Ellipsis, 0], idx[Ellipsis, 1]]
    self.vmap_interpolation_fn = jax.jit(jax.vmap(interpolation_fn, (0, 0), 0))

  def inside_image(self, pcoords):
    """Function to check if projected coordinates are inside the image."""
    return ((pcoords[Ellipsis, 0] >= 0) *
            (pcoords[Ellipsis, 0] <= self.image_height - 1) *
            (pcoords[Ellipsis, 1] >= 0) * (pcoords[Ellipsis, 1] <= self.image_width - 1))

  def project2camera(self, wcoords, ref_worldtocamera, intrinsic_matrix):
    """** Visually verified in colab.

    Function to project woorld coordinates into all the cameras specified
    by the cameras
    Args:
      wcoords: The woorld coordinates for each ray. (#rays, #points_per_ray, 4)
      ref_worldtocamera: world to camera matrix (#near_cam, 4, 4)
      intriscis_matrix: Intrinsic matrix (3,4)

    Returns:
      pcoords: projected coordinates
      proj_frontof_cam_mask: false if projections are behind camera
    """
    ref_worldtocamera = einshape("nyy->n11yy", ref_worldtocamera)
    wcoords = einshape("bpy->1bpy1", wcoords)
    intrinsic_matrix = einshape("1xy->111xy", intrinsic_matrix)
    kw = jnp.matmul(
        intrinsic_matrix, ref_worldtocamera, precision=self.precision)
    pcoord = jnp.matmul(kw, wcoords, precision=self.precision)[Ellipsis, 0]
    # Find projections that are valid (in front of camera)
    proj_frontof_cam_mask = (pcoord[Ellipsis, -1] < 0
                            )  # Camera point toward negative z axis
    # Normalize for perspective projection
    pcoord = pcoord[Ellipsis, :2] / (-pcoord[Ellipsis, 2:3])
    return pcoord[Ellipsis, [1, 0]], proj_frontof_cam_mask  # Reverse the ordering

  def epipolar_projection(self, key, target_rays, ref_worldtocamera,
                          intrinsic_matrix, randomized):
    """** Visually verified in colab.

    Function to map given rays on the epipolar line on reference view at
    mutiple depth values.
    (The depth value are often set to near and far plane of camera).
    Args:
      key: prngkey
      target_rays: The rays that we want to project onto nearby cameras. Often
        these are the rays that we want to render contains origins, directions
        shape (#bs, rays, 3)
      ref_worldtocamera:(#near_cam, 4,4) The worldtocamera matrix of nearby
        cameras that we want to project onto.
      intrinsic_matrix: (1, 3, 4), The intrinsic matrix for the datset
      randomized: if True, use randomized depths for projection.

    Returns:
      pcoords: (#near_cam, batch_size, num_projections, 2)
      valid_proj_mask: (#near_cam, batch_size, num_projections) specifying with
        of the projections are valid i.e. in front of camera and within image
        bound
      wcoords: (batch_size, num_projections, 3)
    """
    # Check shape of intrincs, currently we only support case where all the
    # views are from the same camera
    chex.assert_shape(intrinsic_matrix, (1, 3, 4))
    #intrinsic_matrix = intrinsic_matrix.squeeze(0)

    projection_depths = jnp.linspace(self.min_depth, self.max_depth,
                                     self.num_samples)

    if randomized:
      mids = .5 * (projection_depths[Ellipsis, 1:] + projection_depths[Ellipsis, :-1])
      upper = jnp.concatenate([mids, projection_depths[Ellipsis, -1:]], -1)
      lower = jnp.concatenate([projection_depths[Ellipsis, :1], mids], -1)
      batch_size = target_rays.batch_shape[0]
      p_rand = jax.random.uniform(key, [batch_size, self.num_samples])
      projection_depths = lower + (upper - lower) * p_rand
    # Compute the woorld coordinates for each ray for each depth values
    # wcoords has shape (#rays, num_projections, 3)
    wcoords = (
        target_rays.origins[:, None] +
        target_rays.directions[:, None] * projection_depths[Ellipsis, None])
    #Convert to homogenous coordinates (#rays, num_samples, 3) -> (#rays, num_samples, 4)
    wcoords = jnp.concatenate(
        [wcoords, jnp.ones_like(wcoords[Ellipsis, 0:1])], axis=-1)
    pcoords, proj_frontof_cam_mask = self.project2camera(
        wcoords, ref_worldtocamera, intrinsic_matrix)

    # Find projections that are inside the image
    within_image_mask = self.inside_image(pcoords)

    # Clip coordinates to be withing the images
    pcoords = jnp.concatenate([
        jnp.clip(pcoords[Ellipsis, 0:1], 0, self.image_height - 1),
        jnp.clip(pcoords[Ellipsis, 1:], 0, self.image_width - 1)
    ],
                              axis=-1)

    # A projection is valid if it is in front of camera and within the image
    # boundaries
    valid_proj_mask = proj_frontof_cam_mask * within_image_mask

    return pcoords, valid_proj_mask, wcoords[Ellipsis, :3]

  def epipolar_projection_given_points(self, wcoords, ref_worldtocamera,
                                       intrinsic_matrix):
    """Function to project world points to nearby views.

    Args:
      wcoords: World points
      ref_worldtocamera:(#near_cam, 4,4) The worldtocamera matrix of nearby
        cameras that we want to project onto.
      intrinsic_matrix: (1, 3, 4), The intrinsic matrix for the datset

    Returns:
      pcoords: (#near_cam, batch_size, num_projections, 2)
      mask: (#near_cam, batch_size, num_projections) specifying with of the
      projections are valid
            i.e. in front of camera and within image bound
      wcoords: (batch_size, num_projections, 3)
    """
    # Check shape of intrincs, currently we only support case where all the
    # views are from the same camera
    chex.assert_shape(intrinsic_matrix, (1, 3, 4))
    #intrinsic_matrix = intrinsic_matrix.squeeze(0)

    #Convert to homogenous coordinates (#rays, num_samples, 3) -> (#rays, num_samples, 4)
    wcoords = jnp.concatenate(
        [wcoords, jnp.ones_like(wcoords[Ellipsis, 0:1])], axis=-1)
    pcoords, proj_frontof_cam_mask = self.project2camera(
        wcoords, ref_worldtocamera, intrinsic_matrix)

    # Find projections that are inside the image
    within_image_mask = self.inside_image(pcoords)

    # Clip coordinates to be withing the images
    pcoords = jnp.concatenate([
        jnp.clip(pcoords[Ellipsis, 0:1], 0, self.image_height - 1),
        jnp.clip(pcoords[Ellipsis, 1:], 0, self.image_width - 1)
    ],
                              axis=-1)

    # A projection is valid if it is in front of camera and within the image
    # boundaries
    valid_proj_mask = proj_frontof_cam_mask * within_image_mask

    return pcoords, valid_proj_mask

  def get_near_rays(self, pcoords, ref_cameratoworld, intrinsic_matrix):
    """** Verified in colab.

    Function to get the rays throught the projected pixel coordinates of
    the neighboring views.

    Args:
      pcoords: (#near_cam, batch_size, num_projections, 2), projected
        coordinates
      ref_cameratoworld:(#near_cam, 3,4) The cameratoworld matrix of nearby
        cameras that we want to generate the rays for.
      intrinsic_matrix: (1, 3, 3), The intrinsic matrix for the datset

    Returns:
      rays: (data_types.Rays), rays through the projected pixel coordinates.
    """
    # Abbreaviations:
    # N: #neighbours, B: batch_size, P: #projections
    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    pcoords = pcoords.round() + pixel_center

    pixels = pcoords[Ellipsis, [
        1, 0
    ]]  # Reverse back for ray generation (we need width dim x height dim)
    pixels = jnp.concatenate([pixels, -jnp.ones_like(pixels[Ellipsis, 0:1])],
                             axis=-1)  # (N, B, P, 3)
    inverse_intrisics = jnp.linalg.inv(
        intrinsic_matrix[Ellipsis, :3, :3])  #(   1, 3, 3)

    # We will change pixels to  shape (N, B, P, 3, 1)
    # and inverse_intrinsics to shape (1, 1, 1, 3, 3)
    # camera_dirs = (inverse_intrisics[None, None, :] @ pixels[..., None]
    #              )  #(N, B, P, 3, 1)
    camera_dirs = jnp.matmul(
        inverse_intrisics[None, None, :],
        pixels[Ellipsis, None],
        precision=self.precision)  # (N, B, P, 3, 1)

    # ref_cameratoworld has shape (N, 3, 4). We only want the rotation part of
    # the cameratoworld matrix to compute the direction of rays.
    # For multiplication we want to change the shape of cameratoworld to be
    # (N, 1, 1, 3, 3) so that we can multiply it with camera_dirs with shape
    # (N, B, P, 3, 1)
    directions = jnp.matmul(
        ref_cameratoworld[:, None, None, :3, :3],
        camera_dirs,
        precision=self.precision)[Ellipsis, 0]

    # We extract the origns from last colum of ref_cameratoworld which will have
    # as shape of (N, 3) whihch we will boadcast to (N, B, P, 3)
    origins = jnp.broadcast_to(ref_cameratoworld[:, None, None, :3, -1],
                               directions.shape)
    # We traspose the matrix such that batch dimension is the leading
    # dimension (N, B, P, 3) -> (B, N, P, 3)
    origins = einshape("nbpc->bnpc", origins)
    directions = einshape("nbpc->bnpc", directions)
    viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    near_rays = data_types.Rays(origins=origins, directions=viewdirs)
    return near_rays

  def get_interpolated_rgb(self, pcoords, ref_images):
    """Function to get the interpolated rgb values at the projected pixel coordinates.

    Args:
      pcoords (jnp.float) : (#near_cam, #batch_size, #projections, 2)
      ref_images (jnp.float) : (#near_cam, height, width, 3)

    Returns:
      near_rgb(jnp.float) : interpolated rgb values for shape (#batch_size,
        #near_cam, #projections, 3)
    """
    # Abbreaviations:
    # N: #neighbours, B: batch_size, P: #projections
    if self.interpolation_type == "rounding":
      pcoords = pcoords.round().astype(jnp.int32)
      near_rgb = self.vmap_interpolation_fn(ref_images, pcoords)
      # Tranpose such that batch is the leading dimension
      near_rgb = einshape("nbpc->bnpc", near_rgb)
    else:
      raise NotImplementedError

    return near_rgb

  def get_interpolated_features(self, pcoords, image_features):
    """Function to get the interpolated feature values at the projected pixel coordinates.

    This function is identical to get_interpolated_rgb.

    We write a different function to handle future applications that might
    require treating rgb and pixels differently.
    Args:
      pcoords (jnp.float) : (#near_cam, #batch_size, #projections, 2)
      image_features (jnp.float): (#near_cam, height, width, unet_dim)

    Returns:
      projected_features(jnp.float) : interpolated features for shape
        (#batch_size, #near_cam, #projections, unet_dim)
    """
    # Abbreaviations:
    # N: #neighbours, B: batch_size, P: #projections
    if self.interpolation_type == "rounding":
      pcoords = pcoords.round().astype(jnp.int32)
      projected_features = self.vmap_interpolation_fn(image_features, pcoords)
      # Tranpose such that batch is the leading dimension
      projected_features = einshape("nbpc->bnpc", projected_features)
    else:
      raise NotImplementedError

    return projected_features

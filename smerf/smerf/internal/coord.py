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

"""Tools for manipulating coordinate spaces and distances along rays."""

import chex
import gin
import jax
import jax.numpy as jnp
import numpy as np


def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = 2 ** jnp.arange(min_deg, max_deg)
  shape = x.shape[:-1] + (-1,)
  scaled_x = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]), shape)
  # Note that we're not using safe_sin, unlike IPE.
  four_feat = jnp.sin(
      jnp.concatenate([scaled_x, scaled_x + 0.5 * jnp.pi], axis=-1)
  )
  if append_identity:
    return jnp.concatenate([x] + [four_feat], axis=-1)
  else:
    return four_feat


def piecewise_warp_fwd(x, eps=jnp.finfo(jnp.float32).eps):
  """A piecewise combo of linear and reciprocal to allow t_near=0."""
  return jnp.where(x < 1, 0.5 * x, 1 - 0.5 / jnp.maximum(eps, x))


def piecewise_warp_inv(x, eps=jnp.finfo(jnp.float32).eps):
  """The inverse of `piecewise_warp_fwd`."""
  return jnp.where(x < 0.5, 2 * x, 0.5 / jnp.maximum(eps, 1 - x))


def world_to_submodel(sm_idxs, t, config, grid_config):
  """Converts from world coordinates to submodel coordinates."""
  sm_idxs = jnp.broadcast_to(sm_idxs, t[Ellipsis, :1].shape)
  scale = grid_config['submodel_scale']
  offset = sm_center(sm_idxs, config, grid_config)
  return (t - offset) * scale


def submodel_to_world(sm_idxs, sm, config, grid_config):
  """Converts from submodel coordinates to world coordinates."""
  sm_idxs = jnp.broadcast_to(sm_idxs, sm[Ellipsis, :1].shape)
  scale = grid_config['submodel_scale']
  offset = sm_center(sm_idxs, config, grid_config)
  return sm / scale + offset


def world_to_deferred_mlp(sm_idxs, t, config, grid_config):
  """Converts from world coordinates to DeferredMLP coordinates."""
  # Unlike world_to_submodel(), this function does not take
  # submodel_scale_factor into account.
  sm_idxs = jnp.broadcast_to(sm_idxs, t[Ellipsis, :1].shape)
  scale = config.submodel_grid_resolution
  offset = sm_center(sm_idxs, config, grid_config)
  return (t - offset) * scale


def world_dist_to_sm_dist(sm_idxs, world_dist, config, grid_config):
  """Converts distances in world coordinates to submodel coordinates."""
  del sm_idxs, config
  scale = grid_config['submodel_scale']
  return world_dist * scale


def sm_dist_to_world_dist(sm_idxs, sm_dist, config, grid_config):
  """Converts distances in submodel coordinates to world coordinates."""
  del sm_idxs, config
  scale = grid_config['submodel_scale']
  return sm_dist / scale


def sdist_to_tdist(sdist, t_near, t_far):
  """Convert squash distances ([0,1]) to submodel distances ([t_near, t_far])."""
  s_near, s_far = [piecewise_warp_fwd(x) for x in (t_near, t_far)]
  return piecewise_warp_inv(sdist * s_far + (1 - sdist) * s_near)


def tdist_to_sdist(smdist, t_near, t_far):
  """Inverse of sdist_to_tdist."""
  fn = piecewise_warp_fwd
  s_near, s_far = fn(t_near), fn(t_far)
  return (piecewise_warp_fwd(smdist) - s_near) / (s_far - s_near)


def rays_to_sm_idxs(rays, config, grid_config, ignore_override=False):  # pylint: disable=unused-argument
  """Generates submodel indices given ray origins in world coordinates.

  Args:
    rays: Rays with origins of shape f32[..., 3]. XYZ position of ray's origin
      in world coordinates.  Values in [-1, 1].
    config: ...
    grid_config: ...
    ignore_override: if True, ignore grid_config['submodel_idx_override']

  Returns:
    sm_idxs: i32[..., 1]. Submodel index for each ray.
  """
  if ignore_override or grid_config['submodel_idx_override'] is None:
    # Derive submodel index from camera origin.
    voxel_resolution = config.submodel_grid_resolution
    x_grid = (rays.origins + 1.0) / 2.0  # Values in [0, 1)
    x_grid = x_grid * voxel_resolution  # Values in [0, voxel_resolution)
    x_floor = jnp.floor(x_grid).astype(jnp.int32)

    # ray_origins outside of [-1, 1] box are assigned to their nearest voxel.
    x_floor = jnp.clip(x_floor, 0, voxel_resolution - 1)

    sm_idxs = coord_to_index(
        x_floor, (voxel_resolution, voxel_resolution, voxel_resolution)
    )
  else:
    # If this host is responsible for a single submodel, assign all rays to the
    # requested submodel.
    sm_idx = grid_config['submodel_idx_override']
    sm_idxs = jnp.full(rays.origins.shape[:-1], sm_idx, dtype=jnp.int32)

  sm_idxs = sm_idxs[Ellipsis, None]  # Insert trailing length-1 dim.
  return sm_idxs


@gin.configurable(allowlist=['method'])
def contract(x, method='merf'):
  if method == 'mipnerf360':
    return contract_mipnerf360(x)
  elif method == 'merf':
    return contract_merf(x)
  raise NotImplementedError(method)


@gin.configurable(allowlist=['method'])
def uncontract(x, allow_inf=True, method='merf'):
  if method == 'mipnerf360':
    return uncontract_mipnerf360(x)
  elif method == 'merf':
    return uncontract_merf(x, allow_inf=allow_inf)
  raise NotImplementedError(method)


def contract_mipnerf360(x):
  """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
  # Clamping to 1 produces correct scale inside |x| < 1
  x_mag_sq = jnp.maximum(1, jnp.sum(x**2, axis=-1, keepdims=True))
  scale = (2 * jnp.sqrt(x_mag_sq) - 1) / x_mag_sq
  z = scale * x
  return z


def uncontract_mipnerf360(z):
  """The inverse of contract()."""
  # Clamping to 1 produces correct scale inside |z| < 1
  z_mag_sq = jnp.maximum(1, jnp.sum(z**2, axis=-1, keepdims=True))
  inv_scale = 2 * jnp.sqrt(z_mag_sq) - z_mag_sq
  x = z / inv_scale
  return x


def contract_merf(x):
  """The contraction function we proposed in MERF.

  Args:
    x: f32[..., 3]. Positions in submodel coordinates. Values in (-inf, inf)

  Returns:
    y: f32[..., 3]. Positions in squash coordinates. Values in (-2, 2).
  """
  # For more info check out MERF: Memory-Efficient Radiance Fields for Real-time
  # View Synthesis in Unbounded Scenes: https://arxiv.org/abs/2302.12249,
  # Section 4.2.
  x_abs = jnp.abs(x)

  # scale = ||x||_{\inf} or 1.0, whichever is larger.
  x_max = jnp.amax(x_abs, axis=-1, keepdims=True)
  scale = jnp.maximum(1, x_max)

  # Determine which coordinate has the largest magnitude.
  # Multiple coordinates may share the same magnitude.
  is_largest_coordinate = x_abs == x_max

  # Overwrite the largest coordinate in each vector.
  # When scale=1.0, this is no-op.
  y = jnp.where(
      is_largest_coordinate > 0,
      (x / scale) * (2.0 - 1.0 / scale),
      x / scale,
  )

  return y


def uncontract_merf(y, allow_inf=True):
  """Inverse of `contract`.

  WARNING!! This function is not numerically stable at float32 precision.
  Expect large errors as ||y||_{inf} approaches 2.0.

  This function will not produce NaNs.

  Args:
    y: f32[..., 3]. Positions in squash coordinates. Values in [-2, 2].
    allow_inf: bool. If True, allow infinitely large outputs when y == 2.

  Returns:
    x: f32[..., 3]. Positions in submodel coordinates. Values in [-inf, inf]
  """
  # If a coordinate's magnitude is exactly 2.0, pull it away from the boundary.
  if not allow_inf:
    almost_two = np.nextafter(2., 1., dtype=y.dtype)
    y = jnp.where(jnp.abs(y) >= 2.0, jnp.sign(y) * almost_two, y)

  # contract() doesn't change sign
  y_abs = jnp.abs(y)

  # scale = ||x||_{\inf} or 1.0, whichever is larger.
  x_max = jnp.amax(1.0 / (2.0 - y_abs), axis=-1, keepdims=True)
  scale = jnp.maximum(1.0, x_max)

  # Which coordinate has the largest magnitude?
  y_max = jnp.amax(y_abs, axis=-1, keepdims=True)
  is_largest_coordinate = y_abs == y_max

  # Correct coordinate with the largest magnitude if said coordinate had a
  # magnitude > 1.
  # When scale=1.0, this is a no-op.
  x = jnp.where(
      is_largest_coordinate > 0,
      (y / (2.0 - 1.0 / scale)) * scale,
      y * scale,
  )

  # Avoid nans resulting from `0 * inf`.
  x = jnp.where(y == 0, 0.0, x)

  return x


def t_stepsize_from_s_stepsize(
    t_positions, t_viewdirs, sm_idxs, s_stepsizes, config, grid_config
):
  """Computes step size given a target distance in squash coordinates.

  Step size is given in world coordinates.

  Args:
    t_positions: f32[..., 3]. Position in world coordinates.
    t_viewdirs: f32[..., 3]. Unit-norm direction in submodel coordinates.
    sm_idxs: f32[..., 1]. Which submodel each ray is assigned to.
    s_stepsizes: f32[...]. Target distance in squash coordinates.
    config: ...
    grid_config: ...

  Returns:
    t_stepsizes: f32[...]. Approximate distance in world coordinates
      corresponding to target distance in squash coordinates.
  """
  sm_positions = world_to_submodel(
      sm_idxs, t_positions, config, grid_config
  )  # f32[..., 3]
  sm_viewdirs = t_viewdirs  # f32[..., 3]
  sm_step_size = sm_stepsize_from_s_stepsize(
      sm_positions,
      sm_viewdirs,
      s_stepsizes,
  )  # f32[...]
  t_step_sizes = sm_dist_to_world_dist(
      sm_idxs, sm_step_size, config, grid_config
  )  # f32[...]
  return t_step_sizes


def sm_stepsize_from_s_stepsize(sm_positions, sm_viewdirs, s_stepsizes):
  """Computes step size given a target distance in squash coordinates.

  Step size is given in submodel coordinates.

  Args:
    sm_positions: f32[..., 3]. Position in submodel coordinates.
    sm_viewdirs: f32[..., 3]. Unit-norm direction in submodel coordinates.
    s_stepsizes: f32[...]. Target distance in squash coordinates.

  Returns:
    sm_stepsizes: f32[...]. Approximate distance in submodel coordinates
      corresponding to target distance in squash coordinates.
  """
  # Approximately computes s such that ||c(x+d*s) - c(x)||_2 = v, where c is
  # the contraction function, i.e. we often need to know by how much (s) the ray
  # needs to be advanced to get an advancement of v in contracted space.
  #
  # The further we are from the scene's center the larger steps in world space
  # we have to take to get the same advancement in contracted space.
  #
  # TODO(duckworthd): Consider other approaches to estimating this quantity.
  # For example, binary search in submodel coordinates
  #
  # TODO(duckworthd): Consider doing some sort of search or optimization to
  # find a more accurate step size estimate.
  batch_dims = sm_positions.shape[:-1]
  chex.assert_equal_shape((sm_positions, sm_viewdirs))
  chex.assert_shape((sm_positions, sm_viewdirs), (Ellipsis, 3))
  s_stepsizes = jnp.broadcast_to(s_stepsizes, batch_dims)

  sm_positions = jnp.reshape(sm_positions, (-1, 3))
  sm_viewdirs = jnp.reshape(sm_viewdirs, (-1, 3))
  s_stepsizes = jnp.reshape(s_stepsizes, (-1,))

  result = s_stepsizes / contract_stepsize_v2(sm_positions, sm_viewdirs)

  result = jnp.reshape(result, batch_dims)
  return result


@jax.vmap
def contract_stepsize(x, d):
  """Magnitude of the derivative of contract(x) in direction d.

  Assigns each position/direction pair a scalar describing how much
  contract(x) changes when moving in direction d. This scalar is 1.0 in
  uncontracted space and decreases as we move farther and farther into
  squash space. This value is poorly defined at non-smooth points in the
  contract().

  Args:
    x: f32[B, 3]. Position in submodel coordinates.
    d: f32[B, 3]. Direction in submodel coordinates.

  Returns:
    f32[B]. Equal to ||contract(x+td) - contract(x)|| / t as t goes to zero.
  """
  assert x.shape[-1] == d.shape[-1] == 3
  assert len(x.shape) == len(d.shape) == 1  # inside vmap, batch dim is gone.

  contract_0_grad = jax.grad(lambda x: contract(x)[0])
  contract_1_grad = jax.grad(lambda x: contract(x)[1])
  contract_2_grad = jax.grad(lambda x: contract(x)[2])

  return jnp.sqrt(
      d.dot(contract_0_grad(x)) ** 2
      + d.dot(contract_1_grad(x)) ** 2
      + d.dot(contract_2_grad(x)) ** 2
  )


@jax.jit
def contract_stepsize_v2(x, d):
  """Magnitude of the gradient of contract(x) in direction d.

  See docstring for contract_stepsize(). This implementation produces the same
  outputs with 40x less compute.

  Args:
    x: f32[B, 3]. Position in submodel coordinates.
    d: f32[B, 3]. Direction in submodel coordinates.

  Returns:
    f32[B]. Equal to ||contract(x+td) - contract(x)|| / t as t goes to zero.
  """
  chex.assert_rank((x, d), 2)
  n, _ = x.shape
  chex.assert_shape((x, d), (n, 3))
  _, tangents = jax.jvp(contract, (x,), (d,))
  stepsizes = jnp.linalg.norm(tangents, axis=-1)
  return stepsizes


def coord_to_index(coords, shape):
  """Converts n-dim array coordinates to linear coordinates.

  Args:
    coords: i32[..., k]. Coordinates.
    shape: tuple[int, ...] with k entries. Shape of array being indexed into.

  Returns:
    idxs: i32[...]. Linear index of each coordinate.
  """
  assert coords.shape[-1] == len(shape), f'{coords.shape[-1]} != {len(shape)}'
  result = jnp.zeros(coords.shape[:-1], dtype=jnp.int32)
  spacing = 1
  for i, dim in reversed(list(enumerate(shape))):
    result = result + coords[Ellipsis, i] * spacing
    spacing = spacing * dim
  return result


def index_to_coord(idxs, shape):
  """Converts linear coordinates to n-dim array coordinates.

  Args:
    idxs: i32[...]. Linear index of each coordinate.
    shape: tuple[int, ...] with k entries. Shape of array being indexed into.

  Returns:
    coords: i32[..., k]. n-dim coordinate of each linear index.
  """
  # Calculate number of values per dimension.
  divisors = []
  spacing = 1
  for dim in reversed(shape):
    divisors.append(spacing)
    spacing = spacing * dim
  divisors = list(reversed(divisors))

  # Calculate n-dim index.
  result = []
  remainder = idxs
  for _, div in enumerate(divisors):
    coord, remainder = jnp.divmod(remainder, div)
    result.append(coord.astype(jnp.int32))
  return jnp.stack(result, axis=-1)


def sm_center(sm_idxs, config, grid_config):
  """Calculates center of a submodel's coordinate system in t-coordinates.

  Args:
    sm_idxs: i32[..., 1]. Submodel index for each ray.
    config: ...
    grid_config: ...

  Returns:
    centers: f32[..., 3]. XYZ position of submodel coordinate system's center in
      t-coordinates.
  """
  voxel_resolution = config.submodel_grid_resolution
  voxel_size = grid_config['submodel_voxel_size']
  # IJK index for submodels.
  submodel_coord = index_to_coord(
      sm_idxs[Ellipsis, 0],
      (
          voxel_resolution,
          voxel_resolution,
          voxel_resolution,
      ),  # Values in [0, voxel_resolution)
  )
  # XYZ center for submodels
  x = submodel_coord / voxel_resolution  # [0, 1)
  x = 2 * x - 1  # [-1, 1)
  x = x + voxel_size / 2  # half-voxel offset
  return x


def sm_idxs_to_params_idxs(sm_idxs, config, grid_config):
  """Maps from submodel idxs to submodel parameter idxs.

  Whereas sm_idxs are used for choosing coordinate systems, params_idxs are
  used for choosing submodel parameters. When all submodels are instantiated,
  these are one and the same.

  Args:
    sm_idxs: i32[..., 1]. Submodel indices.
    config: configs.Config instance.
    grid_config: See grid_utils.initialize_grid_config()

  Returns:
    sparse_sm_idxs: i32[..., 1]. Sparse submodel indices.
  """
  del config
  if grid_config['submodel_idx_override'] is not None:
    # All rays are assigned to the one and only submodel on this host.
    sparse_sm_idxs = jnp.zeros_like(sm_idxs)
  else:
    # Only a subset of submodels are instantiated.
    sm_to_params = jnp.array(grid_config['sm_to_params'])  # i32[K]
    sparse_sm_idxs = sm_to_params[sm_idxs[Ellipsis, 0]]  # i32[...]
    sparse_sm_idxs = sparse_sm_idxs[Ellipsis, jnp.newaxis]  # i32[..., 1]
  assert sm_idxs.shape == sparse_sm_idxs.shape
  return sparse_sm_idxs


def distance_to_submodel_volumes(
    t_positions, sm_idxs, config, grid_config, ord=jnp.inf  # pylint: disable=redefined-builtin
):
  """Measures distance between world positions and submodel centers.

  Distances are measured in world coordinate units.

  Args:
    t_positions: f32[..., 3]. Positions in world coordinates.
    sm_idxs: f32[k, 1]. Which submodels to measure distances to.
    config: ...
    grid_config: ...
    ord: Which norm to use. Defaults to inf-norm.

  Returns:
    f32[..., k]. Distance between each position and submodel center.
  """
  chex.assert_shape(sm_idxs, (None, 1))
  chex.assert_shape(t_positions, (Ellipsis, 3))
  batch_dims = t_positions.shape[:-1]
  k = sm_idxs.shape[0]
  t_positions = jnp.broadcast_to(
      t_positions[Ellipsis, jnp.newaxis, :], (*batch_dims, k, 3)
  )  # f32[..., k, 3]

  # Find center of target submodel volume in world coordinates.
  center = sm_center(sm_idxs, config, grid_config)  # f32[k, 3]

  # Calculate distance from each position to center. Distances measured in
  # world-coordinate units.
  distances = jnp.linalg.norm(
      t_positions - center, ord=ord, axis=-1
  )  # f32[..., k]

  return distances


def nearest_submodel(t_positions, config, grid_config):
  """Finds closest submodel idx for each t_position.

  Only submodels on this host are eligible.

  Args:
    t_positions: f32[..., 3]. Positions in world coordinates.
    config: ...
    grid_config: ...

  Returns:
    i32[..., 1]. submodel index whose center is closest to each t_position
      according to the inf norm.
  """
  chex.assert_shape(t_positions, (Ellipsis, 3))
  sm_idxs = jnp.array(grid_config['submodels_on_host']).reshape(-1, 1)
  distances = distance_to_submodel_volumes(
      t_positions, sm_idxs, config, grid_config, ord=jnp.inf
  )  # f32[..., k]
  result = jnp.argmin(distances, axis=-1)  # f32[...]
  result = sm_idxs[result]  # i32[..., 1]

  batch_dims = t_positions.shape[:-1]
  chex.assert_shape(result, (*batch_dims, 1))

  return result


def is_home_or_neighbor_submodel_volume(
    t_positions, sm_idxs, config, grid_config
):
  """Calculates neighboring subvolumes to each point.

  This function is well-defined for points in the [-1, 1]^3 cube.

  Args:
    t_positions: f32[..., 3]. Positions in world coordinates.
    sm_idxs: f32[k, 1]. Which submodels to measure distances to.
    config: ...
    grid_config: ...

  Returns:
    is_home_subvolume: bool[..., k]. True if a subvolume is the home subvolume
      for a point.
    is_neighbor_submodel_volume: bool[..., k]. True if a subvolume is a
      neighbor to a point.
  """

  # Calculate distance to all subvolumes.
  distances = distance_to_submodel_volumes(
      t_positions, sm_idxs, config, grid_config, ord=jnp.inf
  )

  # A subvolume is a "neighbor" if it's ADJACENT to a point's home
  # subvolume but is not the home subvolume itself.
  #
  # A point that straddles the boundary between two or more subvolumes does not
  # have a home subvolume.
  #
  # Subvolumes are assumed to be cubes with equal-length sides.
  is_home_subvolume = distances < (grid_config['submodel_voxel_size'] / 2)
  is_neighbor_submodel_volume = (~is_home_subvolume) & (
      distances < grid_config['submodel_voxel_size']
  )

  batch_dims = t_positions.shape[:-1]
  k, _ = sm_idxs.shape
  chex.assert_shape(t_positions, (*batch_dims, 3))
  chex.assert_shape(
      (is_home_subvolume, is_neighbor_submodel_volume), (*batch_dims, k)
  )

  return is_home_subvolume, is_neighbor_submodel_volume


def sample_random_choice_from_masks(rng, first_choice, second_choice):
  """Randomly samples "True" indices from a boolean array.

  Samples one of k "True" values per row from first_choice. If all of
  first_choice is empty, sample a value from second_choice. If second choice
  is also empty, sample uniformly at random.

  Args:
    rng: jax.random.PRNGKey.
    first_choice: bool[..., k]. Boolean mask. Only values with True entries
      are eligible to be sampled, except when all entries are False.
    second_choice: bool[..., k]. Boolean mask. Secondary set of preferred
      entries if first_choice is empty.

  Returns:
    i32[...] with values in {0, ..., k-1}. Randomly sampled indices.
  """
  logits = jnp.full(first_choice.shape, -1e3, dtype=jnp.float32)
  logits = jnp.where(second_choice, 0.0, logits)
  logits = jnp.where(first_choice, 1e3, logits)
  return jax.random.categorical(rng, logits, axis=-1)


def sample_random_neighbor_sm_idx(
    rng, t_positions, config, grid_config, include_home=False
):
  """Samples the sm_idx of a neighboring subvolume.

  Each position in world coordinates is assigned to a single subvolume. This
  function samples a random NEIGHBORING subvolume that's present on this host.

  If a point does not have a neighboring subvolume, an arbitrary one is chosen.

  Args:
    rng: jax.random.PRNGKey.
    t_positions: f32[..., 3]. Positions in world coordinates. Used to decide
      which subvolumes are "neighbors".
    config: ...
    grid_config: ...
    include_home: If True, home subvolume is also eligible for sampling.

  Returns:
    i32[..., 1]. The sm_idx of a neighboring subvolume.
  """
  sm_idxs = jnp.array(grid_config['submodels_on_host']).reshape(-1, 1)  # [k, 1]
  is_home, is_neighbor = is_home_or_neighbor_submodel_volume(
      t_positions, sm_idxs, config, grid_config
  )  # bool[..., k]

  if include_home:
    is_neighbor = jnp.logical_or(is_neighbor, is_home)

  batch_dims = t_positions.shape[:-1]
  k = sm_idxs.shape[0]
  chex.assert_shape(is_neighbor, (*batch_dims, k))

  neighbor_idxs = sample_random_choice_from_masks(
      rng, is_neighbor, is_home
  )  # i32[...]
  neighbor_sm_idxs = sm_idxs[neighbor_idxs]  # i32[..., 1]
  chex.assert_shape(neighbor_sm_idxs, (*batch_dims, 1))

  return neighbor_sm_idxs

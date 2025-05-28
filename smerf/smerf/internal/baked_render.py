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

"""Rendering from the baked representation.

This serves to verify that their is only a minimal discrepancy between rendering
from the model used during training and the style of rendering employed by the
webviewer. The main difference between real-time rendering and training-style
rendering is the employed sampling scheme. During training we use hierarchical
sampling while during rendering we use uniform sampling with empty space
skipping.
"""
# pylint: disable=logging-fstring-interpolation

from collections import defaultdict  # pylint: disable=g-importing-member
import functools
import gc
import os

from absl import logging
import chex
import flax
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from smerf.internal import baking
from smerf.internal import coord
from smerf.internal import datasets
from smerf.internal import grid_utils
from smerf.internal import image
from smerf.internal import math
from smerf.internal import models
from smerf.internal import quantize
from smerf.internal import render
from smerf.internal import utils
from smerf.internal.mock.concurrent import parallel


# Filename for baked metrics
METRICS_JSON_FILE_NAME = 'metrics.test.json'

# Names for baked metadata fields.
BAKED_METADATA_NAMES = ['sm_idx', 'cam_idx']

# Names for baked metrics fields.
BAKED_METRIC_NAMES = [
    'gt.baked.psnr',
    'gt.baked.ssim',
    'gt.baked.lpips_spin',
    'gt.baked.lpips_m360',
]

# Names for metrics fields produced by MetricHarness. Order and length must
# match BAKED_METRIC_NAMES.
METRIC_NAMES = ['psnr', 'ssim', 'lpips_spin', 'lpips_m360']


@jax.named_scope('_generate_sample_positions')
def _generate_sample_positions(
    t_origins,
    t_directions,
    t_viewdirs,
    sm_idx,
    num_samples,
    step_size_squash,
    config,
    grid_config,
):
  """Generates sample points along the ray.

  Takes equal steps in squash coordinates to generate positions in submodel
  coordinates along a camera ray. Samples may be repeated if too many steps
  are taken.

  Args:
    t_origins: f32[..., 3]. Ray origins in world coordinates.
    t_directions: f32[..., 3]. Ray directions in world coordinates.
    t_viewdirs: f32[..., 3]. Unit-norm ray directions in world coordinates.
    sm_idx: int. Which submodel to use.
    num_samples: int. Number of samples per ray.
    step_size_squash: f32[]. Step size in squash coordinates.
    config: ...
    grid_config: ...

  Returns:
    t_positions: f32[..., num_samples, 3]. Samples in world coordinates.
    sm_positions: f32[..., num_samples, 3]. Samples in submodel coordinates.
    s_positions: f32[..., num_samples, 3]. Samples in squash coordinates.
  """
  t_ray_origins = t_origins + config.near * t_directions

  sm_origins = coord.world_to_submodel(
      sm_idxs=sm_idx,
      t=t_ray_origins,
      config=config,
      grid_config=grid_config,
  )  # Rx3

  sm_positions = _generate_sm_positions(
      sm_origins=sm_origins,
      sm_viewdirs=t_viewdirs,  # Use unit-norm directions
      num_samples=num_samples,
      step_size_squash=step_size_squash,
      sm_idx=sm_idx,
      config=config,
      grid_config=grid_config,
  )  # RxSx3

  # Constructs positions in world coordinates.
  t_positions = coord.submodel_to_world(
      sm_idx, sm_positions, config, grid_config
  )  # RxSx3

  # Constructs positions in squash coordinates
  s_positions = coord.contract(sm_positions)  # RxSx3

  return t_positions, sm_positions, s_positions


@jax.named_scope('_generate_sm_positions')
def _generate_sm_positions(
    sm_origins,
    sm_viewdirs,
    num_samples,
    step_size_squash,
    sm_idx,
    config,
    grid_config,
):
  """Generates sample points along the ray in submodel coordinates.."""
  # Make sure that sample points are `step_size_contracted` apart from each
  # other in contracted space. When setting `step_size_contracted` to the size
  # of a voxel every voxel is sampled once. This mimics the behaviour of the
  # real-time renderer.
  #
  # The actual step size (the distance between sampling points in world space)
  # depends on how deep we are in the contracted space. The higher the magnitude
  # of an input point the larger the steps are becoming, i.e. we sample more
  # sparsely the further away we get from the scene's center.
  batch_shape = sm_origins.shape[:-1]
  sm_origins_flat = jnp.reshape(sm_origins, (-1, 3))
  sm_viewdirs_flat = jnp.reshape(sm_viewdirs, (-1, 3))

  def f(x, _):
    x_t = x
    not_too_big = jnp.linalg.norm(x, axis=-1, keepdims=True) < 10e5
    sm_step_size = coord.sm_stepsize_from_s_stepsize(
        x, sm_viewdirs_flat, step_size_squash
    )
    x = jnp.where(
        not_too_big, x + sm_viewdirs_flat * sm_step_size[:, None], x
    )
    return (x, x_t)

  # Constructs query points in submodel coordinates.
  sm_positions_flat = jax.lax.scan(
      f, sm_origins_flat, xs=None, length=num_samples, unroll=1
  )[1].transpose(1, 0, 2)
  sm_positions = sm_positions_flat.reshape(*batch_shape, num_samples, 3)
  return sm_positions


@jax.named_scope('_generate_step_sizes')
def _generate_step_sizes(xs):
  """Generates step sizes in world coordinates.

  Args:
    xs: f32[..., S, 3]. Positions in world coordinates along camera rays.

  Returns:
    f32[..., S, 1]. Distance between adjacent pairs of points along camera
      rays. The final distance is 1e10.
  """
  # Calculate step_size[t] = dist(xs[t+1], xs[t]) for all adjacent
  # pairs of points along each camera ray. Distances are in world coordinates.
  deltas = xs[Ellipsis, 1:, :] - xs[Ellipsis, :-1, :]  # Rx(S-1)x3
  assert deltas.shape[-2:] == (
      xs.shape[-2] - 1,  # S-1
      xs.shape[-1],  # 3
  )
  # t_step_sizes[..., i] = distance from xs[..., i] to xs[..., i+1].
  step_sizes = jnp.linalg.norm(deltas, axis=-1)[Ellipsis, None]  # Rx(S-1)x1
  step_sizes = jnp.concatenate(
      [step_sizes, 1e10 * jnp.ones_like(step_sizes[Ellipsis, :1, :])], axis=1
  )  # RxSx1

  return step_sizes


@jax.named_scope('_generate_triplane_positions')
def _generate_triplane_positions(s_positions, grid_config):
  """Generates voxel positions for triplane representation."""
  triplane_voxel_size = grid_config['triplane_voxel_size']
  triplane_positions = grid_utils.world_to_grid(
      s_positions, triplane_voxel_size, jnp
  )  # RxSx3
  return triplane_positions


@jax.named_scope('_generate_sparse_grid_positions')
def _generate_sparse_grid_positions(
    s_positions, sparse_grid_block_indices, config, grid_config
):
  """Calculates sparse_grid representation for each s_position."""
  sparse_grid_resolution = config.sparse_grid_resolution
  sparse_grid_voxel_size = grid_config['sparse_grid_voxel_size']
  data_block_size = config.data_block_size

  # Positions in sparse grid voxel coordinates. Includes half-voxel offset.
  sparse_grid_positions = grid_utils.world_to_grid(
      s_positions, sparse_grid_voxel_size, jnp
  )  # RxSx3

  # Index into sparse_grid_block_indices for each s_position. This will tell
  # us which macroblock to use.
  atlas_grid_idxs = (sparse_grid_positions / data_block_size).astype(
      jnp.uint32
  )  # RxSx3

  # Which macroblock to use for each s_position.
  macroblock_idxs = sparse_grid_block_indices[
      atlas_grid_idxs[Ellipsis, 0],
      atlas_grid_idxs[Ellipsis, 1],
      atlas_grid_idxs[Ellipsis, 2],
  ]  # RxSx3

  # Set macroblock index to -1 if the sparse grid position is invalid.
  def is_in_bounds(x):
    """True if an index is in the bounds of the voxel grid."""
    # TODO(duckworthd): Understand how epsilon is chosen.
    epsilon = 0.1
    buffer = 0.5 + epsilon
    lower = jnp.all(x >= 0 + buffer, axis=-1)
    upper = jnp.all(x < sparse_grid_resolution - buffer, axis=-1)
    return lower & upper

  macroblock_idxs = jnp.where(
      is_in_bounds(sparse_grid_positions)[Ellipsis, jnp.newaxis],
      macroblock_idxs,  # RxSx3
      -1 * jnp.ones_like(macroblock_idxs),  # RxSx3
  )  # RxSx3

  return sparse_grid_positions, atlas_grid_idxs, macroblock_idxs


@jax.named_scope('_generate_is_occupied')
def _generate_is_occupied(s_positions, occupancy_grid, occupancy_voxel_size):
  """Calculates which s_positions lie in an occupied voxel."""
  occupancy_positions = grid_utils.world_to_grid(
      s_positions, occupancy_voxel_size, jnp
  )  # RxSx3

  # TODO(duckworthd): Understand why jnp.round() is used.
  occupancy_indices = jnp.round(occupancy_positions).astype(jnp.uint32)

  is_occupied = occupancy_grid[
      occupancy_indices[Ellipsis, 0],
      occupancy_indices[Ellipsis, 1],
      occupancy_indices[Ellipsis, 2],
  ][Ellipsis, None]  # RxSx1
  return is_occupied


@jax.named_scope('_first_m_hits_per_ray')
def _first_m_hits_per_ray(
    variables,
    is_occupied,
    max_steps,
):
  """Finds values for the first 'max_steps' hits per ray.

  Args:
    variables: PyTree of arrays of shape [R, N, ?]. Values for N values along
      R camera rays.
    is_occupied: bool[R, N, 1]. True if this sample intersects an occupied
      voxel.
    max_steps: int. Maximum number of hits per ray. Referred to as 'M' below.
      Only the first M occupied values per ray are returned.

  Returns:
    variables: PyTree of arrays of shape [R, M, ?]. First M entries per ray
      where is_occupied[r, m] = True.
    is_placeholder: bool[R, M]. True if a value should be ignored.
  """
  # Let R = num_rays, N = num_steps, and M = max_steps
  # We want to find the first M hits per ray out of N sampling points.
  # Let's say our ray has h hits. There are two cases to consider:
  #   case A: h >= M
  #   case B: h < M
  # case A is easy to deal with since keeping the first M hits per ray will
  # result in a subarray of exactly length M (which is not true for case B)
  # Therefore we want to eliminate case B and append M "placeholder" hits to
  # each ray. As a result there will be h+M hits per ray and h+M >= M (case A).
  num_rays, num_steps, _ = is_occupied.shape

  # We append M placeholder entries to the end of each array: RxNxC) -->
  # Rx(N+M)xC. We append M ones to occupancy_grid, and the value 1 indicates
  # "occupied" to create the placeholder hits. Likewise we append non-negative
  # block_indices which encodes non-empty space.
  def append_placeholders(x, fill_value):
    p = jnp.full((num_rays, max_steps, x.shape[-1]), fill_value, dtype=x.dtype)
    return jnp.concatenate([x, p], axis=1)

  variables = jax.tree_util.tree_map(
      lambda x: append_placeholders(x, 0), variables
  )
  is_occupied = append_placeholders(is_occupied, 1)

  # Compute mask based on occupancy.
  mask = is_occupied[Ellipsis, 0]  # Rx(N+M)
  if variables['macroblock_idxs'] is not None:
    # .. and also make sure that we are referring to a non-empty atlas block.
    # (Empty atlas blocks are indicated by the value -1.)
    mask = mask & (variables['macroblock_idxs'][Ellipsis, 0] >= 0)  # Rx(N+M)

  # TODO(duckworthd): When using a coarse-grained occupancy grid, we may use
  # positions that don't have corresponding baked triplane features. Fix this by
  # ignoring positions that have placeholder feature values.

  # Keep only the first M hits. Thanks to our placeholder hits every ray also
  # has at least M hits and therefore it is guranteed that the resulting mask
  # will have exactly R*M nonzero entries.
  mask = mask & (mask.cumsum(axis=-1) <= max_steps)  # Rx(N+M)
  # Each ray now has exactly M hits. This is the linear index correspodning to
  # those hits.
  inds = jnp.nonzero(
      mask.reshape(-1), size=(num_rays * max_steps), fill_value=-1
  )[0]  # R*M

  # Keep track of the placeholder entries to ignore these values later on.
  # is_placeholder is True where a placeholder value is required.
  #
  # Construct the array indices corresponding to the first M hits per ray.
  # Discard the 'rays' index.
  _, is_placeholder = jnp.unravel_index(
      inds, (num_rays, num_steps + max_steps)
  )  # R*(N+M), R*(N+M)
  # Which entries along the 'samples along ray' index are placeholder entries?
  is_placeholder = jnp.reshape(
      is_placeholder >= num_steps,
      (num_rays, max_steps),
  )  # RxM

  # Filter arrays by only keeping the first M hits: Rx(N+M)xC --> RxMxC.
  def first_m_per_ray(x):
    d = x.shape[-1]
    return x.reshape(-1, d)[inds].reshape(num_rays, max_steps, d)

  variables = jax.tree_util.tree_map(first_m_per_ray, variables)

  return variables, is_placeholder


@jax.named_scope('gather_sparse_grid')
def _gather_sparse_grid(
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_positions,
    atlas_grid_idxs,
    macroblock_idxs,
    config,
):
  """Gathers sparse grid features and densities.

  Args:
    sparse_grid_features: ...
    sparse_grid_density: ...
    sparse_grid_positions: f32[..., 3]. Positions in sparse grid voxel
      coordinates.
    atlas_grid_idxs: i32[..., 3]. ijk indices into macroblock_idxs. Determines
      which macroblock to use.
    macroblock_idxs: i32[I, J, K, 3]. ijk indices into sparse_grid_features.
    config: ...

  Returns:
    features: f32[..., C]. Feature vectors preactivation for each
      sparse_grid_position.
    densities: f32[..., 1]. Density preactivation for each sparse_grid_position.
  """
  batch_dims = sparse_grid_positions.shape[:-1]
  data_block_size = config.data_block_size
  atlas_block_size = baking.get_atlas_block_size(data_block_size)
  range_features = config.range_features
  range_density = config.range_density

  # Sparse grid voxel position of the lower corner of the macroblock.
  # Doesn't include half-voxel offset.
  min_aabb_positions = (atlas_grid_idxs * data_block_size).astype(
      jnp.float32
  )  # RxMx3

  # Voxel position within a macroblock. Includes half-voxel offset.
  positions_within_macroblock = (
      sparse_grid_positions - min_aabb_positions
  )  # RxMx3

  # Voxel position in the atlas. Includes half-voxel offset.
  positions_within_atlas = positions_within_macroblock + (
      macroblock_idxs * atlas_block_size
  ).astype(jnp.float32)  # RxMx3

  # Look up quantities in the atlas.
  gathered_features = quantize.dequantize_and_interpolate(
      positions_within_atlas,
      sparse_grid_features,
      range_features[0],
      range_features[1],
  )  # RxMxD
  gathered_density = quantize.dequantize_and_interpolate(
      positions_within_atlas,
      sparse_grid_density,
      range_density[0],
      range_density[1],
  )  # RxMx1

  features_sparse_grid = jnp.reshape(
      gathered_features, (*batch_dims, models.NUM_CHANNELS)
  )
  density_sparse_grid = jnp.reshape(gathered_density, (*batch_dims, 1))

  return features_sparse_grid, density_sparse_grid


@jax.named_scope('gather_triplanes')
def _gather_triplanes(
    planes_features,
    planes_density,
    triplane_positions,
    config,
):
  """Gathers triplane features and densities.

  Args:
    planes_features: ...
    planes_density: ...
    triplane_positions: f32[..., 3]. Positions in triplane voxel coordinates.
    config: ...

  Returns:
    features: f32[..., 3, C]. Feature vectors preactivation for each
      triplane position. One per plane.
    densities: f32[..., 3, 1]. Density preactivation for each triplane
      position. One per plane.
  """
  batch_dims = triplane_positions.shape[:-1]
  range_features = config.range_features
  range_density = config.range_density

  features_triplanes = []
  density_triplanes = []
  for plane_idx in range(3):
    # Compute indices of the two ouf of three dims along which we bilinearly
    # interpolate.
    axis_idxs = [h for h in range(3) if h != plane_idx]
    positions_projected_to_plane = triplane_positions[Ellipsis, axis_idxs]  # RxMx2
    gathered_features = quantize.dequantize_and_interpolate(
        positions_projected_to_plane,
        planes_features[plane_idx],
        range_features[0],
        range_features[1],
    )  # RxMxD
    gathered_density = quantize.dequantize_and_interpolate(
        positions_projected_to_plane,
        planes_density[plane_idx],
        range_density[0],
        range_density[1],
    )  # RxMx1

    features_triplanes.append(
        jnp.reshape(gathered_features, (*batch_dims, models.NUM_CHANNELS))
    )
    density_triplanes.append(jnp.reshape(gathered_density, (*batch_dims, 1)))

  features_triplanes = jnp.stack(features_triplanes, axis=-2)
  density_triplanes = jnp.stack(density_triplanes, axis=-2)

  return features_triplanes, density_triplanes


@jax.named_scope('merge_features_and_density')
def _merge_features_and_density(
    features_sparse_grid,
    density_sparse_grid,
    features_triplanes,
    density_triplanes,
    is_placeholder,
    config,
):
  """Combines features and density contributions.

  Args:
    features_sparse_grid: f32[..., c]. Feature preactivations from sparse grid
      representation.
    density_sparse_grid: f32[..., 1]. Density preactivations from sparse grid
      representation.
    features_triplanes: f32[..., 3, c]. Feature preactions from triplane
      representation.
    density_triplanes: f32[..., 3, 1]. Density preactivations from triplane
      representation.
    is_placeholder: bool[...]. True if an entry should be ignored.
    config: ...

  Returns:
    features: f32[..., c]. Final feature preactivations.
    density: f32[..., 1]. Final density preactivations.
  """
  features, density = grid_utils.sum_with_triplane_weights(
      triplane_features=features_triplanes,
      sparse_grid_features=features_sparse_grid,
      triplane_density=density_triplanes,
      sparse_grid_density=density_sparse_grid,
      config=config,
  )

  # Replace placeholder entries with default values.
  def replace_placeholder(x, v):
    return jnp.where(is_placeholder[Ellipsis, None], jnp.full_like(x, v), x)

  features = replace_placeholder(features, 0)  # RxMxD
  density = replace_placeholder(density, -1 * jnp.inf)  # RxMx1

  # Apply activation functions after interpolation.
  features = math.feature_activation(features)  # RxMxD
  density = math.density_activation(density)  # RxMx1

  return features, density


@functools.partial(
    jax.pmap,
    in_axes=(0,) * 11 + (None,),
    static_broadcasted_argnums=list(range(12, 20)),
)
def _render_rays_pmap(
    # partitioned arguments:
    origins,  # 0
    directions,  # 1
    viewdirs,  # 2
    exposure_values,  # 3
    sparse_grid_features,  # 4
    sparse_grid_density,  # 5
    sparse_grid_block_indices,  # 6
    planes_features,  # 7
    planes_density,  # 8
    deferred_mlp_vars,  # 9
    occupancy_grid,  # 10
    # replicated arguments:
    sm_idx,  # 11
    # static arguments:
    occupancy_voxel_size,  # 12
    num_steps,  # 13
    max_steps,  # 14
    step_size,  # 15
    bg_intensity,  # 16
    config,  # 17
    grid_config,  # 18
    return_ray_results,  # 19
):
  """Renders from the baked representation.

  Args:
    origins: f32[..., 3]. Ray origins in world coordinates.
    directions: f32[..., 3]. Ray directions in world coordinates.
    viewdirs: f32[..., 3]. Unit-norm view directions in world coordinates.
    exposure_values: Optional f32[..., 1]. Camera exposure for each ray.
    sparse_grid_features:
    sparse_grid_density:
    sparse_grid_block_indices:
    planes_features:
    planes_density:
    deferred_mlp_vars:
    occupancy_grid:
    sm_idx: i32[]. Submodel index.
    occupancy_voxel_size: float. Length of a voxel's side in squash coordinates.
    num_steps: int. Number of points to sample per camera ray.
    max_steps: int. Maximum number of points to use for compositing along a
      camera ray.
    step_size: float. Distance to travel per step in squash coordinates.
    bg_intensity: float. How bright the background is.
    config: Config instance.
    grid_config: See grid_utils.initialize_grid_config().
    return_ray_results: bool. If True, return debug outputs.

  Returns:
    ...

  """
  # Render from the baked representation in the same way as during WebGL
  # rendering, i.e. by using a constant step size. Empty space skipping is
  # attributed for by passing in the finest `occupancy_grid` used by the
  # WebGL renderer.

  # We first generate `num_steps` sampling points per ray by densely
  # marching along the ray with a step size equal to the size of a voxel, i.e.
  # each voxel along the ray is sampled once. As a result `num_steps` must be
  # choosen such that in the worst case scenario of traversing the diagonal
  # of the volume one sample per voxel can placed: `num_steps = sqrt(3) *
  # resolution`.

  # Most of the `num_steps` sample points will lie in empty space and can be
  # discarded, i.e. the representation does not need to be queried at those
  # points.
  # We asssume that there only will be maximally `max_steps` hits per ray.
  # In this context we define a hit as a sample that is in non-empty space as
  # indicated by the `occupancy_grid`. By imposing this upper bound we can
  # work with fixed-size arrays as required by JAX's jit. `max_steps` needs
  # to be determined empirically.
  use_triplanes = config.triplane_resolution > 0
  use_sparse_grid = config.sparse_grid_resolution > 0

  if use_sparse_grid:
    assert sparse_grid_features is not None
    assert sparse_grid_density is not None
    assert sparse_grid_block_indices is not None
    assert planes_density is not None

  if use_triplanes:
    assert planes_features is not None
    assert planes_density is not None

  # Generate sampling points along ray in submodel coordinates.
  t_positions, _, s_positions = _generate_sample_positions(
      t_origins=origins,
      t_directions=directions,
      t_viewdirs=viewdirs,
      sm_idx=sm_idx,
      num_samples=num_steps,
      step_size_squash=step_size,
      config=config,
      grid_config=grid_config,
  )  # RxSx3, RxSx3, RxSx3

  # Calculate step_size[t] = dist(position[t+1], position[t]) for all adjacent
  # pairs of points along each camera ray. Distances are in world coordinates.
  t_step_sizes = _generate_step_sizes(t_positions)  # RxSx1
  assert t_step_sizes.shape[:-1] == t_positions.shape[:-1]

  # Constructs positions in triplane voxel coordinates.
  triplane_positions = None
  if use_triplanes:
    triplane_positions = _generate_triplane_positions(
        s_positions=s_positions,
        grid_config=grid_config,
    )  # RxSx3

  # Constructs positions in sparse grid and macroblock voxel coordinates.
  sparse_grid_positions = atlas_grid_idxs = macroblock_idxs = None
  if use_sparse_grid:
    sparse_grid_positions, atlas_grid_idxs, macroblock_idxs = (
        _generate_sparse_grid_positions(
            s_positions=s_positions,
            sparse_grid_block_indices=sparse_grid_block_indices,
            config=config,
            grid_config=grid_config,
        )
    )  # RxSx3, RxSx3, RxSx3

  # Determine which positions lie in an occupied voxel.
  # WARNING: You will get false positives when using downsampled occupancy
  # grids! Renders will look wrong unless the baked representation captures
  # the neighborhood of each alive_voxel!!
  is_occupied = _generate_is_occupied(
      s_positions=s_positions,
      occupancy_grid=occupancy_grid,
      occupancy_voxel_size=occupancy_voxel_size,
  )  # RxSx1

  # Let R = num_rays, N = num_steps, and M = max_steps.  We find the first
  # M hits per ray out of N sampling points.
  variables = dict(
      t_positions=t_positions,
      macroblock_idxs=macroblock_idxs,
      sparse_grid_positions=sparse_grid_positions,
      atlas_grid_idxs=atlas_grid_idxs,
      t_step_sizes=t_step_sizes,
      triplane_positions=triplane_positions,
  )
  variables, is_placeholder = _first_m_hits_per_ray(
      variables, is_occupied, max_steps
  )
  t_positions = variables['t_positions']
  macroblock_idxs = variables['macroblock_idxs']
  sparse_grid_positions = variables['sparse_grid_positions']
  atlas_grid_idxs = variables['atlas_grid_idxs']
  t_step_sizes = variables['t_step_sizes']
  triplane_positions = variables['triplane_positions']

  # Fetch from sparse grid.
  features_sparse_grid = density_sparse_grid = None
  if use_sparse_grid:
    features_sparse_grid, density_sparse_grid = _gather_sparse_grid(
        sparse_grid_features=sparse_grid_features,
        sparse_grid_density=sparse_grid_density,
        sparse_grid_positions=sparse_grid_positions,
        atlas_grid_idxs=atlas_grid_idxs,
        macroblock_idxs=macroblock_idxs,
        config=config,
    )

  # From from triplanes.
  features_triplanes = density_triplanes = []
  if use_triplanes:
    features_triplanes, density_triplanes = _gather_triplanes(
        planes_features=planes_features,
        planes_density=planes_density,
        triplane_positions=triplane_positions,
        config=config,
    )

  # Merge features and density
  features, density = _merge_features_and_density(
      features_sparse_grid=features_sparse_grid,
      density_sparse_grid=density_sparse_grid,
      features_triplanes=features_triplanes,
      density_triplanes=density_triplanes,
      is_placeholder=is_placeholder,
      config=config,
  )  # RxMxC, RxMx1

  # Alpha-blending.
  #
  # Compute distance from camera origin to each point along the camera ray.
  # Distances in world coordinates. This is the distance to the beginning of
  # each ray interval.
  t_dist = jnp.linalg.norm(
      t_positions - origins[Ellipsis, jnp.newaxis, :], axis=-1, keepdims=True
  )  # RxMx1
  t_intervals = jnp.concatenate([t_dist, t_dist + t_step_sizes], axis=-1)
  weights = render.compute_volume_rendering_weights_from_intervals(
      density[Ellipsis, 0],  # RxM
      intervals=t_intervals,  # RxMx2
  )[Ellipsis, jnp.newaxis]  # RxMx1
  features_blended = jnp.sum(weights * features, axis=-2)  # RxC

  # Compute accumulated opacity.
  acc = jnp.sum(weights, axis=-2)  # Rx1

  # Compute depth = average distance along camera ray.
  depth = render.compute_depth_from_intervals(t_intervals, weights[Ellipsis, 0])
  depth = depth[Ellipsis, jnp.newaxis]  # Rx1

  # Evaluate view-dependency MLP.
  render_fn = models.DEFERRED_RENDER_FNS[config.deferred_rendering_mode]

  def apply_deferred_mlp(x):
    deferred_mlp = models.DeferredMLP(
        use_exposure=config.use_exposure_in_deferred_mlp,
        num_kernels=grid_config['num_submodel_hash_encoding_kernels'],
    )
    sm_idxs = jnp.array([sm_idx], dtype=jnp.int32)
    param_idxs = coord.sm_idxs_to_params_idxs(sm_idxs, config, grid_config)
    param_idxs = jnp.broadcast_to(param_idxs, x[Ellipsis, :1].shape)
    deferred_mlp_origins = coord.world_to_deferred_mlp(
        sm_idxs, origins, config, grid_config
    )
    deferred_mlp_viewdirs = viewdirs
    return deferred_mlp.apply(
        {'params': deferred_mlp_vars},
        features=x,
        origins=deferred_mlp_origins,
        viewdirs=deferred_mlp_viewdirs,
        exposures=exposure_values,
        param_idxs=param_idxs,
        # TODO(duckworthd): Add these feature vectors.
        viewdir_features=None,
        origin_features=None,
    )

  _, rendering, rendering_extras = render_fn(
      rng=None,
      features=features_blended,
      acc=acc[Ellipsis, 0],
      model=apply_deferred_mlp,
      bg_rgbs=bg_intensity,
  )
  chex.assert_equal_shape_prefix(
      (rendering['rgb'], features_blended, acc, depth), prefix_len=1
  )

  result = dict(
      depth=depth,
      acc=acc,
      **rendering,
  )
  if return_ray_results:
    extras = dict(
        density=density,
        s_positions=s_positions,
        weights=weights,
        **rendering_extras,
    )
    result.update(extras)
  return result


def _render_rays_chunked(
    chunk_size,
    rays,
    psparse_grid_features,
    psparse_grid_density,
    psparse_grid_block_indices,
    pplanes_features,
    pplanes_density,
    pdeferred_mlp_vars,
    poccupancy_grid,
    sm_idx,
    config,
    max_steps,
    bg_intensity,
    occupancy_grid_factor,
    return_ray_results,
):
  """Chunking wrapper around _render_rays."""
  grid_config = config.grid_config

  # Side-length of the smallest voxel quantity worth considering in the scene.
  # Measured in squash coordinates.
  voxel_size_to_use = grid_config['voxel_size_to_use']

  # Calculates the side-length of an occupancy grid voxel in squash coordinates.
  occupancy_voxel_size = voxel_size_to_use * occupancy_grid_factor

  # Step size is expressed in squash coordinates. Take equal-sized steps in
  # squash coordinates based on smallest voxel length.
  step_size = voxel_size_to_use

  # Adjust step size based on the submodel resolution to ensure that the squash
  # zone is sufficiently well-sampled. With smaller submodels, more
  # visually-salient content is pushed outside of the Euclidean zone. This
  # increases the number of samples per ray.
  step_size = step_size * (
      grid_config['submodel_voxel_size']
      / grid_utils.calculate_submodel_voxel_size(1)
  )

  # If desired, take more than one sample per voxel. This decreases the step
  # size again.
  step_size = step_size / config.num_samples_per_voxel

  # Calculate the number of steps based on step size. Account for the diagonal
  # length of a voxel by increasing the number of steps by sqrt(3).
  scene_size = jnp.sqrt(3) * (grid_utils.WORLD_MAX - grid_utils.WORLD_MIN)
  num_steps = int(scene_size / step_size)

  logging.info(
      f'Querying up to {num_steps=} points along each camera ray. The first'
      f' {max_steps=} hits will be used in alpha-compositing.'
  )

  batch_shape = rays.origins.shape[:-1]
  origins = rays.origins.reshape(-1, 3)
  directions = rays.directions.reshape(-1, 3)
  viewdirs = rays.viewdirs.reshape(-1, 3)
  exposure_values = None
  if rays.exposure_values is not None:
    exposure_values = rays.exposure_values.reshape(-1, 1)

  rendering = None  # Will be overwritten in the loop below.
  num_rays = origins.shape[0]
  for chunk_start in list(range(0, num_rays, chunk_size)):
    chunk_end = min(chunk_start + chunk_size, num_rays)
    # Set up ray origins, directions for pmap.
    origins_chunk, chunk_batch_info = utils.pre_pmap(
        origins[chunk_start:chunk_end], 1
    )
    directions_chunk, _ = utils.pre_pmap(directions[chunk_start:chunk_end], 1)
    viewdirs_chunk, _ = utils.pre_pmap(viewdirs[chunk_start:chunk_end], 1)
    exposure_values_chunk = None
    if exposure_values is not None:
      exposure_values_chunk, _ = utils.pre_pmap(
          exposure_values[chunk_start:chunk_end], 1
      )
    rendering_chunk = _render_rays_pmap(
        origins_chunk,  # 0
        directions_chunk,  # 1
        viewdirs_chunk,  # 2
        exposure_values_chunk,  # 3
        psparse_grid_features,  # 4
        psparse_grid_density,  # 6
        psparse_grid_block_indices,  # 6
        pplanes_features,  # 7
        pplanes_density,  # 8
        pdeferred_mlp_vars,  # 9
        poccupancy_grid,  # 10
        sm_idx,  # 11
        float(occupancy_voxel_size),  # 12
        int(num_steps),  # 13
        int(max_steps),  # 14
        float(step_size),  # 15
        bg_intensity,  # 16
        config,  # 17
        grid_config,  # 18
        bool(return_ray_results),  # 19
    )
    unbatch_chunk = lambda x: utils.post_pmap(x, chunk_batch_info)  # pylint: disable=cell-var-from-loop
    rendering_chunk = jax.tree_util.tree_map(unbatch_chunk, rendering_chunk)

    # Initialize buffers for outputs.
    if chunk_start == 0:
      # Use regular numpy if returning ray results. The results are too large
      # to hold in device memory.
      rendering = {
          key: np.zeros((num_rays, *x.shape[1:]), dtype=x.dtype)
          for key, x in rendering_chunk.items()
      }

    # Update buffers with subbatch results.
    def update_array_with_chunk(x, x_chunk):
      x[chunk_start:chunk_end] = x_chunk  # pylint: disable=cell-var-from-loop
      return x

    rendering = jax.tree_util.tree_map(
        update_array_with_chunk,
        rendering,
        rendering_chunk,
    )

  # Reintroduce batch dimension.
  assert rendering is not None
  rendering = jax.tree_util.tree_map(
      lambda x: x.reshape(*batch_shape, *x.shape[1:]), rendering
  )
  return rendering


def render_dataset(
    config,
    planes_features,
    planes_density,
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_block_indices,
    deferred_mlp_vars,
    occupancy_grid_factor,
    occupancy_grid,
    sm_idx,
    dataset,
    *,
    baked_render_dir=None,
    max_num_images=None,
    max_steps=2048,
    batch_size=8192,
    bg_intensity=0.5,
    return_ray_results=False,
):
  """Renders a baked model.

  Note: only cameras 'owned' by this submodel will be rendered. Others will be
  skipped.

  Args:
    config:
    planes_features:
    planes_density:
    sparse_grid_features:
    sparse_grid_density:
    sparse_grid_block_indices:
    deferred_mlp_vars:
    occupancy_grid_factor:
    occupancy_grid:
    sm_idx: Which submodel is being used.
    dataset: Which dataset ot render.
    baked_render_dir:  Where to save results.
    max_num_images: Maximum number of frames to render.
    max_steps: Maximum number of points to use in alpha compositing per ray.
    batch_size: Number of rays to evaluate at once.
    bg_intensity: How strong the background color is.
    return_ray_results: bool. If True, return debug outputs.

  Returns:
    results: [{str: Any}]. Each dict represents the results from rendering a
      single camera.
    metrics: {str: float | list[float]}. Contains PSNR and SSIM numbers
      per-frame and averaged over the entire dataset.
  """
  grid_config = config.grid_config

  if baked_render_dir is not None:
    baked_render_dir.mkdir(parents=True, exist_ok=True)

  # Replicate representation to each device.
  gc.collect()
  (
      pplanes_features,
      pplanes_density,
      psparse_grid_features,
      psparse_grid_density,
      psparse_grid_block_indices,
      pdeferred_mlp_vars,
      poccupancy_grid,
  ) = flax.jax_utils.replicate((
      planes_features,
      planes_density,
      sparse_grid_features,
      sparse_grid_density,
      sparse_grid_block_indices,
      deferred_mlp_vars,
      occupancy_grid,
  ))
  gc.collect()
  threadpool = utils.AsyncThreadPool()

  # Compute metrics on baked model.
  metric_harness = image.MetricHarness()

  # Decide which cameras to render.
  render_idxs = range(len(dataset.camtoworlds))
  if max_num_images is None:
    max_num_images = len(dataset.camtoworlds)
  elif isinstance(max_num_images, (tuple, list)):
    render_idxs = max_num_images
    max_num_images = len(render_idxs)
  else:
    max_num_images = int(max_num_images)

  # Render frames assigned to this submodel.
  results = []
  num_rendered_frames = 0
  for render_idx in render_idxs:
    # Exit if a sufficient number of frames have been rendered.
    if num_rendered_frames >= max_num_images:
      break

    # Determine if this image belongs to this submodel. If not, skip it.
    cam_sm_idx = baking.sm_idx_for_camera(
        dataset, render_idx, config, grid_config
    )
    if cam_sm_idx != sm_idx:
      logging.debug(
          f'Skipping training view {render_idx}. It belongs to'
          f' submodel={cam_sm_idx}.'
      )
      continue

    # Increment number of rendered frames.
    logging.info(
        f'Rendering frame #{render_idx} ({num_rendered_frames} of'
        f' {max_num_images})'
    )
    num_rendered_frames += 1

    # Construct paths for images.
    rgb_test_path = depth_test_path = gt_test_path = None
    if baked_render_dir is not None:
      rgb_test_path = baked_render_dir / f'rgb.test.{render_idx:03d}.png'
      depth_test_path = baked_render_dir / f'depth.test.{render_idx:03d}.png'
      gt_test_path = baked_render_dir / f'gt.test.{render_idx:03d}.png'

    try:
      # Load images from disk. If this succeeds, continue the computation
      rgb, depth, gt_rgb = parallel.ParallelMap(
          lambda p: utils.load_img(os.fspath(p)) / 255.0,
          [rgb_test_path, depth_test_path, gt_test_path],
      )
      rendering = {'rgb': rgb, 'depth': depth}

    except Exception:  # pylint: disable=broad-exception-caught
      # Images couldn't be loaded for some reason. This is the typical case.
      # Regardless of the reason, the logic below regenerates them.

      # Generate rays.
      rays = datasets.cam_to_rays(dataset, render_idx)
      rays = datasets.preprocess_rays(
          rays=rays, mode='test', merf_config=config, dataset=dataset
      )

      # Render rays.
      gc.collect()
      rendering = _render_rays_chunked(
          chunk_size=batch_size,
          rays=rays,
          psparse_grid_features=psparse_grid_features,
          psparse_grid_density=psparse_grid_density,
          psparse_grid_block_indices=psparse_grid_block_indices,
          pplanes_features=pplanes_features,
          pplanes_density=pplanes_density,
          pdeferred_mlp_vars=pdeferred_mlp_vars,
          poccupancy_grid=poccupancy_grid,
          sm_idx=sm_idx,
          config=config,
          max_steps=max_steps,
          bg_intensity=bg_intensity,
          occupancy_grid_factor=occupancy_grid_factor,
          return_ray_results=return_ray_results,
      )
      rendering = jax.device_get(rendering)
      rgb = rendering['rgb']
      depth = image.colorize_depth(rendering['depth'][Ellipsis, 0])
      gt_rgb = jax.device_get(dataset.images[render_idx])

    # Save render to disk.
    if baked_render_dir is not None:
      for img, path in [
          (rgb, rgb_test_path),
          (depth, depth_test_path),
          (gt_rgb, gt_test_path),
      ]:
        threadpool.submit(utils.save_img_u8, img, os.fspath(path))

    # Save outputs for later.
    result = {
        'cam_idx': render_idx,
        'sm_idx': sm_idx,
        'gt': gt_rgb,
        **rendering,
    }

    # Compute metrics.
    # TODO(duckworthd): Compare against teacher.
    metric = metric_harness(rendering['rgb'], gt_rgb)
    for metric_name, baked_metric_name in zip(METRIC_NAMES, BAKED_METRIC_NAMES):
      result[baked_metric_name] = metric.get(metric_name, np.nan)

    # Log metrics to STDERR.
    logging.info(f'{render_idx=} {metric=}')

    # Save results for later.
    results.append(jax.device_get(result))

  # Compute average metrics across entire dataset.
  metrics = defaultdict(list)
  for key in BAKED_METADATA_NAMES + BAKED_METRIC_NAMES:
    for result in results:
      metrics[key].append(result[key])
    if key in BAKED_METRIC_NAMES:
      key_avg = f'{key}.avg'
      metrics[key_avg] = np.mean(metrics[key])

  if baked_render_dir is not None:
    utils.save_json(
        metrics, os.fspath(baked_render_dir / METRICS_JSON_FILE_NAME)
    )

  avg_metrics = {k: v for k, v in metrics.items() if k.endswith('.avg')}
  logging.info(f'Average over all test images: {avg_metrics}')

  # Finish async operations
  threadpool.flush()

  return results, metrics


def render_path(
    config,
    planes_features,
    planes_density,
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_block_indices,
    deferred_mlp_vars,
    occupancy_grid_factor,
    occupancy_grid,
    sm_idx,
    dataset,
    *,
    baked_render_dir=None,
    max_steps=2048,
    batch_size=8192,
    bg_intensity=0.5,
):
  """Renders dataset's render_path for a baked model.

  Note: Renders *all* frames, even if the camera does not belong to this
  submodel.

  Args:
    config:
    planes_features:
    planes_density:
    sparse_grid_features:
    sparse_grid_density:
    sparse_grid_block_indices:
    deferred_mlp_vars:
    occupancy_grid_factor:
    occupancy_grid:
    sm_idx: Which submodel is being used.
    dataset: Which dataset ot render.
    baked_render_dir:  Where to save results.
    max_steps: Maximum number of points to use in alpha compositing per ray.
    batch_size: Number of rays to evaluate at once.
    bg_intensity: How strong the background color is.

  Returns:
    results: [{str: Any}]. Each dict represents the results from rendering a
      single camera.
  """
  if baked_render_dir is not None:
    baked_render_dir.mkdir(parents=True, exist_ok=True)

  # Replicate representation to each device.
  gc.collect()
  (
      pplanes_features,
      pplanes_density,
      psparse_grid_features,
      psparse_grid_density,
      psparse_grid_block_indices,
      pdeferred_mlp_vars,
      poccupancy_grid,
  ) = flax.jax_utils.replicate((
      planes_features,
      planes_density,
      sparse_grid_features,
      sparse_grid_density,
      sparse_grid_block_indices,
      deferred_mlp_vars,
      occupancy_grid,
  ))
  # Threadpool for async operations.
  threadpool = utils.AsyncThreadPool()

  results = []
  num_frames = len(dataset.camtoworlds)
  render_idxs = list(range(0, num_frames, config.baked_render_path_video_every))
  skip_foreign_cameras = not config.baked_render_path_all_cameras
  for i, render_idx in enumerate(render_idxs):
    # Determine if this image belongs to this submodel. If not, skip it.
    cam_sm_idx = baking.sm_idx_for_camera(
        dataset, render_idx, config, config.grid_config
    )
    if skip_foreign_cameras and cam_sm_idx != sm_idx:
      logging.debug(
          f'Skipping training view {render_idx}. It belongs to'
          f' submodel={cam_sm_idx}.'
      )
      continue
    logging.info(f'Rendering frame #{render_idx} ({i} of {len(render_idxs)})')

    # Construct paths for images.
    rgb_test_path = depth_test_path = None
    if baked_render_dir is not None:
      rgb_test_path = baked_render_dir / f'rgb.render_path.{render_idx:05d}.png'
      depth_test_path = (
          baked_render_dir / f'depth.render_path.{render_idx:05d}.png'
      )

    try:
      # Load images from disk. If this succeeds, continue the computation
      rgb, depth = parallel.ParallelMap(
          lambda p: utils.load_img(os.fspath(p)) / 255.0,
          [rgb_test_path, depth_test_path],
      )

    except Exception:  # pylint: disable=broad-exception-caught
      # Images couldn't be loaded for some reason. This is the typical case.
      # Regardless of the reason, the logic below regenerates them.
      rays = datasets.cam_to_rays(dataset, render_idx)
      rays = datasets.preprocess_rays(
          rays=rays, mode='test', merf_config=config, dataset=dataset
      )

      # Render rays.
      gc.collect()
      rendering = _render_rays_chunked(
          chunk_size=batch_size,
          rays=rays,
          psparse_grid_features=psparse_grid_features,
          psparse_grid_density=psparse_grid_density,
          psparse_grid_block_indices=psparse_grid_block_indices,
          pplanes_features=pplanes_features,
          pplanes_density=pplanes_density,
          pdeferred_mlp_vars=pdeferred_mlp_vars,
          poccupancy_grid=poccupancy_grid,
          sm_idx=sm_idx,
          config=config,
          max_steps=max_steps,
          bg_intensity=bg_intensity,
          occupancy_grid_factor=occupancy_grid_factor,
          return_ray_results=False,
      )
      rendering = jax.device_get(rendering)
      rgb = rendering['rgb']
      depth = image.colorize_depth(rendering['depth'][Ellipsis, 0])

    # Write images to disk.
    for img, path in [(rgb, rgb_test_path), (depth, depth_test_path)]:
      threadpool.submit(utils.save_img_u8, img, os.fspath(path))

    # Save outputs for later.
    result = {
        'cam_idx': render_idx,
        'cam_sm_idx': cam_sm_idx,
        'sm_idx': sm_idx,
        'rgb': rgb,
        'depth': depth,
    }
    results.append(jax.device_get(result))

  # Finish all asynchronous operations.
  threadpool.flush()

  # Convert frames into videos. These videos may contain missing frames in the
  # multi-submodel setting.
  create_render_path_mp4_files(baked_render_dir, config)

  return results


def merge_all_baked_renders(sm_idxs, log_dir, baked_log_dir):
  """Copy baked renders from all submodels into a single folder.

  Copies all rendered images to "${BAKED_DIR}/baked_render/all"

  Args:
    sm_idxs: list[int]. Which submodels to merge.
    log_dir: epath.Path. Checkpoint directory for MERF model.
    baked_log_dir: epath.Path. Checkpoint directory for baked assets.

  Returns:
    {str: float or list[float]}. Baked metrics accumulated from all models.
      Includes per-image information such as sm_idx, cam_idx, and
      gt.baked.<metric>. Also contains average information such as
      gt.baked.<metric>.avg.
  """
  # Determine where prebaked assets are stored.
  orig_render_dir = log_dir / 'orig_render'
  output_dir = construct_sm_render_dir(baked_log_dir, 'all')
  kwargs = []

  # Gather all baked renders
  for sm_idx in sm_idxs:
    sm_render_dir = construct_sm_render_dir(baked_log_dir, sm_idx)
    for kind in ['rgb', 'depth']:
      for input_path in sm_render_dir.glob(f'{kind}.test.*.png'):
        output_path = output_dir / input_path.name
        kwargs.append({'src': input_path, 'dst': output_path})

  # Gather all ground truth images and teacher renders
  for kind in ['teacher', 'gt']:
    for input_path in orig_render_dir.glob(f'{kind}.test.*.png'):
      output_path = output_dir / input_path.name
      kwargs.append({'src': input_path, 'dst': output_path})

  def copy_file(src, dst):
    src.copy(dst, overwrite=True)

  # Copy all files
  output_dir.mkdir(parents=True, exist_ok=True)
  parallel.RunInParallel(copy_file, kwargs, num_workers=20)

  # Load all baked metrics files.
  metrics = []
  for sm_idx in sm_idxs:
    sm_render_dir = construct_sm_render_dir(baked_log_dir, sm_idx)
    sm_metrics_path = sm_render_dir / METRICS_JSON_FILE_NAME
    if not sm_metrics_path.exists():
      raise ValueError(
          f'No metrics json file found for {sm_idx=}. It should be here:'
          f' {sm_metrics_path}'
      )
    sm_metrics = utils.load_json(sm_metrics_path)
    metrics.append(sm_metrics)

  # Combine them together into one super file.
  result = defaultdict(list)
  for key in BAKED_METADATA_NAMES + BAKED_METRIC_NAMES:
    # Accumulate results from all submodels.
    for sm_metrics in metrics:
      # Some keys are missing when a submodel has no test renders.
      result[key] += sm_metrics.get(key, [])
    # Final post-processing for metrics
    if key in BAKED_METRIC_NAMES:
      key_avg = f'{key}.avg'
      result[key_avg] = np.mean(result[key])

  # Save result
  utils.save_json(result, os.fspath(output_dir / METRICS_JSON_FILE_NAME))

  return result


def construct_sm_render_dir(baked_log_dir, sm_idx):
  """Constructs output directory for renders of a particular submodel."""
  if isinstance(sm_idx, str):
    name = sm_idx
  else:  # sm_idx is an int.
    name = f'sm_{sm_idx:03d}'
  return baked_log_dir / 'baked_render' / name


def create_render_path_mp4_files(output_dir, config):
  """Creates mp4 files from PNG frames."""
  for key in ['rgb', 'depth']:
    video_path = output_dir / f'{key}.render_path.mp4'
    fps = 60. / config.baked_render_path_video_every
    frame_paths = list(output_dir.glob(f'{key}.render_path.*.png'))
    frame_paths = list(sorted(frame_paths))
    if not frame_paths:
      logging.info(f'No frames found for {key=}. Skipping mp4 file creation.')
      continue
    frames = parallel.ParallelMap(media.read_image, frame_paths)
    media.write_video(video_path, frames, fps=fps)


def merge_all_baked_path_renders(sm_idxs, baked_log_dir, config):
  """Merge all render path frames into videos.

  Copies all rendered images to "${BAKED_DIR}/baked_render/all" and creates mp4
  videos.

  Args:
    sm_idxs: list[int]. Which submodels to merge.
    baked_log_dir: epath.Path. Checkpoint directory for baked assets.
    config: configs.Config instance
  """
  # Determine where prebaked assets are stored.
  output_dir = construct_sm_render_dir(baked_log_dir, 'all')
  kwargs = []

  # Gather all baked renders
  for sm_idx in sm_idxs:
    sm_render_dir = construct_sm_render_dir(baked_log_dir, sm_idx)
    for kind in ['rgb', 'depth']:
      for input_path in sm_render_dir.glob(f'{kind}.render_path.*.png'):
        output_path = output_dir / input_path.name
        kwargs.append({'src': input_path, 'dst': output_path})

  def copy_file(src, dst):
    src.copy(dst, overwrite=True)

  # Copy all files
  output_dir.mkdir(parents=True, exist_ok=True)
  parallel.RunInParallel(copy_file, kwargs, num_workers=20)

  # Create render path mp4 videos.
  create_render_path_mp4_files(output_dir, config)

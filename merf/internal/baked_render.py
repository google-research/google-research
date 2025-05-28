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

import functools

from internal import baking
from internal import coord
from internal import grid_utils
from internal import math
from internal import models
from internal import quantize
import jax
import jax.numpy as jnp
import tqdm


def gen_sample_points(origins, directions, num_samples, step_size_contracted):
  """Generate sample points along the ray in world space."""

  # Make sure that sample points are `step_size_contracted` apart from each
  # other in contracted space. When setting `step_size_contracted` to the size
  # of a voxel every voxel is sampled once. This mimics the behaviour of the
  # real-time renderer.
  #
  # The actual step size (the distance between sampling points in world space)
  # depends on how deep we are in the contracted space. The higher the magnitude
  # of an input point the larger the steps are becoming, i.e. we sample more
  # sparsely the further away we get from the scene's center.

  def f(x, _):
    x_t = x
    not_too_big = jnp.linalg.norm(x, axis=-1, keepdims=True) < 10e5
    t_next = coord.stepsize_in_squash(x, directions, step_size_contracted)
    x = jnp.where(not_too_big, x + directions * t_next[:, None], x)
    return (x, x_t)

  return jax.lax.scan(
      f, origins, xs=None, length=num_samples, unroll=1)[1].transpose(1, 0, 2)


@functools.partial(jax.jit, static_argnums=list(range(9, 21)))
def _render_rays(
    origins,  # 0
    directions,  # 1
    sparse_grid_features,  # 2
    sparse_grid_density,  # 3
    sparse_grid_block_indices,  # 4
    planes_features,  # 5
    planes_density,  # 6
    deferred_mlp_vars,  # 7
    occupancy_grid,  # 8
    # static arguments:
    sparse_grid_resolution,  # 9
    sparse_grid_voxel_size,  # 10
    triplane_voxel_size,  # 11
    occupancy_voxel_size,  # 12
    data_block_size,  # 13
    num_steps,  # 14
    max_steps,  # 15
    step_size,  # 16
    bg_intensity,  # 17
    range_features,  # 18
    range_density,  # 19
    near,  # 20
):
  """Renders from the baked representation."""
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
  atlas_block_size = baking.get_atlas_block_size(data_block_size)
  use_triplanes = planes_features is not None and planes_density is not None
  use_sparse_grid = sparse_grid_block_indices is not None

  # Generate sampling points along ray.
  origins = origins + near * directions
  positions_world = gen_sample_points(origins, directions, num_steps, step_size)
  distances = jnp.linalg.norm(
      positions_world[:, 1:] - positions_world[:, :-1], axis=-1)[:, :, None]
  positions_world = coord.contract(positions_world)
  distances = jnp.concatenate(
      [distances, 1e10 * jnp.ones_like(distances[Ellipsis, :1, :])], 1)

  if use_triplanes:
    positions_grid_triplane = grid_utils.world_to_grid(
        positions_world, triplane_voxel_size, jnp)

  if use_sparse_grid:
    positions_grid = grid_utils.world_to_grid(
        positions_world, sparse_grid_voxel_size, jnp)
    positions_atlas_grid = (positions_grid / data_block_size).astype(jnp.uint32)

    # Fetch the atlas indices from the indirection grid.
    block_indices = sparse_grid_block_indices[
        positions_atlas_grid[Ellipsis, 0],
        positions_atlas_grid[Ellipsis, 1],
        positions_atlas_grid[Ellipsis, 2],
    ]

    # Checks that samples are inside the scene's bounding box.
    epsilon = 0.1
    valid_mask = positions_grid[Ellipsis, 0] < sparse_grid_resolution - 0.5 - epsilon
    valid_mask = valid_mask & (
        positions_grid[Ellipsis, 1] < sparse_grid_resolution - 0.5 - epsilon)
    valid_mask = valid_mask & (
        positions_grid[Ellipsis, 2] < sparse_grid_resolution - 0.5 - epsilon)
    valid_mask = valid_mask & (positions_grid[Ellipsis, 0] > 0.5 + epsilon)
    valid_mask = valid_mask & (positions_grid[Ellipsis, 1] > 0.5 + epsilon)
    valid_mask = valid_mask & (positions_grid[Ellipsis, 2] > 0.5 + epsilon)
    invalid_mask = ~valid_mask
    block_indices = jnp.where(
        jnp.expand_dims(invalid_mask, -1),
        -1 * jnp.ones_like(block_indices),
        block_indices,
    )

  # Fetch occupancy.
  positions_occ = grid_utils.world_to_grid(
      positions_world, occupancy_voxel_size, jnp
  )
  occupancy_indices = jnp.round(positions_occ).astype(jnp.uint32)
  occupancy = occupancy_grid[
      occupancy_indices[Ellipsis, 0],
      occupancy_indices[Ellipsis, 1],
      occupancy_indices[Ellipsis, 2],
  ][Ellipsis, None]

  num_rays = positions_world.shape[0]

  # Let R = num_rays, N = num_steps, and M = max_steps
  # We want to find the first M hits per ray out of N sampling points.
  # Let's say our ray has h hits. There are two cases to consider:
  #   case A: h >= M
  #   case B: h < M
  # case A is easy to deal with since keeping the first M hits per ray will
  # result in a subarray of exactly length M (which is not true for case B)
  # Therefore we want to eliminate case B and append M "placeholder" hits to
  # each ray. As a result there will be h+M hits per ray and h+M >= M (case A).

  # We append M placeholder entries to the end of each array: RxNxC) --> RxN+MxC
  # We append M ones to occupancy_grid, and the value 1 indicates "occupied"
  # to create the placeholder hits. Likewise we append non-negative
  # block_indices which encodes non-empty space.
  if use_sparse_grid:
    p = jnp.zeros((num_rays, max_steps, 3), dtype=block_indices.dtype)
    block_indices = jnp.concatenate([block_indices, p], axis=1)
    p = jnp.zeros((num_rays, max_steps, 3), dtype=positions_grid.dtype)
    positions_grid = jnp.concatenate([positions_grid, p], axis=1)
    p = jnp.zeros((num_rays, max_steps, 3), dtype=positions_atlas_grid.dtype)
    positions_atlas_grid = jnp.concatenate([positions_atlas_grid, p], axis=1)

  p = jnp.zeros((num_rays, max_steps, 1), dtype=distances.dtype)
  distances = jnp.concatenate([distances, p], axis=1)

  if use_triplanes:
    p = jnp.zeros((num_rays, max_steps, 3), dtype=positions_grid_triplane.dtype)
    positions_grid_triplane = jnp.concatenate(
        [positions_grid_triplane, p], axis=1)
  p = jnp.ones((num_rays, max_steps, 1), dtype=occupancy.dtype)
  occupancy = jnp.concatenate([occupancy, p], axis=1)

  # Compute mask based on occupancy..
  mask = occupancy[Ellipsis, 0]
  if use_sparse_grid:
    # .. and also make sure that we are referring to a non-empty atlas block.
    # (Empty atlas blocks are indicated by the value -1.)
    mask = mask & (block_indices[Ellipsis, 0] >= 0)

  # Keep only the first M hits. Thanks to our placeholder hits every ray also
  # has at least M hits and therefore it is guranteed that the resulting mask
  # will have exactly R*M nonzero entries.
  mask = mask & (mask.cumsum(axis=1) <= max_steps)
  inds = jnp.nonzero(
      mask.reshape(-1), size=num_rays * max_steps, fill_value=-1)[0]

  # Keep track of the placeholder entries to ignore these values later on.
  placeholder_mask = jnp.unravel_index(inds, (num_rays, num_steps + max_steps))
  placeholder_mask = (placeholder_mask[1] >= num_steps).reshape(
      num_rays, max_steps)

  # Filter arrays by only keeping the first M hits: RxN+MxC --> RxMxC.
  if use_sparse_grid:
    block_indices = block_indices.reshape(-1, 3)[inds].reshape(
        num_rays, max_steps, 3)
    positions_grid = positions_grid.reshape(-1, 3)[inds].reshape(
        num_rays, max_steps, 3)
    positions_atlas_grid = positions_atlas_grid.reshape(-1, 3)[inds].reshape(
        num_rays, max_steps, 3)
  distances = distances.reshape(-1, 1)[inds].reshape(num_rays, max_steps, 1)
  if use_triplanes:
    positions_grid_triplane = positions_grid_triplane.reshape(-1, 3)[
        inds].reshape(num_rays, max_steps, 3)

  features = jnp.zeros(
      (num_rays, max_steps, models.NUM_CHANNELS), dtype=jnp.float32)  # RxMx7.
  density = jnp.zeros((num_rays, max_steps, 1), dtype=jnp.float32)  # RxMx1.

  # Fetch from sparse grid.
  if use_sparse_grid:
    min_aabb_positions = (positions_atlas_grid * data_block_size).astype(
        jnp.float32)
    positions_within_block = positions_grid - min_aabb_positions
    positions_atlas = positions_within_block + (
        block_indices * atlas_block_size
    ).astype(jnp.float32)
    gathered_features = quantize.dequantize_and_interpolate(
        positions_atlas,
        sparse_grid_features,
        range_features[0],
        range_features[1],
    )
    gathered_density = quantize.dequantize_and_interpolate(
        positions_atlas, sparse_grid_density, range_density[0], range_density[1]
    )
    features += gathered_features.reshape(features.shape)
    density += gathered_density.reshape(density.shape)

  # From from the three planes.
  if use_triplanes:
    for plane_idx in range(3):
      # Compute indices of the two ouf of three dims along which we bilineary
      # interpolate.
      axis_inds = [h for h in range(3) if h != plane_idx]
      positions_projected_to_plane = positions_grid_triplane[
          Ellipsis, [axis_inds[0], axis_inds[1]]
      ]
      gathered_features = quantize.dequantize_and_interpolate(
          positions_projected_to_plane,
          planes_features[plane_idx],
          range_features[0],
          range_features[1],
      )
      gathered_density = quantize.dequantize_and_interpolate(
          positions_projected_to_plane,
          planes_density[plane_idx],
          range_density[0],
          range_density[1],
      )
      features += gathered_features.reshape(features.shape)
      density += gathered_density.reshape(density.shape)

  # Apply activation functions after interpolation.
  features = jax.nn.sigmoid(features)
  density = math.density_activation(density)
  alpha = math.density_to_alpha(density, distances)

  # Ignore values corresponding to placeholder hits by overwriting them with
  # zeros.
  alpha = jnp.where(placeholder_mask[Ellipsis, None], jnp.zeros_like(alpha), alpha)
  features = jnp.where(
      placeholder_mask[Ellipsis, None], jnp.zeros_like(features), features
  )

  # Alpha-blending.
  alpha_padded = jnp.concatenate([jnp.zeros((num_rays, 1, 1)), alpha], axis=1)
  transmittance = (1.0 - alpha_padded).cumprod(axis=1)[:, :-1]
  weights = alpha * transmittance
  features_blended = (weights * features).sum(axis=1)

  # Blend diffuse RGB color with solid background color.
  acc = weights.sum(axis=-2)
  bg_weight = jnp.maximum(0, 1.0 - acc)
  rgb_diffuse = features_blended[Ellipsis, :3] + bg_weight * bg_intensity
  features_blended = features_blended.at[Ellipsis, :3].set(rgb_diffuse)

  # Evaluate view-dependency MLP.
  deferred_mlp = models.DeferredMLP()
  directions /= jnp.linalg.norm(directions, axis=-1, keepdims=True)
  rgb_specular = deferred_mlp.apply(
      {'params': deferred_mlp_vars}, features_blended, directions
  )
  rgb = rgb_diffuse + rgb_specular
  return dict(rgb=rgb)


def render_rays(
    chunk_size,
    origins,
    directions,
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_block_indices,
    planes_features,
    planes_density,
    deferred_mlp_vars,
    occupancy_grid,
    config,
    grid_config,
    max_steps,
    bg_intensity,
    occupancy_grid_factor,
):
  """Chunking wrapper around _render_rays."""
  voxel_size_to_use = grid_config['voxel_size_to_use']
  occupancy_voxel_size = voxel_size_to_use * occupancy_grid_factor
  step_size = voxel_size_to_use / config.num_samples_per_voxel
  num_steps = int(
      jnp.sqrt(3) * (grid_utils.WORLD_MAX - grid_utils.WORLD_MIN) / step_size
  )

  batch_sh = origins.shape[:-1]
  origins = origins.reshape(-1, 3)
  directions = directions.reshape(-1, 3)

  # Set up the rays and transform them to the voxel grid coordinate space.
  num_rays = origins.shape[0]
  for chunk_start in tqdm.tqdm(list(range(0, num_rays, chunk_size))):
    chunk_end = min(chunk_start + chunk_size, num_rays)
    origins_chunk = origins[chunk_start:chunk_end]
    directions_chunk = directions[chunk_start:chunk_end]
    rendering_chunk = _render_rays(
        origins_chunk,
        directions_chunk,
        sparse_grid_features,
        sparse_grid_density,
        sparse_grid_block_indices,
        planes_features,
        planes_density,
        deferred_mlp_vars,
        occupancy_grid,
        config.sparse_grid_resolution,
        grid_config['sparse_grid_voxel_size'],
        grid_config['triplane_voxel_size'],
        occupancy_voxel_size,
        config.data_block_size,
        num_steps,
        max_steps,
        step_size,
        bg_intensity,
        config.range_features,
        config.range_density,
        config.near,
    )
    rendering_chunk = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, x.shape[-1]), rendering_chunk
    )
    if chunk_start == 0:
      rendering = {
          key: jnp.zeros((num_rays, x.shape[-1]), dtype=jnp.float32)
          for key, x in rendering_chunk.items()
      }
    rendering = jax.tree_util.tree_map(
        lambda x, x_chunk: x.at[chunk_start:chunk_end].set(x_chunk),  # pylint: disable=cell-var-from-loop
        rendering,
        rendering_chunk,
    )
  rendering = jax.tree_util.tree_map(
      lambda x: x.reshape(*batch_sh, x.shape[-1]), rendering
  )
  return rendering

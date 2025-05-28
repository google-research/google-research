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

"""Triplane and voxel grid simulation."""

import itertools

import jax.numpy as jnp


# After contraction point lies within [-2, 2]^3. See coord.contract.
WORLD_MIN = -2.0
WORLD_MAX = 2.0


def calculate_voxel_size(resolution):
  return (WORLD_MAX - WORLD_MIN) / resolution


def grid_to_world(x_grid, voxel_size, xnp):
  """Converts grid coordinates [0, res]^3 to a world coordinates ([-2, 2]^3)."""
  # We also account for the fact that the grid is going to be queried in WebGL
  # by adopting WebGL's indexing: Y and Z coordinates are swapped and the X
  # coordinate is mirrored. Inverse of world_to_grid.

  x = xnp.empty_like(x_grid)

  def get_x():
    return (WORLD_MAX - voxel_size / 2) - voxel_size * x_grid[Ellipsis, 0]

  def get_yz():
    return (WORLD_MIN + voxel_size / 2) + voxel_size * x_grid[Ellipsis, [2, 1]]

  if xnp.__name__ == 'jax.numpy':
    x = x.at[Ellipsis, 0].set(get_x())
    x = x.at[Ellipsis, [1, 2]].set(get_yz())
  elif xnp.__name__ == 'numpy':
    x[Ellipsis, 0] = get_x()
    x[Ellipsis, [1, 2]] = get_yz()
  return x


def world_to_grid(x, voxel_size, xnp):
  """Converts a world coordinate (in [-2, 2]^3) to a grid coordinate [0, res]^3."""
  # Inverse of grid_to_world.
  x_grid = xnp.empty_like(x)

  def get_x():
    return ((WORLD_MAX - voxel_size / 2) - x[Ellipsis, 0]) / voxel_size

  def get_yz():
    return (x[Ellipsis, [1, 2]] - (WORLD_MIN + voxel_size / 2)) / voxel_size

  if xnp.__name__ == 'jax.numpy':
    x_grid = x_grid.at[Ellipsis, 0].set(get_x())
    x_grid = x_grid.at[Ellipsis, [2, 1]].set(get_yz())
  elif xnp.__name__ == 'numpy':
    x_grid[Ellipsis, 0] = get_x()
    x_grid[Ellipsis, [2, 1]] = get_yz()
  return x_grid


def calculate_num_evaluations_per_sample(config):
  """Calculates number of MLP evals required for each sample along the ray."""
  # For grid evaluation we need to evaluate the MLP multiple times per query
  # to the representation. The number of samples depends on wether a sparse
  # grid, tri-planes or both are used.
  assert (
      config.triplane_resolution > 0 or config.sparse_grid_resolution > 0
  ), 'one or both of these values needs to be specified'
  x = 0
  if config.triplane_resolution > 0:
    x += 3 * 4  # Three planes and 4 samples for bi-linear interpolation.
  if config.sparse_grid_resolution > 0:
    x += 8  # Tri-linear interpolation.
  return x


def calculate_grid_config(config):
  """Computes voxel sizes from grid resolutions."""
  # `voxel_size_to_use` is for instance used to infer the step size used during
  # rendering, which should equal to the voxel size of the finest grid
  # (tri-plane or sparse grid) that is used.
  triplane_voxel_size = calculate_voxel_size(config.triplane_resolution)
  sparse_grid_voxel_size = calculate_voxel_size(config.sparse_grid_resolution)
  # Assuming that triplane_resolution is higher than sparse_grid_resolution
  # when a tri-planes are used.
  voxel_size_to_use = (
      triplane_voxel_size
      if config.triplane_resolution > 0
      else sparse_grid_voxel_size
  )
  resolution_to_use = (
      config.triplane_resolution
      if config.triplane_resolution > 0
      else config.sparse_grid_resolution
  )
  return dict(
      triplane_voxel_size=triplane_voxel_size,
      sparse_grid_voxel_size=sparse_grid_voxel_size,
      voxel_size_to_use=voxel_size_to_use,
      resolution_to_use=resolution_to_use,
  )


def get_eval_positions_and_local_coordinates(positions, config, grid_config):
  """Given as input is a batch of positions of shape Lx3."""
  # Prepare grid simulation, the returned `positions` has the shape S*Lx3
  #   S = 1 if no grid simulation is used (no-op)
  #   S = 8 if only a 3D grid is simulated
  #   S = 3*4 = 12 if a if only tri-planes are simulated
  #   S = 20 if both tri-planes and a 3D grid are used (MERF)
  #   see: calculate_num_evaluations_per_sample
  #
  # For every query to our grid-based representation we have to perform S
  # queries to the grid which is parameterized by an MLP.
  #
  # Further we compute positions (∈ [0,1]^3) local to a texel/voxel,
  # which are later used to compute interpolation weights:
  #     triplane_positions_local, sparse_grid_positions_local: Lx3
  triplane_positions_local = sparse_grid_positions_local = None
  if config.triplane_resolution > 0:
    if config.sparse_grid_resolution > 0:
      sparse_grid_positions, sparse_grid_positions_local = (
          sparse_grid_get_eval_positions_and_local_coordinates(
              positions, grid_config['sparse_grid_voxel_size'], axis=1
          )
      )  # 8*Lx3 and Lx3.
    positions, triplane_positions_local = (
        triplane_get_eval_posititons_and_local_coordinates(
            positions, grid_config['triplane_voxel_size'], axis=1
        )
    )  # 12*Lx3 and Lx3.
    if config.sparse_grid_resolution > 0:
      # Concantenate sparse grid and tri-plane positions for MERF.
      positions = jnp.concatenate([sparse_grid_positions, positions], axis=1)
  else:  # implies config.sparse_grid_resolution > 0.
    positions, sparse_grid_positions_local = (
        sparse_grid_get_eval_positions_and_local_coordinates(
            positions, grid_config['sparse_grid_voxel_size'], axis=1
        )
    )  # 8*Lx3 and Lx3.
  positions = positions.reshape(-1, *positions.shape[2:])
  return positions, triplane_positions_local, sparse_grid_positions_local


def interpolate_based_on_local_coordinates(
    y, triplane_positions_local, sparse_grid_positions_local, config
):
  """Linearly interpolates values fetched from grid corners."""
  # Linearly interpolates values fetched from grid corners based on
  # blending weights computed from within-voxel/texel local coordinates.
  #
  # y: S*LxC
  # triplane_positions_local: Lx3
  # sparse_grid_positions_local: Lx3
  #
  # Output: LxC
  s = calculate_num_evaluations_per_sample(config)
  y = y.reshape(-1, s, *y.shape[1:])
  if config.triplane_resolution > 0:
    if config.sparse_grid_resolution > 0:
      sparse_grid_y, y = y.split([8], axis=1)
    r = triplane_interpolate_based_on_local_coordinates(
        y, triplane_positions_local, axis=1
    )
    if config.sparse_grid_resolution > 0:
      r += sparse_grid_interpolate_based_on_local_coordinates(
          sparse_grid_y, sparse_grid_positions_local, axis=1
      )
    return r
  else:  # implies sparse_grid_resolution is > 0.
    return sparse_grid_interpolate_based_on_local_coordinates(
        y, sparse_grid_positions_local, axis=1
    )


def sparse_grid_get_eval_positions_and_local_coordinates(x, voxel_size, axis):
  """Compute positions of 8 surrounding voxel corners and within-voxel coords."""
  x_grid = world_to_grid(x, voxel_size, jnp)
  x_floor = jnp.floor(x_grid)
  x_ceil = jnp.ceil(x_grid)
  local_coordinates = x_grid - x_floor
  positions_corner = []
  corner_coords = [[False, True] for _ in range(x.shape[-1])]
  for z in itertools.product(*corner_coords):
    l = []
    for i, b in enumerate(z):
      l.append(x_ceil[Ellipsis, i] if b else x_floor[Ellipsis, i])
    positions_corner.append(jnp.stack(l, axis=-1))
  positions_corner = jnp.stack(positions_corner, axis=axis)
  positions_corner = grid_to_world(positions_corner, voxel_size, jnp)
  return positions_corner, local_coordinates


def sparse_grid_interpolate_based_on_local_coordinates(
    y, local_coordinates, axis
):
  """Blend 8 MLP outputs based on weights computed from local coordinates."""
  y = jnp.moveaxis(y, axis, -2)
  res = jnp.zeros(y.shape[:-2] + (y.shape[-1],))
  corner_coords = [[False, True] for _ in range(local_coordinates.shape[-1])]
  for corner_index, z in enumerate(itertools.product(*corner_coords)):
    w = jnp.ones(local_coordinates.shape[:-1])
    for i, b in enumerate(z):
      w = w * (
          local_coordinates[Ellipsis, i] if b else (1 - local_coordinates[Ellipsis, i])
      )
    res = res + w[Ellipsis, None] * y[Ellipsis, corner_index, :]
  return res


def triplane_get_eval_posititons_and_local_coordinates(x, voxel_size, axis):
  """For each of the 3 planes return the 4 sampling positions at texel corners."""
  x_grid = world_to_grid(x, voxel_size, jnp)
  x_floor = jnp.floor(x_grid)
  x_ceil = jnp.ceil(x_grid)
  local_coordinates = x_grid - x_floor
  corner_coords = [
      [False, True] for _ in range(2)
  ]  # (0, 0), (0, 1), (1, 0), (1, 1).
  r = []
  for plane_idx in range(3):  # Index of the plane to project to.
    # Indices of the two ouf of three dims along which we bilineary interpolate.
    inds = [h for h in range(3) if h != plane_idx]
    for z in itertools.product(*corner_coords):
      l = [None for _ in range(3)]
      l[plane_idx] = jnp.zeros_like(x_grid[Ellipsis, 0])
      for i, b in enumerate(z):
        l[inds[i]] = x_ceil[Ellipsis, inds[i]] if b else x_floor[Ellipsis, inds[i]]
      r.append(jnp.stack(l, axis=-1))
  r = jnp.stack(r, axis=axis)
  return grid_to_world(r, voxel_size, jnp), local_coordinates


def triplane_interpolate_based_on_local_coordinates(y, local_coordinates, axis):
  """Blend 3*4=12 MLP outputs based on weights computed from local coordinates."""
  y = jnp.moveaxis(y, axis, -2)
  res = jnp.zeros(y.shape[:-2] + (y.shape[-1],))
  corner_coords = [[False, True] for _ in range(2)]
  query_index = 0
  for plane_idx in range(3):
    # Indices of the two ouf of three dims along which we bilineary interpolate.
    inds = [h for h in range(3) if h != plane_idx]
    for z in itertools.product(*corner_coords):
      w = jnp.ones(local_coordinates.shape[:-1])
      for i, b in enumerate(z):
        w = w * (
            local_coordinates[Ellipsis, inds[i]]
            if b
            else (1 - local_coordinates[Ellipsis, inds[i]])
        )
      res += w[Ellipsis, None] * y[Ellipsis, query_index, :]
      query_index += 1
  return res

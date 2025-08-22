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

import dataclasses
import itertools
import pprint

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
from smerf.internal import distill


# After contraction point lies within [-2, 2]^3. See coord.contract.
WORLD_MIN = -2.0
WORLD_MAX = 2.0

# Invalid index in forward mappings
INVALID_IDX = -1


def calculate_voxel_size(resolution):
  if resolution is None or resolution <= 0:
    resolution = 1.
  return (WORLD_MAX - WORLD_MIN) / resolution


def calculate_submodel_voxel_size(resolution):
  # While voxels for triplanes and sparse grids cover [-2, 2]^3 in squash
  # coordinates, voxels for submodels cover [-1, 1]^3 in world coordinates.
  # This assumption is met here:
  return calculate_voxel_size(resolution) / 2.0


def grid_to_world(x_grid, voxel_size, xnp):
  """Converts grid coordinates [0, res]^3 to a squash coordinates [-2, 2]^3."""
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
  """Converts squash coordinates [-2, 2]^3 to grid coordinates [0, res]^3."""
  # Inverse of grid_to_world.
  x_grid = xnp.empty_like(x)

  def get_x():
    # TODO(duckworthd): Understand why a voxel_size/2 offset is applied.
    # Squash space extends from [-2, 2]^2, but not all of it maps to a valid
    # grid coordinate!
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


def _calculate_grid_config(config):
  """Computes voxel sizes from grid resolutions."""
  # Calculate length of a voxel's side in squash space.
  triplane_voxel_size = calculate_voxel_size(config.triplane_resolution)
  sparse_grid_voxel_size = calculate_voxel_size(config.sparse_grid_resolution)

  # Calculate side-length of a submodel voxel in world coordinates. Cameras
  # origins are expected to be in [-1, 1].
  #
  # This parameter is *NOT* affected by config.submodel_scale_factor. It is
  # used to determine the mapping from ray to submodel and the location of
  # submodel coordinate system origins in world coordinates.
  submodel_voxel_size = calculate_submodel_voxel_size(
      config.submodel_grid_resolution
  )

  # This parameter controls the amount of dilation when converting between
  # world and submodel coordinates.
  #
  # To prevent cameras from walking up the the edge of the [-1, 1]^3, set
  # config.submodel_scale_factor < 1. For example, (1.0, -1.2, 0.5) in world
  # coordinates will map to (0.8, -0.96, 0.4) in submodel coordinates when
  # factor=0.8.
  submodel_scale = (
      config.submodel_grid_resolution * config.submodel_scale_factor
  )

  # The number of submodels to train. The submodel grid resolution splits each
  # dimension into M buckets, so with 3 dimensions there are M^3 buckets.
  num_submodels = config.submodel_grid_resolution ** 3

  # When training submodels independently, this option declares which submodel
  # is being trained on this host. If this option is None, all submodels are
  # trained on a single host.
  submodel_idx_override = config.submodel_idx_override

  # The number of submodels being trained on this host. Typically equal to the
  # number of submodels in total, except when training one submodel per host.
  if submodel_idx_override is None:
    num_local_submodels = num_submodels
    submodels_on_host = list(range(num_submodels))
  else:
    if not submodel_idx_override < num_submodels:
      raise ValueError(f'{submodel_idx_override=} >= {num_submodels=}')
    num_local_submodels = 1
    submodels_on_host = [submodel_idx_override]

  # If submodel_enable_multimlp=True, each submodel is assigned its own MLP
  # parameters. If False, the same MLP parameters are shared across all MLPs.
  if config.submodel_enable_multimlp:
    num_submodel_mlp_kernels = num_local_submodels
  else:
    num_submodel_mlp_kernels = 1

  # In all cases, each submodel is assigned its own set of hash encoding
  # parameters.
  num_submodel_hash_encoding_kernels = num_local_submodels

  # `voxel_size_to_use` is for instance used to infer the step size used during
  # rendering, which should equal to the voxel size of the finest grid that
  # is used.
  voxel_size_to_use = min(
      [triplane_voxel_size, sparse_grid_voxel_size]
  )
  resolution_to_use = max([
      config.triplane_resolution,
      config.sparse_grid_resolution,
  ])
  return flax.core.FrozenDict(
      triplane_voxel_size=triplane_voxel_size,
      sparse_grid_voxel_size=sparse_grid_voxel_size,
      submodel_voxel_size=submodel_voxel_size,
      submodel_idx_override=submodel_idx_override,
      submodels_on_host=tuple(submodels_on_host),
      submodel_scale=submodel_scale,
      voxel_size_to_use=voxel_size_to_use,
      resolution_to_use=resolution_to_use,
      num_submodels=num_submodels,
      num_local_submodels=num_local_submodels,
      num_submodel_mlp_kernels=num_submodel_mlp_kernels,
      num_submodel_hash_encoding_kernels=num_submodel_hash_encoding_kernels,
  )


def _create_forward_mapping(backward, num_idxs):
  """Constructs mappings {0..n} and {0..m} for m <= n.

  Args:
    backward: i32[k]. backward[j] == i means that input 'i' maps to output 'j'.
    num_idxs: int. Inputs 'i' are between {0, ..., num_idxs-1}.

  Returns:
    forward: i32[num_idxs]. forward[i] == j as defined by backward. All other
      inputs 'i' are set to INVALID_IDX.
  """
  assert len(backward.shape) == 1
  outputs = np.arange(len(backward))

  # idx -> pseudo_idx
  forward = np.full((num_idxs,), INVALID_IDX, dtype=np.int32)
  forward[backward] = outputs

  return forward


def _calculate_dataset_config(config, grid_config, datasets):
  """Calculates dataset-derived properties."""
  # Determine which sm_idxs are alive.
  camtoworlds = jnp.concatenate([d.camtoworlds for d in datasets], axis=0)
  params_to_sm, _ = distill.alive_sm_idxs(
      camtoworlds[:, 0:3, 3], config, grid_config,
  )
  logging.info(
      '%d of %d possible submodels have at least one training camera assigned'
      ' to them.', len(params_to_sm), grid_config['num_submodels'],
  )

  # Construct a mapping from sm to sparse_sm
  sm_to_params = _create_forward_mapping(
      params_to_sm, grid_config['num_submodels']
  )

  # The following parameters, also set in _calculate_grid_config(), control
  # the shape of the model with respect to the number of submodels. Submodels
  # that aren't assigned any train cameras are never updated, so there is no
  # need to instantiate their parameters.
  if config.submodel_idx_override is None:
    num_local_submodels = len(params_to_sm)
    submodels_on_host = params_to_sm.tolist()
  else:
    num_local_submodels = grid_config['num_local_submodels']
    submodels_on_host = grid_config['submodels_on_host']
  assert num_local_submodels <= grid_config['num_local_submodels']
  assert set(submodels_on_host) <= set(grid_config['submodels_on_host'])

  # Override the number of MLP kernels and hash encoding kernels, if necessary.
  if config.submodel_enable_multimlp:
    num_submodel_mlp_kernels = num_local_submodels
  else:
    num_submodel_mlp_kernels = grid_config['num_submodel_mlp_kernels']
  num_submodel_hash_encoding_kernels = num_local_submodels

  grid_config = flax.core.FrozenDict(
      num_local_submodels=num_local_submodels,
      num_submodel_mlp_kernels=num_submodel_mlp_kernels,
      num_submodel_hash_encoding_kernels=num_submodel_hash_encoding_kernels,
      sm_to_params=tuple(sm_to_params.tolist()),
      params_to_sm=tuple(params_to_sm.tolist()),
      submodels_on_host=tuple(submodels_on_host),
  )

  # Initialize exposure-related properties.
  exposure_config = flax.core.FrozenDict(
      default_exposure=None, exposure_range=None
  )

  # Not all dataset provide exposure information. If they do, populate the
  # exposure config. This is used in preprocess_rays() to ensure that a default
  # exposure value is available for the teacher.
  exposures = [d.exposures for d in datasets if d.exposures is not None]
  if exposures:
    exposures = np.concatenate(exposures, axis=0)
    exposure_config = flax.core.FrozenDict(
        default_exposure=np.median(exposures),
        exposure_range=(np.min(exposures), np.max(exposures)),
    )
    exposure_config = jax.tree_util.tree_map(
        lambda x: x.tolist(), exposure_config
    )

  return grid_config, exposure_config


def initialize_grid_config(config, datasets=None):
  """Initializes Config.grid_config.

  Args:
    config: configs.Config instance.
    datasets: Optional list of mipnerf360 Dataset instances. If this isn't
      provided, dataset-specific fields will not be set. You may specify
      multiple dataset instances (e.g. for train and test splits).

  Returns:
    Config with config.{grid_config, exposure_config} updated.
  """
  grid_properties = _calculate_grid_config(config)

  dataset_properties = exposure_properties = flax.core.FrozenDict({})
  if datasets is not None:
    dataset_properties, exposure_properties = _calculate_dataset_config(
        config, grid_properties, datasets
    )

  grid_config = flax.core.FrozenDict(
      flax.core.unfreeze(grid_properties)
      | flax.core.unfreeze(dataset_properties)
  )
  logging.info(
      'grid_config initialized:\n%s',
      pprint.pformat(flax.core.unfreeze(grid_config)),
  )
  config = dataclasses.replace(
      config,
      grid_config=grid_config,
      exposure_config=exposure_properties,
  )
  return config


def get_eval_positions_and_local_coordinates(
    sm_idxs, s_positions, config, grid_config
):
  """Computes positions in squash space to query the representation at.

  Args:
    sm_idxs: i32[n, 1]. Which submodel each point belongs to.
    s_positions: i32[n, 3]. Target positions in s-coordinates to query.
    config: configs.Config object.
    grid_config: Configuration for grid discretization.

  Returns:
    query_sm_idxs: i32[n*s, 1]. Submodel indices for each query point.
    query_s_positions: f32[n*s, 3]. s-positions to query representation at.
      Must be paired with query_sm_idxs to query the correct submodel.
    triplane_positions_local: f32[n, 3]. Positions within a triplane's voxel.
    sparse_grid_positions_local: f32[n, 3]. Positions within a sparse grid's
      voxel.
  """
  # Prepare grid simulation, the returned `positions` has the shape S*Lx3
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
  triplane_s_positions_local = None
  sparse_grid_s_positions_local = None

  query_sm_idxs = []
  query_s_positions = []

  # Important! Order is sparse_grid, then triplanes. See
  # interpolate_based_on_local_coordinates() below.
  if config.sparse_grid_resolution > 0:
    (
        sparse_grid_sm_idxs,
        sparse_grid_s_positions,
        sparse_grid_s_positions_local,
    ) = sparse_grid_get_eval_positions_and_local_coordinates(
        sm_idxs, s_positions, grid_config['sparse_grid_voxel_size'], axis=1
    )  # Lx8x1, Lx8x3, and Lx3.
    query_sm_idxs.append(sparse_grid_sm_idxs)
    query_s_positions.append(sparse_grid_s_positions)

  if config.triplane_resolution > 0:
    triplane_sm_idxs, triplane_s_positions, triplane_s_positions_local = (
        triplane_get_eval_positions_and_local_coordinates(
            sm_idxs, s_positions, grid_config['triplane_voxel_size'], axis=1
        )
    )  # Lx12x1, Lx12x3, and Lx3.
    query_sm_idxs.append(triplane_sm_idxs)
    query_s_positions.append(triplane_s_positions)

  query_sm_idxs = jnp.concatenate(query_sm_idxs, axis=1)  # LxSx1
  query_sm_idxs = query_sm_idxs.reshape(-1, *query_sm_idxs.shape[2:])  # L*Sx1

  query_s_positions = jnp.concatenate(query_s_positions, axis=1)  # LxSx3
  query_s_positions = query_s_positions.reshape(
      -1, *query_s_positions.shape[2:]
  )  # L*Sx3

  return (
      query_sm_idxs,
      query_s_positions,
      triplane_s_positions_local,
      sparse_grid_s_positions_local,
  )


def interpolate_based_on_local_coordinates(
    y,
    triplane_positions_local,
    sparse_grid_positions_local,
    config,
):
  """Linearly interpolates values fetched from grid corners.

  Args:
    y: f32[n*s, c]. Queried values to use for interpolation.
    triplane_positions_local: f32[n, c]. Positions within a triplane voxel.
    sparse_grid_positions_local: f32[n, c]. Positions within a sparse grid
      voxel.
    config: Config object.

  Returns:
    triplane_features: Optional f32[n, 3, c]. Feature vectors for each plane in
      the triplane representation.
    sparse_grid_features: Optional f32[n, c]. Feature vector from sparse voxel
      grid.
  """
  s = calculate_num_evaluations_per_sample(config)
  y = y.reshape(-1, s, *y.shape[1:])  # [L, S, C]

  triplane_features = sparse_grid_features = None
  if config.triplane_resolution > 0:
    sparse_grid_y = None
    if config.sparse_grid_resolution > 0:
      # Important! Order is sparse_grid, then triplanes. See
      # get_eval_positions_and_local_coordinates() above.
      sparse_grid_y, y = jnp.split(y, [8], axis=1)
    triplane_features = triplane_interpolate_based_on_local_coordinates(
        y, triplane_positions_local, axis=1
    )  # [n, 3, c]
    if config.sparse_grid_resolution > 0:
      sparse_grid_features = sparse_grid_interpolate_based_on_local_coordinates(
          sparse_grid_y, sparse_grid_positions_local, axis=1
      )
  else:  # implies sparse_grid_resolution is > 0.
    assert config.sparse_grid_resolution > 0, config.sparse_grid_resolution
    sparse_grid_features = sparse_grid_interpolate_based_on_local_coordinates(
        y, sparse_grid_positions_local, axis=1
    )
  return triplane_features, sparse_grid_features


def sum_with_triplane_weights(
    triplane_features,
    sparse_grid_features,
    triplane_density,
    sparse_grid_density,
    *,
    config,
):
  """Triplane features weighted by sparse grid's last feature.

  At least one of triplane_* or sparse_grid_* must be defined.

  Args:
    triplane_features: Optional f32[..., 3, c]
    sparse_grid_features: Optional f32[..., c]
    triplane_density: Optional f32[..., 3, 1]
    sparse_grid_density: Optional f32[..., 1]
    config: Config object.

  Returns:
    f32[..., c]. Features.
    f32[..., 1]. Density preactivations.
  """
  # Method for combining interpolated triplane and sparse voxel feature vectors.
  combine_ops = {
      'sum': sum,
      'coarse_sum': _coarse_sum,
      'cross_product_sum': _cross_product_sum,
  }

  if config.sparse_grid_resolution > 0:
    weights = sparse_grid_features[Ellipsis, -1:]  # [..., 1]

  def merge(triplane, sparse_grid, apply_weights, combine_op):
    contributions = []
    if triplane is not None:
      triplane = jnp.sum(triplane, axis=-2)  # [..., c]
      if apply_weights:
        assert weights is not None
        triplane = triplane * weights
      contributions.append(triplane)
    if sparse_grid is not None:
      contributions.append(sparse_grid)
    if not contributions:
      raise ValueError(
          'At least one of triplane or sparse_grid must be defined.'
      )
    return combine_op(contributions)

  features = merge(
      triplane_features,
      sparse_grid_features,
      config.use_low_res_features_as_weights,
      combine_ops[config.merge_features_combine_op],
  )
  density = merge(
      triplane_density,
      sparse_grid_density,
      config.use_triplane_weights_for_density,
      combine_ops['sum'],
  )

  return features, density


def sparse_grid_get_eval_positions_and_local_coordinates(
    sm_idxs, s_positions, voxel_size, axis
):
  """Compute positions of 8 surrounding voxel corners and within-voxel coords.

  Args:
    sm_idxs: i32[..., 1]. Which submodel to query for each point.
    s_positions: f23[..., 3]. Points to query in s-coordinates.
    voxel_size: length of a voxel's side in s-coordinates.
    axis: Which axis to stack on.

  Returns:
    sparse_grid_sm_idxs: i32[..., 8, 1]. Which submodel to query for each
      sparse_grid corner.
    sparse_grid_s_positions: f32[..., 8, 3]. sparse_grid corners in
      s-coordinates.
    sparse_grid_local_coordinates: f32[..., 3]. XYZ position with a sparse_grid
      voxel.
  """
  x_grid = world_to_grid(s_positions, voxel_size, jnp)
  x_floor = jnp.floor(x_grid)
  x_ceil = jnp.ceil(x_grid)
  local_coordinates = x_grid - x_floor
  positions_corner = []
  corner_coords = [[False, True] for _ in range(s_positions.shape[-1])]
  for z in itertools.product(*corner_coords):
    l = []
    for i, b in enumerate(z):
      l.append(x_ceil[Ellipsis, i] if b else x_floor[Ellipsis, i])
    positions_corner.append(jnp.stack(l, axis=-1))
  positions_corner = jnp.stack(positions_corner, axis=axis)
  sparse_grid_s_positions = grid_to_world(positions_corner, voxel_size, jnp)

  # Assign sm_idxs to each query position.
  sparse_grid_sm_idxs = jnp.expand_dims(sm_idxs, axis=axis)
  sparse_grid_sm_idxs = jnp.broadcast_to(
      sparse_grid_sm_idxs, (*positions_corner.shape[:-1], 1)
  )

  return sparse_grid_sm_idxs, sparse_grid_s_positions, local_coordinates


def sparse_grid_interpolate_based_on_local_coordinates(
    y, local_coordinates, axis
):
  """Blends MLP outputs from sparse voxel grid.

  Args:
    y: f32[..., 8, c]. Feature vector from 8 voxel corners.
    local_coordinates: f32[..., 3]. Position within a voxel.
    axis: int. Which axis in "y" contains the "8"

  Returns:
    f32[..., c]. trilerp'd feature vector.
  """
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


def triplane_get_eval_positions_and_local_coordinates(
    sm_idxs, s_positions, voxel_size, axis
):
  """For each of the 3 planes return the 4 sampling positions at texel corners.

  Args:
    sm_idxs: i32[..., 1]. Which submodel to query for each point.
    s_positions: f23[..., 3]. Points to query in s-coordinates.
    voxel_size: length of a voxel's side in s-coordinates.
    axis: Which axis to stack on.

  Returns:
    triplane_sm_idxs: i32[..., 12, 1]. Which submodel to query for each triplane
      corner.
    triplane_s_positions: f32[..., 12, 3]. Triplane corners in s-coordinates.
    triplane_local_coordinates: f32[..., 3]. XYZ position with a triplane voxel.
  """
  x_grid = world_to_grid(s_positions, voxel_size, jnp)
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
  triplane_s_positions = grid_to_world(r, voxel_size, jnp)

  # Assign sm_idxs to each query position.
  triplane_sm_idxs = jnp.expand_dims(sm_idxs, axis=axis)
  triplane_sm_idxs = jnp.broadcast_to(
      triplane_sm_idxs, (*triplane_s_positions.shape[:-1], 1)
  )

  return triplane_sm_idxs, triplane_s_positions, local_coordinates


def triplane_interpolate_based_on_local_coordinates(y, local_coordinates, axis):
  """Blends MLP outputs for three feature planes with bilerp.

  Args:
    y: f32[..., 12, c]. Feature vectors from query points, 4 per plane.
    local_coordinates: f32[..., 3]. Position within a texel, one per plane
    axis: int. Which axis in "y" contains the "12"

  Returns:
    f32[..., 3, c]. Feature vectors for each feature plane.
  """
  y = jnp.moveaxis(y, axis, -2)
  *batch, s, c = y.shape
  if s != 12:
    raise ValueError(
        'The second-to-last axis must have exactly 12 entries, 4 per plane.'
        f' Actual shape: {y.shape}'
    )
  result = []
  corner_coords = [[False, True] for _ in range(2)]
  query_index = 0
  for plane_idx in range(3):
    # Indices of the two ouf of three dims along which we bilerp.
    inds = [h for h in range(3) if h != plane_idx]
    # Container for features from this plane.
    features = jnp.zeros((*batch, c), dtype=y.dtype)
    for z in itertools.product(*corner_coords):
      w = jnp.ones(local_coordinates.shape[:-1])  # [...]
      for i, b in enumerate(z):
        w = w * (
            local_coordinates[Ellipsis, inds[i]]
            if b
            else (1 - local_coordinates[Ellipsis, inds[i]])
        )
      features = features + w[Ellipsis, None] * y[Ellipsis, query_index, :]  # [..., c]
      query_index += 1
    result.append(features)
  result = jnp.stack(result, axis=-2)  # [..., 3, c]
  return result


def _coarse_sum(xs):
  """Concatenates triplane + sparse voxel to sparse voxel features.."""
  assert len(xs) == 2
  triplane, sparse_grid = xs
  return jnp.concatenate([triplane + sparse_grid, sparse_grid], axis=-1)


def _cross_product_sum(xs):
  """Computes cross-product of triplane & sparse voxel features."""
  assert len(xs) == 2
  triplane, sparse_grid = xs

  # Compute cross-product features
  cross_prod = jnp.einsum('...i,...j->...ij', triplane, sparse_grid)

  # Merge cross-product dimensions
  batch_dims = triplane.shape[:-1]
  cross_prod = jnp.reshape(cross_prod, (*batch_dims, -1))

  # Concatenate to sum.
  return jnp.concatenate([triplane + sparse_grid, cross_prod], axis=-1)

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

"""JAX implementation of multi_hash_resample_3d."""

import jax
import jax.numpy as jnp


@jax.named_scope('multi_resample_3d')
def multi_resample_3d(
    data,
    data_idxs,
    locations,
    edge_behavior='CLAMP',
):
  """Resamples input data at the provided locations from a volume.

  Assumes voxel centers at 0.5. If X = Y = Z = 5, then values are anchored at
  {0.5, ..., 4.5}^3 and locations outside are clamped to that range.

  Args:
    data: f32[K, X, Y, Z, C] array containing values to gather from.
    data_idxs: i32[N, 1] array containing integer indices for data's 0th
      dimension. Values must be in {0, ..., K-1}. Behavior undefined if outside
      of this range.
    locations: f32[N, 3] array containing 3D floating point locations to sample
      data at. Locations are in x,y,z order.
    edge_behavior: The behaviour for sample points outside of params. - CLAMP:
      Clamps to volume.

  Returns:
    A tensor of shape f32[..., C] containing the sampled values.
  """
  assert len(data.shape) == 5, data.shape
  assert len(data_idxs.shape) == len(locations.shape), (
      data_idxs.shape,
      locations.shape,
  )
  assert data_idxs.shape[-1] == 1, data_idxs.shape
  assert locations.shape[-1] == 3, locations.shape
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']

  num_hashmaps, *num_spatial_buckets, num_channels = data.shape  # pylint: disable=unused-variable
  *batch_shape, _ = locations.shape

  # Half-pixel offset.
  locations = locations - 0.5

  if edge_behavior != 'CLAMP':
    raise NotImplementedError(
        'CUDA implementation only supports CLAMP. CONSTANT_OUTSIDE has been'
        ' disabled to prevent divergent behavior.'
    )

  floored = jnp.floor(locations)
  ceil = floored + 1.0

  # Trilinearly interpolates by finding the weighted sum of the eight corner
  # points.
  positions = [
      jnp.stack([floored[Ellipsis, 0], floored[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1),
      jnp.stack([floored[Ellipsis, 0], floored[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1),
      jnp.stack([floored[Ellipsis, 0], ceil[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1),
      jnp.stack([floored[Ellipsis, 0], ceil[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1),
      jnp.stack([ceil[Ellipsis, 0], floored[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1),
      jnp.stack([ceil[Ellipsis, 0], floored[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1),
      jnp.stack([ceil[Ellipsis, 0], ceil[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1),
      jnp.stack([ceil[Ellipsis, 0], ceil[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1),
  ]
  ceil_w = locations - floored
  floor_w = 1.0 - ceil_w
  weights = [
      floor_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
      floor_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
      floor_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
      floor_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
      ceil_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
      ceil_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
      ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
      ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
  ]
  max_indices = jnp.array(num_spatial_buckets, dtype=jnp.int32) - 1

  output = jnp.zeros((*batch_shape, num_channels), dtype=data.dtype)

  for position, weight in zip(positions, weights):
    position_indexes = position.astype(jnp.int32)
    position_indexes = jnp.maximum(position_indexes, 0)
    position_indexes = jnp.minimum(position_indexes, max_indices)
    weighted_gathered = (
        multi_gather_volume(data, data_idxs, position_indexes)
        * weight[Ellipsis, None]
    )
    output += weighted_gathered

  expected_shape = (*batch_shape, num_channels)
  assert output.shape == expected_shape, f'{output.shape} != {expected_shape}'
  return output.astype(data.dtype)


@jax.named_scope('multi_gather_volume')
def multi_gather_volume(data, data_idxs, locations):
  """Gather from data at locations.

  Args:
    data: A f32[K, X, Y, Z, C] tensor.
    data_idxs: A i32[..., 1] tensor of indices for 'data'. Values in [0, K).
    locations: A i32[..., 3] tensor containing the locations to sample at.

  Returns:
    A [..., C] tensor containing the gathered locations.
  """
  x_coordinate = locations[Ellipsis, 0]
  y_coordinate = locations[Ellipsis, 1]
  z_coordinate = locations[Ellipsis, 2]

  # Use Advanced indexing to gather data data.
  assert (
      data_idxs.shape[:-1]
      == x_coordinate.shape
      == y_coordinate.shape
      == z_coordinate.shape
  ), (
      data_idxs.shape,
      x_coordinate.shape,
      y_coordinate.shape,
      z_coordinate.shape,
  )
  return data[data_idxs[Ellipsis, 0], x_coordinate, y_coordinate, z_coordinate]

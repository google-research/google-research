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


@jax.named_scope('multi_hash_resample_3d')
def multi_hash_resample_3d(data, data_idxs, locations):
  """Resamples input data at the provided locations from a hash table.

  Args:
    data: A f32[K, D, C] tensor from which to sample.
    data_idxs: A i32[..., 1] tensor of indices for 'data'. Values in [0, K).
    locations: A f32[..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.

  Returns:
    A tensor of shape f32[..., C] containing the sampled values.
  """
  assert len(data.shape) == 3, data.shape
  assert len(data_idxs.shape) == len(locations.shape), (
      data_idxs.shape,
      locations.shape,
  )
  assert data_idxs.shape[-1] == 1, data_idxs.shape
  assert locations.shape[-1] == 3, locations.shape

  num_hashmaps, num_hash_buckets, num_channels = data.shape  # pylint: disable=unused-variable
  *batch_shape, _ = locations.shape

  locations = locations - 0.5
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
  output = None
  for position, weight in zip(positions, weights):
    # Going via int32 enables the wraparound signed -> unsigned conversion which
    # matches the CUDA kernel.
    position = position.astype(jnp.int32).astype(jnp.uint32)
    pi_2 = 19349663
    pi_3 = 83492791

    bucket_idxs = jnp.mod(
        jnp.bitwise_xor(
            position[Ellipsis, 0],
            jnp.bitwise_xor(position[Ellipsis, 1] * pi_2, position[Ellipsis, 2] * pi_3),
        ),
        num_hash_buckets,
    ).astype(jnp.int32)
    # TODO(duckworthd): Consider linearizing these indices.
    weighted_gathered = data[data_idxs[Ellipsis, 0], bucket_idxs] * weight[Ellipsis, None]
    if output is None:
      output = weighted_gathered
    else:
      output += weighted_gathered

  expected_shape = (*batch_shape, num_channels)
  assert output.shape == expected_shape, f'{output.shape} != {expected_shape}'
  return output

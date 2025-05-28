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

"""JAX hashed resample implementations.

See paper at  https://nvlabs.github.io/instant-ngp/ for details.
"""
import jax.numpy as jnp


def hash_resample_3d(
    data, locations, method='TRILINEAR', half_pixel_center=True
):
  """Resamples input data at the provided locations from a hash table.

  Args:
    data: A [D, C] tensor from which to sample.
    locations: A [D, ..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.
    method: The interpolation kernel to use, must be 'TRILINEAR' or 'NEAREST'.
    half_pixel_center: A bool that determines if half-pixel centering is used.

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) == 2

  if method == 'TRILINEAR':
    if half_pixel_center:
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
  elif method == 'NEAREST':
    # Interpolate into the nearest cell. A weight of `None` is treated as 1.
    positions = [(jnp.floor if half_pixel_center else jnp.round)(locations)]
    weights = [None]
  else:
    raise ValueError('interpolation method {method} not supported')

  output = None
  for position, weight in zip(positions, weights):
    position = position.astype(jnp.int32).astype(jnp.uint32)
    pi_2 = 19349663
    pi_3 = 83492791

    data_indexes = jnp.mod(
        jnp.bitwise_xor(
            position[Ellipsis, 0],
            jnp.bitwise_xor(position[Ellipsis, 1] * pi_2, position[Ellipsis, 2] * pi_3),
        ),
        data.shape[0],
    ).astype(jnp.int32)
    gathered = data[(data_indexes,)]
    weighted_gathered = (
        gathered if weight is None else gathered * weight[Ellipsis, None]
    )
    if output is None:
      output = weighted_gathered
    else:
      output += weighted_gathered

  return output
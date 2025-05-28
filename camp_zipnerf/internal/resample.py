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

"""JAX resample implementations."""

import functools
import jax
import jax.numpy as jnp
import numpy as np


def gather_volume(data, locations, coordinate_order='xyz'):
  """Gather from data at locations.

  Args:
    data: A [D, H, W, C] tensor.
    locations: A [D, ..., 3] int32 tensor containing the locations to sample at.
    coordinate_order: Whether the sample locations are x,y,z or z,y,x.

  Returns:
    A [D, ..., C] tensor containing the gathered locations.
  """
  if coordinate_order == 'xyz':
    x_coordinate = locations[Ellipsis, 0]
    y_coordinate = locations[Ellipsis, 1]
    z_coordinate = locations[Ellipsis, 2]
  elif coordinate_order == 'zyx':
    z_coordinate = locations[Ellipsis, 0]
    y_coordinate = locations[Ellipsis, 1]
    x_coordinate = locations[Ellipsis, 2]

  # Use Advanced indexing to gather data data.
  return data[z_coordinate, y_coordinate, x_coordinate]


def resample_3d(
    data,
    locations,
    edge_behavior='CONSTANT_OUTSIDE',
    constant_values=0.0,
    coordinate_order='xyz',
    method='TRILINEAR',
    half_pixel_center=False,
):
  """Resamples input data at the provided locations from a volume.

  Args:
    data: A [D, H, W, C] tensor from which to sample.
    locations: A [D, ..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.
    edge_behavior: The behaviour for sample points outside of params.
      -CONSTANT_OUTSIDE: First pads params by 1 with constant_values in the
      x-y-z dimensions, then clamps samples to this padded tensor. The effect is
      that sample points interpolate towards the constant value just outside the
      tensor. -CLAMP: clamps to volume.
    constant_values: The constant value to use with edge_behvaior
      'CONSTANT_OUTSIDE.'
    coordinate_order: Whether the sample locations are x,y,z or z,y,x.
    method: The interpolation kernel to use, must be 'TRILINEAR' or 'NEAREST'.
    half_pixel_center: A bool that determines if half-pixel centering is used.

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) >= 3
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']
  if edge_behavior == 'CONSTANT_OUTSIDE':
    data = jnp.pad(
        data,
        np.array([[1, 1], [1, 1], [1, 1]] + (data.ndim - 3) * [[0, 0]]),
        constant_values=constant_values,
    )
    locations = locations + 1.0

  if method == 'TRILINEAR':
    # Trilinearly interpolates by finding the weighted sum of the eight corner
    # points.
    if half_pixel_center:
      locations = locations - 0.5
    floored = jnp.floor(locations)
    ceil = floored + 1.0
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

  max_indices = jnp.array(data.shape[:3], dtype=jnp.int32) - 1
  if coordinate_order == 'xyz':
    max_indices = jnp.flip(max_indices)

  output = jnp.zeros((*locations.shape[:-1], data.shape[-1]), dtype=data.dtype)

  for position, weight in zip(positions, weights):
    indexes = position.astype(jnp.int32)

    indexes = jnp.maximum(indexes, 0)
    indexes = jnp.minimum(indexes, max_indices)
    gathered = gather_volume(data, indexes, coordinate_order)
    weighted_gathered = (
        gathered if weight is None else gathered * weight[Ellipsis, None]
    )
    output += weighted_gathered

  return output.astype(data.dtype)

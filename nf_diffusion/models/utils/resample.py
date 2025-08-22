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

"""Pure JAX resample implementations."""

import functools
import jax
import jax.numpy as jnp
import numpy as np


def gather_batch_2d(data, locations, coordinate_order='xy'):
  """Gather from data at locations.

  Args:
    data: A [D, H1, W1, C] or [1, H1, W1, C] tensor. If the leading dimension is
      1 the tensor will be effectively broadcast along the z dimension.
    locations: A [D, ..., 2] int32 tensor containing the locations to sample at.
    coordinate_order: Whether the sample locations are x,y or y,x.

  Returns:
    A [D, ..., C] tensor containing the gathered locations.
  """
  if coordinate_order == 'xy':
    x_coordinate = locations[Ellipsis, 0]
    y_coordinate = locations[Ellipsis, 1]
  elif coordinate_order == 'yx':
    y_coordinate = locations[Ellipsis, 0]
    x_coordinate = locations[Ellipsis, 1]

  if data.shape[0] == 1:
    z_coordinate = jnp.zeros_like(x_coordinate)
  else:
    z_coordinate = jnp.arange(0, locations.shape[0])
    for _ in x_coordinate.shape[1:]:
      z_coordinate = z_coordinate[Ellipsis, jnp.newaxis]

  # Use Advanced indexing to gather data data.
  return data[z_coordinate, y_coordinate, x_coordinate]


# TODO(guandao) unable to jit this since str is not a valid JAX type
def resample_2d(data,
                locations,
                edge_behavior='CONSTANT_OUTSIDE',
                constant_values=0.0,
                coordinate_order='xy'):
  """Resamples input data at the provided locations.

  'locations' contains the sampling locations for each plane, and the
  corresponding plane of data is sampled these locations.

  Args:
    data: A [D, H1, W1, C] or [1, H1, W1, C] tensor from which to sample, if the
      leading dimension is 1 the tensor will be broadcast along the leading
      dimension.
    locations: A [..., 2] containing floating point locations to sample data at.
      Like tf.resampler, assumes pixel centers at integer coordinates. If the
      leading dimension of locations is 1 it will be broadcast over D.
    edge_behavior: The behaviour for sample points outside of params.
       -CONSTANT_OUTSIDE: First pads params by 1 with constant_values in the x-y
         dimensions, then clamps samples to this padded tensor. The effect is
         that sample points interpolate towards the constant value just outside
         the tensor.
       -CLAMP: clamps to volume.
    constant_values: The constant value to use with edge_behvaior
      'CONSTANT_OUTSIDE.'
    coordinate_order: Whether the sample locations are x,y or y,x.

  Returns:
    If locations.shape[0] != 1 then a tensor of shape locations.shape[:-1] + [C]
    containing the sampled values.
    If locations.shape[0] == 1 then a tensor of shape
    data.shape[0] + locations.shape[1:-1] + [C] containing the sampled values.
  """
  loc_ndim = len(locations.shape)
  if loc_ndim == 2:
    locations = jnp.expand_dims(locations, axis=0)
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']
  assert coordinate_order in ['xy', 'yx']
  if locations.shape[0] == 1:
    locations = jnp.broadcast_to(locations,
                                 data.shape[:1] + locations.shape[1:])
  if edge_behavior == 'CONSTANT_OUTSIDE':
    paddings = ((0, 0), (1, 1), (1, 1), (0, 0))
    data = jnp.pad(data, paddings, constant_values=constant_values)
    locations = locations + 1.0

  floored = jnp.floor(locations)
  ceil = floored + 1.0

  # Bilinear interpolates by finding the weighted sum of the four corner points.
  positions = [
      jnp.stack([floored[Ellipsis, 0], floored[Ellipsis, 1]], axis=-1),
      jnp.stack([floored[Ellipsis, 0], ceil[Ellipsis, 1]], axis=-1),
      jnp.stack([ceil[Ellipsis, 0], floored[Ellipsis, 1]], axis=-1),
      jnp.stack([ceil[Ellipsis, 0], ceil[Ellipsis, 1]], axis=-1)
  ]
  ceil_w = locations - floored
  floor_w = 1.0 - ceil_w
  weights = [
      floor_w[Ellipsis, 0] * floor_w[Ellipsis, 1], floor_w[Ellipsis, 0] * ceil_w[Ellipsis, 1],
      ceil_w[Ellipsis, 0] * floor_w[Ellipsis, 1], ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1]
  ]

  max_indices = jnp.array(data.shape[1:3], dtype=jnp.int32) - 1
  if coordinate_order == 'xy':
    max_indices = jnp.flip(max_indices)

  output = jnp.zeros((*locations.shape[:-1], data.shape[-1]), dtype=data.dtype)
  for position, weight in zip(positions, weights):
    indexes = position.astype(jnp.int32)
    indexes = jnp.maximum(indexes, 0)
    indexes = jnp.minimum(indexes, max_indices)
    weighted_gathered = gather_batch_2d(
        data, indexes, coordinate_order) * jnp.expand_dims(weight, axis=-1)
    output += weighted_gathered

  if loc_ndim == 2:
    output = jnp.squeeze(output, axis=0)
  assert len(output.shape) == loc_ndim
  return output.astype(data.dtype)


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
  else:
    raise ValueError("Invalid coord_order {}".format(coordinate_order))

  # Use Advanced indexing to gather data data.
  return data[z_coordinate, y_coordinate, x_coordinate]


def resample_3d(data,
                locations,
                edge_behavior='CONSTANT_OUTSIDE',
                constant_values=0.0,
                coordinate_order='xyz'):
  """Resamples input data at the provided locations from a volume.

  Args:
    data: A [D, H, W, C] tensor from which to sample.
    locations: A [D, ..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.
    edge_behavior: The behaviour for sample points outside of params.
       -CONSTANT_OUTSIDE: First pads params by 1 with constant_values in the
         x-y-z dimensions, then clamps samples to this padded tensor. The effect
         is that sample points interpolate towards the constant value just
         outside the tensor.
       -CLAMP: clamps to volume.
    constant_values: The constant value to use with edge_behvaior
      'CONSTANT_OUTSIDE.'
    coordinate_order: Whether the sample locations are x,y,z or z,y,x.

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) >= 3
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']
  if edge_behavior == 'CONSTANT_OUTSIDE':
    data = jnp.pad(
        data,
        np.array([[1, 1], [1, 1], [1, 1]] + (data.ndim - 3) * [[0, 0]]),
        constant_values=constant_values)
    locations = locations + 1.0

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
      jnp.stack([ceil[Ellipsis, 0], ceil[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1)
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
      ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * ceil_w[Ellipsis, 2]
  ]
  max_indices = jnp.array(data.shape[:3], dtype=jnp.int32) - 1
  if coordinate_order == 'xyz':
    max_indices = jnp.flip(max_indices)

  output = jnp.zeros((*locations.shape[:-1], data.shape[-1]), dtype=data.dtype)

  for position, weight in zip(positions, weights):
    indexes = position.astype(jnp.int32)

    indexes = jnp.maximum(indexes, 0)
    indexes = jnp.minimum(indexes, max_indices)
    weighted_gathered = gather_volume(
        data, indexes, coordinate_order) * weight[Ellipsis, jnp.newaxis]
    output += weighted_gathered

  return output.astype(data.dtype)


def splat_2d(data, locations, output_dims, coordinate_order='xy'):
  """"Splats input data at the provided locations.

  'locations' contains the splatting locations for each batch, and the
  corresponding batch of data is splatted at these locations.

  Args:
    data: A [{D, 1} H, W, C] tensor to splat into the output. If the batch
      dimension is 1 it will be broadcast with locations.
    locations: A [{D, 1}, H, W, 2] containing floating point splat locations.
      Like tf.resampler, assumes pixel centers at integer coordinates. If the
      batch dimension is 1 it will be broadcast with data.
    output_dims: The height and width of the output array.
    coordinate_order: A string, either 'xy' or 'yx' to specify the order of the
      coordinates stored in the locations parameter.

  Returns:
    A [{D, 1} , output_dims[0], output_dims[1], C] array containing splatted
    values.
  """

  if data.shape[0] == 1:
    data = data[0]
    in_axes = (None,)
  else:
    in_axes = (0,)
  if locations.shape[0] == 1:
    locations = locations[0]
    in_axes = in_axes + (None,)
  else:
    in_axes = in_axes + (0,)

  if in_axes[0] is None and in_axes[1] is None:
    return _splat_batch(
        data,
        locations,
        output_dims=output_dims,
        coordinate_order=coordinate_order)[jnp.newaxis]
  else:
    return jax.vmap(
        functools.partial(
            _splat_batch,
            output_dims=output_dims,
            coordinate_order=coordinate_order),
        in_axes=in_axes)(data, locations)


def _splat_batch(data, locations, output_dims, coordinate_order='xy'):
  """Splats a single batch of data. See splat_2d for details."""

  # Make the output one pixel bigger on all sides  so we can splat and not
  # worry about clipping issues.
  output = jnp.zeros((output_dims[0] + 2, output_dims[1] + 2, data.shape[-1]),
                     dtype=data.dtype)

  # Add 1 to coordinates to  account for expanded output.
  if coordinate_order == 'xy':
    x = locations[Ellipsis, 0] + 1.0
    y = locations[Ellipsis, 1] + 1.0
  else:
    assert coordinate_order == 'yx'
    x = locations[Ellipsis, 1] + 1.0
    y = locations[Ellipsis, 0] + 1.0

  x_0 = jnp.floor(x).astype(int)
  x_1 = x_0 + 1
  y_0 = jnp.floor(y).astype(int)
  y_1 = y_0 + 1

  x_0 = jnp.clip(x_0, 0, output.shape[1] - 1)
  x_1 = jnp.clip(x_1, 0, output.shape[1] - 1)
  y_0 = jnp.clip(y_0, 0, output.shape[0] - 1)
  y_1 = jnp.clip(y_1, 0, output.shape[0] - 1)

  output = output.at[y_0, x_0, :].add(data * ((x_1 - x) *
                                              (y_1 - y))[Ellipsis, jnp.newaxis])
  output = output.at[y_1, x_0, :].add(data * ((x_1 - x) *
                                              (y - y_0))[Ellipsis, jnp.newaxis])
  output = output.at[y_0, x_1, :].add(data * ((x - x_0) *
                                              (y_1 - y))[Ellipsis, jnp.newaxis])
  output = output.at[y_1, x_1, :].add(data * ((x - x_0) *
                                              (y - y_0))[Ellipsis, jnp.newaxis])
  # Remove the one pixel border.
  return output[1:-1, 1:-1]

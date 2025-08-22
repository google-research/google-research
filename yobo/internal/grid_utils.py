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

# pylint: skip-file
"""Voxel grid interpolation and Instant NGP hash encoding utility functions."""

# This Python/Jax program is a re-implementation of the multiresolution
# hash encoding structure described in Section 3 of the
# Instant Neural Graphics Primitives SIGGRAPH 2022 paper by
# Müller, Evans, Schied, and Keller.
#  see https://github.com/NVlabs/instant-ngp

import enum
import functools
from typing import Union

from flax import linen as nn
import gin
from google_research.yobo.internal import math
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp

# A bounding box defined as a tuple containing (min_coord, max_coord).
BboxType = tuple[tuple[float, float, float], tuple[float, float, float]]


def jax_hash_resample_3d(
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
    # Going via int32 enables the wraparound signed -> unsigned conversion which
    # matches the CUDA kernel.
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


def jax_hash_simlex_resample_3d(data, locations):
  """Uses simplex interpolation to resample 3D data from a hash table.

  Args:
    data: A [D, C] tensor from which to sample.
    locations: A [D, ..., 3] containing floating point locations to sample data
      at. Assumes voxels centers at integer coordinates.

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) == 2
  locations = locations - 0.5
  floored = jnp.floor(locations)
  voxel_locations = locations - floored

  # First we find out which simplex we are in by comparing each axis of the
  # offset inside the voxel (x > y, y > z, z > x).
  e = (
      voxel_locations - jnp.roll(voxel_locations, shift=-1, axis=-1) > 0
  ).astype(locations.dtype)
  er = jnp.roll(e, shift=1, axis=-1)

  # These are offset vectors defining the relative location of each vertex in
  # the simplex.
  io0 = jnp.zeros_like(locations)
  io1 = e * (1.0 - er)
  io2 = 1.0 - er * (1.0 - e)
  io3 = jnp.ones_like(locations)

  # Using this information, we can project the sample location to barycentric
  # coordinates for the simplex it's in.

  # This integer tells us which of the six simplices we are in.
  simplex_id = (jnp.dot(e, jnp.array([4, 2, 1])) - 1).astype(jnp.int32)
  simplex_id = jnp.maximum(0, simplex_id)

  # Here are the precomputed 3x3 matrices that project the 3D point into the
  # barycentric coordinates of each simplex.
  inverse_basis_tensor = jnp.array([
      # Case 0: z > y > x
      [[0.0, -1.0, 1.0], [-1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
      # Case 1: y > x > z
      [[-1.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
      # Case 2: y > z > x
      [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -0.0]],
      # Case 3: x > z > y
      [[1.0, 0.0, -1.0], [0.0, -1.0, 1.0], [0.0, 1.0, 0.0]],
      # Case 4: z > x > y
      [[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
      # Case 5: x > y > z
      [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
  ])
  inverse_basis = inverse_basis_tensor[simplex_id, Ellipsis]

  # Now compute the barycentric weights for the sample.
  bcd = jnp.matmul(inverse_basis, voxel_locations[Ellipsis, None])[Ellipsis, 0]
  a = 1.0 - jnp.sum(bcd, axis=-1, keepdims=True)
  weights = jnp.concatenate([a, bcd], axis=-1)

  output = None
  pi_2 = 19349663
  pi_3 = 83492791
  for io, weight in zip(
      [io0, io1, io2, io3],
      [weights[Ellipsis, 0], weights[Ellipsis, 1], weights[Ellipsis, 2], weights[Ellipsis, 3]],
  ):
    # Going via int32 enables the wraparound signed -> unsigned conversion which
    # matches the CUDA kernel.
    vertex_positions = floored + io
    vertex_positions = vertex_positions.astype(jnp.int32)
    data_indices = jnp.mod(
        jnp.bitwise_xor(
            vertex_positions[Ellipsis, 0],
            jnp.bitwise_xor(
                vertex_positions[Ellipsis, 1] * pi_2, vertex_positions[Ellipsis, 2] * pi_3
            ),
        ),
        data.shape[0],
    ).astype(jnp.int32)
    weighted_gathered = data[(data_indices,)] * weight[Ellipsis, None]

    if output is None:
      output = weighted_gathered
    else:
      output += weighted_gathered

  return output


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


def jax_resample_2d(
    data,
    locations,
    edge_behavior='CONSTANT_OUTSIDE',
    constant_values=0.0,
    coordinate_order='xy',
):
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
      dimensions, then clamps samples to this padded tensor. The effect is that
      sample points interpolate towards the constant value just outside the
      tensor. -CLAMP: clamps to volume.
    constant_values: The constant value to use with edge_behvaior
      'CONSTANT_OUTSIDE.'
    coordinate_order: Whether the sample locations are x,y or y,x.

  Returns:
    If locations.shape[0] != 1 then a tensor of shape locations.shape[:-1] + [C]
    containing the sampled values.
    If locations.shape[0] == 1 then a tensor of shape
    data.shape[0] + locations.shape[1:-1] + [C] containing the sampled values.
  """

  assert len(locations.shape) >= 3
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']
  assert coordinate_order in ['xy', 'yx']
  if locations.shape[0] == 1:
    locations = jnp.broadcast_to(
        locations, data.shape[:1] + locations.shape[1:]
    )
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
      jnp.stack([ceil[Ellipsis, 0], ceil[Ellipsis, 1]], axis=-1),
  ]
  ceil_w = locations - floored
  floor_w = 1.0 - ceil_w
  weights = [
      floor_w[Ellipsis, 0] * floor_w[Ellipsis, 1],
      floor_w[Ellipsis, 0] * ceil_w[Ellipsis, 1],
      ceil_w[Ellipsis, 0] * floor_w[Ellipsis, 1],
      ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1],
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
        data, indexes, coordinate_order
    ) * jnp.expand_dims(weight, axis=-1)
    output += weighted_gathered

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

  # Use Advanced indexing to gather data data.
  return data[z_coordinate, y_coordinate, x_coordinate]


def jax_resample_3d(
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
        onp.array([[1, 1], [1, 1], [1, 1]] + (data.ndim - 3) * [[0, 0]]),
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


def splat_2d(data, locations, output_dims, coordinate_order='xy'):
  """Splats input data at the provided locations.

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
        coordinate_order=coordinate_order,
    )[jnp.newaxis]
  else:
    return jax.vmap(
        functools.partial(
            _splat_batch,
            output_dims=output_dims,
            coordinate_order=coordinate_order,
        ),
        in_axes=in_axes,
    )(data, locations)


def _splat_batch(data, locations, output_dims, coordinate_order='xy'):
  """Splats a single batch of data. See splat_2d for details."""

  # Make the output one pixel bigger on all sides  so we can splat and not
  # worry about clipping issues.
  output = jnp.zeros(
      (output_dims[0] + 2, output_dims[1] + 2, data.shape[-1]), dtype=data.dtype
  )

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

  output = output.at[y_0, x_0, :].add(
      data * ((x_1 - x) * (y_1 - y))[Ellipsis, jnp.newaxis]
  )
  output = output.at[y_1, x_0, :].add(
      data * ((x_1 - x) * (y - y_0))[Ellipsis, jnp.newaxis]
  )
  output = output.at[y_0, x_1, :].add(
      data * ((x - x_0) * (y_1 - y))[Ellipsis, jnp.newaxis]
  )
  output = output.at[y_1, x_1, :].add(
      data * ((x - x_0) * (y - y_0))[Ellipsis, jnp.newaxis]
  )
  # Remove the one pixel border.
  return output[1:-1, 1:-1]


def jax_simplex_resample_3d(
    data,
    locations,
    edge_behavior='CONSTANT_OUTSIDE',
    constant_values=0.0,
    coordinate_order='xyz',
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

  Returns:
    A tensor of shape [D, ..., C] containing the sampled values.
  """

  assert len(data.shape) >= 3
  assert edge_behavior in ['CONSTANT_OUTSIDE', 'CLAMP']
  if edge_behavior == 'CONSTANT_OUTSIDE':
    data = jnp.pad(
        data,
        onp.array([[1, 1], [1, 1], [1, 1]] + (data.ndim - 3) * [[0, 0]]),
        constant_values=constant_values,
    )
    locations = locations + 1.0

  floored = jnp.floor(locations)
  voxel_locations = locations - floored

  # First we find out which simplex we are in by comparing each axis of the
  # offset inside the voxel (x > y, y > z, z > x).
  e = (
      voxel_locations - jnp.roll(voxel_locations, shift=-1, axis=-1) > 0
  ).astype(locations.dtype)
  er = jnp.roll(e, shift=1, axis=-1)

  # These are offset vectors defining the relative location of each vertex in
  # the Simplex.
  io0 = jnp.zeros_like(locations)
  io1 = e * (1.0 - er)
  io2 = 1.0 - er * (1.0 - e)
  io3 = jnp.ones_like(locations)

  # Using this information, we can project the sample location to barycentric
  # coordinates for the simplex it's in.

  # This integer tells us which of the six simplices we are in.
  simplex_id = (jnp.dot(e, jnp.array([4, 2, 1])) - 1).astype(jnp.int32)
  simplex_id = jnp.maximum(0, simplex_id)

  # Here are the precomputed 3x3 matrices that project the 3D point into the
  # barycentric coordinates of each simplex.
  inverse_basis_tensor = jnp.array([
      # Case 0: z > y > x
      [[0.0, -1.0, 1.0], [-1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
      # Case 1: y > x > z
      [[-1.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
      # Case 2: y > z > x
      [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -0.0]],
      # Case 3: x > z > y
      [[1.0, 0.0, -1.0], [0.0, -1.0, 1.0], [0.0, 1.0, 0.0]],
      # Case 4: z > x > y
      [[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
      # Case 5: x > y > z
      [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
  ])
  inverse_basis = inverse_basis_tensor[simplex_id, Ellipsis]

  # Now compute the barycentric weights for the sample.
  bcd = jnp.matmul(inverse_basis, voxel_locations[Ellipsis, None])[Ellipsis, 0]
  a = 1.0 - jnp.sum(bcd, axis=-1, keepdims=True)
  weights = jnp.concatenate([a, bcd], axis=-1)

  output = None
  max_indices = jnp.array(data.shape[:3], dtype=jnp.int32) - 1
  if coordinate_order == 'xyz':
    max_indices = jnp.flip(max_indices)

  for io, weight in zip(
      [io0, io1, io2, io3],
      [weights[Ellipsis, 0], weights[Ellipsis, 1], weights[Ellipsis, 2], weights[Ellipsis, 3]],
  ):
    vertex_positions = floored + io
    vertex_positions = vertex_positions.astype(jnp.int32)
    vertex_positions = jnp.maximum(vertex_positions, 0)
    vertex_positions = jnp.minimum(vertex_positions, max_indices)

    weighted_gathered = (
        gather_volume(data, vertex_positions, coordinate_order)
        * weight[Ellipsis, None]
    )
    if output is None:
      output = weighted_gathered
    else:
      output += weighted_gathered

  return output


@gin.constants_from_enum
class ResampleOpMode(enum.Enum):
  """An enum defining the different implementations for resampling."""

  # The JAX implementation of the resampling ops.
  DEFAULT_JAX = enum.auto()

  # The JAX version of the simplex resampling ops.
  SIMPLEX_JAX = enum.auto()


_HASH_RESAMPLE_OP_MAP = {
    ResampleOpMode.DEFAULT_JAX: jax_hash_resample_3d,
    ResampleOpMode.SIMPLEX_JAX: jax_hash_simlex_resample_3d,
}

_RESAMPLE_OP_MAP = {
    ResampleOpMode.DEFAULT_JAX: jax_resample_3d,
    ResampleOpMode.SIMPLEX_JAX: jax_simplex_resample_3d,
}

_RESAMPLE_2D_OP_MAP = {
    ResampleOpMode.DEFAULT_JAX: jax_resample_2d,
}


def trilerp(
    values,
    coordinates,
    datastructure,
    op_mode,
):
  """Sample from a hash or 3D voxel grid `values` using `coordinates`.

  Args:
    values: A (D,H,W,C) array containing values if datastructure == 'grid' or a
      (N,C) array containing values if datastructure == 'hash'.
    coordinates: A (..., 3) array containing coordinates to sample. The values
      must be between 0 and the size of that dimension.
    datastructure: Which datastructure to use, either 'grid' or 'hash'.
    op_mode: Which resample op implementation to use, see `ResampleOpMode`.

  Returns:
    A (..., C) array containing the interpolated values at the given
      coordinates.

  Raises:
    ValueError: If an invalid datastructure is passed.
  """

  if datastructure == 'hash':
    fn = _HASH_RESAMPLE_OP_MAP[op_mode]
  elif datastructure == 'grid':
    # Note: unlike hash_resample_3d, resample_3d expects integer coordinate
    # voxel centers, so we offset the coordinates by 0.5 here. We also
    # flip the input coordinates since the convention used in `resample_3d`
    # is for input point (x, y, z) to index grid_values[z, y, x]. We prefer the
    # grid axis order to align with the Cartesian coordinate axes.
    coordinates = jnp.flip(coordinates - 0.5, axis=-1)

    def fn(v, c):
      """Add and remove two extra dims at the front of coord/output tensors."""
      return _RESAMPLE_OP_MAP[op_mode](v, c[None, None])[0, 0]

  else:
    raise ValueError(
        'datastructure must be either `grid` or `hash` but '
        f'`{datastructure}` was given.'
    )

  coordinates_flat = coordinates.reshape(-1, coordinates.shape[-1])
  result_flat = fn(values, coordinates_flat)
  result = result_flat.reshape(coordinates.shape[:-1] + (values.shape[-1],))
  return result


# Each of the L (`num_scales`) resolution levels in the 3D hash table stores
# “neural feature” vectors of length F (`num_features`).
# A given level is discretized into N^3 cells,
# where N (`grid_size`) ranges from
# Nmin=16 to Nmax ∈ [512..524288] (or more),
# which are then hashed into a table with T (`hash_map_size`) entries.
# This is summarized in Table 1 in the InstantNGP paper.


@gin.configurable
class HashEncoding(nn.Module):
  """Multiresolution grid/hash encoding from Instant NGP."""

  hash_map_size: int = 2**19  # parameter T in InstantNGP
  num_features: int = 2  # parameter F in InstantNGP
  scale_supersample: float = 2.0  # The "supersampling" factor between scales.
  # == 0.25 scales sizes by 16x, like (16, 256).
  # == 0.5 scales sizes by 4x, like (16, 64, 256).
  # == 1 scales sizes by 2x, like (16, 32, 64, 128, 256).
  # == 2 scales sizes by sqrt(2)x, like (16, 23, 32, 45, 64, ..., 256).
  # If you want a ratio of R between adjacent grid scales, set
  #   scale_supersample = 1 / log2(R)
  min_grid_size: int = 16  # parameter N_min in InstantNGP
  max_grid_size: int = 2048  # parameter N_max in InstantNGP
  hash_init_range: float = 1e-4
  precondition_scaling: float = 10.0  # Modification to NGP made by hedman@.
  # Defines the bounding box of the coordinates hash grid contains. If it is a
  # float, it will cover the bounding box ((-s, -s, -s), (s, s, s)). Otherwise,
  # it can be a tuple containing (min_coord, max_coord), e.g.:
  #   `((xmin, ymin, zmin), (xmax, ymax, zmax))`.
  # Defaults to 2 for the MipNeRF 360 "squash" space.
  bbox_scaling: Union[float, BboxType] = 2.0
  resample_op_mode: ResampleOpMode = ResampleOpMode.DEFAULT_JAX
  feature_aggregator: str = 'concatenate'  # How features are combined.
  append_scale: bool = True  # Append an explicit scale feature.
  # To retrieve the “neural” feature vector for a given 3D coordinate
  # x in the [0,1]^3 volume (which MipNeRF360 extends to an unbounded volume),
  # the voxels surrounding the coordinate are fetched from the hash table
  # and their corresponding feature vectors are then tri-linearly interpolated.
  # The feature vectors from each level are concatenated together,
  # and then returned for further processing by a following MLP.
  # This is summarized in Figure 3 of the paper InstantNGP paper.

  @property
  def grid_sizes(self):
    """Returns the grid sizes."""
    desired_num_scales = 1 + self.scale_supersample * onp.log2(
        self.max_grid_size / self.min_grid_size
    )
    num_scales = int(onp.round(desired_num_scales))
    if onp.abs(desired_num_scales - num_scales) > 1e-4:
      raise ValueError(
          'grid scale parameters are ('
          + f'min_grid_size={self.min_grid_size}, '
          + f'max_grid_size={self.max_grid_size}, '
          + f'scale_supersample={self.scale_supersample}), '
          + f'which yields a non-integer number of scales {desired_num_scales}.'
      )

    return onp.round(
        onp.geomspace(
            self.min_grid_size,
            self.max_grid_size,
            num_scales,
        )
    ).astype(onp.int32)

  def get_grid_size_str(self, grid_size):
    grid_size_str_len = len(str(onp.max(self.grid_sizes)))  # For zero paddding.
    return str(grid_size).zfill(grid_size_str_len)  # Zero pad.

  @property
  def bbox(self):
    bbox = self.bbox_scaling
    if isinstance(bbox, float):
      bbox = ((-bbox,) * 3, (bbox,) * 3)
    return onp.array(bbox)

  @nn.compact
  def __call__(
      self,
      x,
      *,
      x_scale=None,
      per_level_fn=None,
      train=True,
  ):
    # Map x to [0,1]^3
    x = (x - self.bbox[0]) / (self.bbox[1] - self.bbox[0])

    if x_scale is not None:
      bbox_sizes = onp.diff(self.bbox, axis=0)[0]
      if any(abs(bbox_sizes[0] - bbox_sizes[1:]) > onp.finfo(onp.float32).eps):
        raise ValueError('x_scale must be None when bbox is not square.')
      x_scale /= bbox_sizes[0]

    # Create a list of per-level features.
    grid_values = []
    grid_sizes = []
    grid_datastructures = []

    features = []
    for grid_size in self.grid_sizes:
      if grid_size**3 <= self.hash_map_size:
        # For smaller levels (fewer cells), store entries in a dense grid.
        datastructure = 'grid'
        shape_prefix = [grid_size] * 3
      else:
        datastructure = 'hash'
        shape_prefix = [self.hash_map_size]

      # Initialize/grab the tensor of grid or hash values.
      maxval = self.hash_init_range / self.precondition_scaling
      init_fn = functools.partial(
          random.uniform,
          shape=shape_prefix + [self.num_features],
          minval=-maxval,
          maxval=maxval,
      )
      grid_size_str = self.get_grid_size_str(grid_size)
      values = self.param(f'{datastructure}_{grid_size_str}', init_fn)
      grid_values.append(values)
      grid_sizes.append(grid_size)
      grid_datastructures.append(datastructure)

    for values, grid_size, datastructure in zip(
        grid_values, grid_sizes, grid_datastructures
    ):
      # Interpolate into `values` to get a per-coordinate feature vector.
      f = trilerp(values, x * grid_size, datastructure, self.resample_op_mode)

      if x_scale is not None:
        # Weight the feature by assuming that x_scale is the standard deviation
        # of an isotropic gaussian whose mean is x, and by computing the
        # fraction of the PDF of that Gaussian that is inside a [-1/2, 1/2]^3
        # cube centered at x.
        weighting = math.approx_erf(1 / (jnp.sqrt(8) * (x_scale * grid_size)))
        f *= weighting
        if self.append_scale:
          # Take the `weighting` used to rescale `f` and concatenate
          # `2 * weighting - 1`` as a feature. Training can get unstable if the
          # feature and the weight-feature have very different magnitudes, and
          # the features usually start small and grow large, so we rescale the
          # weight-feature with the current standard deviation of the features
          # (softly clipped to be >= the maximum initialized value to guard
          # against the case where `values`` shrinks to `0`) so that they're
          # matched. We have a stop-gradient so training doesn't
          # try to change `f_scale`` by messing with `f``).
          f_scale = (2 * weighting - 1) * jnp.sqrt(
              maxval**2 + jnp.mean(jax.lax.stop_gradient(values) ** 2)
          )
          f = jnp.concatenate([f, f_scale], axis=-1)

      if per_level_fn is not None:
        f = per_level_fn(f)

      features.append(f)

    # Aggregate into a single "neural feature" vector.
    if self.feature_aggregator == 'concatenate':
      features = jnp.concatenate(features, axis=-1)
    elif self.feature_aggregator == 'sum':
      features = jnp.sum(jnp.stack(features, axis=-1), axis=-1)
    else:
      raise ValueError(f'Aggregator {self.feature_aggregator} not implemented.')

    features *= self.precondition_scaling

    return features


@gin.configurable
class FactoredGrid(nn.Module):
  """Low-rank 3D voxel grids from TensoRF."""

  grid_size: int = 300  # This roughly matches default num of params for NGP.
  num_features: int = 28
  num_components: int = 64
  feature_init_scale: float = 0.1
  bbox_scaling: float = 2.0  # Defaults to 2 for the MipNeRF 360 "squash" space.
  reduction: str = 'sum'  # Defaults to 2 for the MipNeRF 360 "squash" space.

  @property
  def bbox(self):
    bbox = self.bbox_scaling
    if isinstance(bbox, float):
      bbox = ((-bbox,) * 3, (bbox,) * 3)
    return onp.array(bbox)

  @nn.compact
  def __call__(
      self,
      x,
      *,
      x_scale=None,
      per_level_fn=None,
      train=True,
  ):
    # NOTE: x_scale is currently unused.
    if x_scale is not None:
      raise ValueError('x_scale should be None for Triplane.')

    # For now, use simple permutations of the x/y/z axes as the transforms.
    frames = [onp.roll(onp.eye(3), i, axis=0) for i in range(3)]
    num_frames = len(frames)
    # frames is [num_frames, 3, 3].
    frames = onp.stack(frames, axis=0) / self.bbox_scaling
    # Project the point into coordinate system for each array.
    # x will be [..., num_frames, 3] after matmul.
    x = math.matmul(frames, x[Ellipsis, None, :, None])[Ellipsis, 0]

    # Shift and scale [-1, 1]^3 to [0, N]^3.
    x = (x + 1.0) / 2.0 * self.grid_size
    # Convert x to [num_frames, 3, ...].
    x = jnp.moveaxis(x, (-2, -1), (0, 1))
    # Grab respective 1d and 2d coordinate vectors for interpolation.
    coords_1d = x[:, :1]
    coords_2d = x[:, 1:3]

    # Create feature grids.
    def feat_init(key, shape):
      return random.normal(key, shape) * self.feature_init_scale

    # We leave the first two dimensions separate here since the per-frame
    # transformed coordinates are shared by components, allowing a nice vmap.
    shape_prefix = (self.num_components, num_frames)
    shape_1d = shape_prefix + (self.grid_size,) * 1
    features_1d = self.param('grid_features_1d', feat_init, shape_1d)
    shape_2d = shape_prefix + (self.grid_size,) * 2
    features_2d = self.param('grid_features_2d', feat_init, shape_2d)

    shape_appearance = (self.num_components * num_frames, self.num_features)
    features_appearance = self.param(
        'grid_features_appearance', feat_init, shape_appearance
    )

    # We want to vmap over both inputs for the `num_frames` dimension but only
    # the feature grids for the `num_components` dimension.
    lerp_fn = functools.partial(jax.scipy.ndimage.map_coordinates, order=1)
    interp_fn = jax.vmap(jax.vmap(lerp_fn), (0, None))
    # Shape [num_components, num_frames, ...].
    gathered_1d = interp_fn(features_1d, coords_1d)
    gathered_2d = interp_fn(features_2d, coords_2d)
    # Reconstruct 3D features as product of low rank decomposition.
    gathered_3d = gathered_1d * gathered_2d
    # Shift/flatten the first axis to do matmul with `features_appearance`.
    gathered_3d = gathered_3d.reshape((-1,) + gathered_3d.shape[2:])
    gathered_3d = jnp.moveaxis(gathered_3d, 0, -1)
    # Input shapes are now [..., num_components * num_frames] and
    # [num_components * num_frames, num_features].
    features = math.matmul(gathered_3d, features_appearance)
    if per_level_fn is not None:
      features = per_level_fn(features)
    return features


@gin.configurable
class Triplane(nn.Module):
  """Triplane representation from EG3D."""

  grid_size: int = 512
  num_features: int = 48
  feature_init_scale: float = 0.1
  bbox_scaling: float = 2.0  # Defaults to 2 for the MipNeRF 360 "squash" space.
  resample_op_mode: ResampleOpMode = ResampleOpMode.DEFAULT_JAX

  @property
  def bbox(self):
    bbox = self.bbox_scaling
    if isinstance(bbox, float):
      bbox = ((-bbox,) * 3, (bbox,) * 3)
    return onp.array(bbox)

  @nn.compact
  def __call__(
      self,
      x,
      *,
      x_scale=None,
      per_level_fn=None,
      train=True,
  ):
    # NOTE: x_scale is currently unused.
    if x_scale is not None:
      raise ValueError('x_scale should be None for Triplane.')

    # Permute the xyz axes to get the matrix for each plane.
    frames = [onp.roll(onp.eye(3), i, axis=0) for i in range(3)]
    num_frames = len(frames)
    # frames is [num_frames, 3, 3].
    frames = onp.stack(frames, axis=0) / self.bbox_scaling
    # We only need 2 basis vectors for each plane.
    frames = frames[Ellipsis, 1:3, :]
    # Project the point into coordinate system for each array.
    # x will be [..., num_frames, 2] after matmul.
    x = math.matmul(frames, x[Ellipsis, None, :, None])[Ellipsis, 0]

    # Shift and scale [-1, 1]^2 to [0, N]^2.
    x = (x + 1.0) / 2.0 * self.grid_size
    # Convert x to [num_frames, ..., 2].
    coords_2d = jnp.moveaxis(x, -2, 0)

    # Create feature grids.
    def feat_init(key, shape):
      return random.normal(key, shape) * self.feature_init_scale

    shape_2d = (num_frames,) + (self.grid_size,) * 2 + (self.num_features,)
    features_2d = self.param('triplane_grid_features_2d', feat_init, shape_2d)

    interp_fn = _RESAMPLE_2D_OP_MAP[self.resample_op_mode]
    # Input arg shapes will be:
    #  [num_frames, grid_size, grid_size, num_features], [num_frames, ..., 2].
    gathered_2d = interp_fn(features_2d, coords_2d)
    # Sum over the features from each plane.
    if self.reduction == 'sum':
      features = jnp.sum(gathered_2d, axis=0)
    else:
      features = jnp.mean(gathered_2d, axis=0)

    if per_level_fn is not None:
      features = per_level_fn(features)
    return features


GRID_REPRESENTATION_BY_NAME = {
    'ngp': HashEncoding,
    'hash': HashEncoding,
    'triplane': Triplane,
    'tensorf': FactoredGrid,
}

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

"""Instant NGP multi-resolution hash encoding."""

# This Python/Jax program is a re-implementation of the multiresolution
# hash encoding structure described in Section 3 of the
# Instant Neural Graphics Primitives SIGGRAPH 2022 paper by
# Müller, Evans, Schied, and Keller.
#  see https://github.com/NVlabs/instant-ngp

import functools

from flax import linen as nn
import gin
from internal import grid_utils
from internal import math
from jax import random
import jax.numpy as jnp
import numpy as np


def trilerp(
    values,
    coordinates,
    datastructure,
):
  """Sample from a hash or 3D voxel grid `values` using `coordinates`."""

  if datastructure == 'hash':
    fn = hash_resample_3d
  elif datastructure == 'grid':
    # Note: unlike hash_resample_3d, resample_3d expects integer coordinate
    # voxel centers, so we offset the coordinates by 0.5 here. We also
    # flip the input coordinates since the convention used in `resample_3d`
    # is for input point (x, y, z) to index grid_values[z, y, x]. We prefer the
    # grid axis order to align with the Cartesian coordinate axes.
    coordinates = jnp.flip(coordinates - 0.5, axis=-1)

    def fn(v, c):
      """Add and remove two extra dims at the front of coord/output tensors."""
      return resample_3d(v, c[None, None])[0, 0]

  else:
    return ValueError(f'datastructure {datastructure} not implemented.')

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


class HashEncoding(nn.Module):
  """Mulitresolution hash encoding from Instant NGP."""

  hash_map_size: int = 2**21  # parameter T in InstantNGP.
  num_features: int = 2  # parameter F in InstantNGP.
  num_scales: int = 20  # parameter L in InstantNGP.
  min_grid_size: int = 16  # parameter N_min in InstantNGP.
  max_grid_size: int = 8192  # parameter N_max in InstantNGP.
  hash_init_range: float = 1e-4
  precondition_scaling: float = 10.0  # Modification to NGP made by hedman@.

  # To retrieve the “neural” feature vector for a given 3D coordinate
  # x in the [0,1]^3 volume (which MipNeRF360 extends to an unbounded volume),
  # the voxels surrounding the coordinate are fetched from the hash table
  # and their corresponding feature vectors are then tri-linearly interpolated.
  # The feature vectors from each level are concatenated together,
  # and then returned for further processing by a following MLP.
  # This is summarized in Figure 3 of the paper InstantNGP paper.

  def setup(self):
    self.grid_sizes = np.round(
        np.geomspace(
            self.min_grid_size,
            self.max_grid_size,
            self.num_scales,
        )
    ).astype(np.int32)

  @nn.compact
  def __call__(self, x):
    # Maps x ∈ [WORLD_MIN,WORLD_MAX]^3 to [0,1]^3.
    x = math.normalize(x, grid_utils.WORLD_MIN, grid_utils.WORLD_MAX)

    # Create a list of per-level features.
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
      values = self.param(f'{datastructure}_{grid_size}', init_fn)

      # Interpolate into `values` to get a per-coordinate feature vector.
      feature_level = trilerp(values, x * grid_size, datastructure)
      features.append(feature_level)

    # Aggregate into a single "neural feature" vector.
    features = jnp.concatenate(features, axis=-1)
    features *= self.precondition_scaling
    return features


# Initial proposal MLP with hash encoding used for generating densities
# with a different set of parameters than the main HashEncoding.
@gin.configurable
class PropHashEncoding(HashEncoding):
  hash_map_size: int = 2**16
  num_scales: int = 10
  max_grid_size: int = 512


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
        np.array([[1, 1], [1, 1], [1, 1]] + (data.ndim - 3) * [[0, 0]]),
        constant_values=constant_values,
    )
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
  max_indices = jnp.array(data.shape[:3], dtype=jnp.int32) - 1
  if coordinate_order == 'xyz':
    max_indices = jnp.flip(max_indices)

  output = jnp.zeros((*locations.shape[:-1], data.shape[-1]), dtype=data.dtype)

  for position, weight in zip(positions, weights):
    indexes = position.astype(jnp.int32)

    indexes = jnp.maximum(indexes, 0)
    indexes = jnp.minimum(indexes, max_indices)
    weighted_gathered = (
        gather_volume(data, indexes, coordinate_order)
        * weight[Ellipsis, None]
    )
    output += weighted_gathered

  return output.astype(data.dtype)


def hash_resample_3d(data, locations):
  """Resamples input data at the provided locations from a hash table.

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

    data_indexes = jnp.mod(
        jnp.bitwise_xor(
            position[Ellipsis, 0],
            jnp.bitwise_xor(position[Ellipsis, 1] * pi_2, position[Ellipsis, 2] * pi_3),
        ),
        data.shape[0],
    ).astype(jnp.int32)
    weighted_gathered = data[(data_indexes,)] * weight[Ellipsis, None]
    if output is None:
      output = weighted_gathered
    else:
      output += weighted_gathered

  return output

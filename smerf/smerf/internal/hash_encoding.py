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
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from smerf.internal import grid_utils
from smerf.internal import hash_resample
from smerf.internal import math
from smerf.internal import quantize
from smerf.internal import resample


_HASH_RESAMPLE_OP_MAP = {
    'jax': hash_resample.multi_hash_resample_3d_jax,
}

_RESAMPLE_OP_MAP = {
    'jax': resample.multi_resample_3d_jax,
}


@jax.named_scope('multi_trilerp')
def multi_trilerp(
    values,
    idxs,
    coordinates,
    datastructure,
    op_mode,
):
  """Samples from a hash or 3D voxel grid `values` using `coordinates`.

  Args:
    values: f32[n, k, c] or f32[n, k, k, k, c]. Hash maps or 3D grids with
      feature vectors. One per submodel.
    idxs: i32[..., 1]. Submodel indices. Determines which hash map or 3D grid
      to reference.
    coordinates: f32[..., 3]. 3D locations for lookups.
    datastructure: hash or grid. Determines which lookup method is used.
    op_mode: jax or cuda. Determines which implementation is used. CUDA is
      faster and both should produce identical outputs.

  Returns:
    f32[..., c]. Trilinearly interpolated feature vectors from 'values'.
  """

  if datastructure == 'hash':
    fn = _HASH_RESAMPLE_OP_MAP[op_mode]
  elif datastructure == 'grid':
    fn = _RESAMPLE_OP_MAP[op_mode]
  else:
    return ValueError(f'datastructure {datastructure} not implemented.')

  idxs_flat = idxs.reshape(-1, idxs.shape[-1])
  coordinates_flat = coordinates.reshape(-1, coordinates.shape[-1])
  result_flat = fn(values, idxs_flat, coordinates_flat)
  result = result_flat.reshape(coordinates.shape[:-1] + (values.shape[-1],))
  return result


# Each of the L (`num_scales`) resolution levels in the 3D hash table stores
# “neural feature” vectors of length F (`num_features`).
# A given level is discretized into N^3 cells,
# where N (`grid_size`) ranges from
# Nmin=16 to Nmax ∈ [512..524288] (or more),
# which are then hashed into a table with T (`hash_map_size`) entries.
# This is summarized in Table 1 in the InstantNGP paper.
@gin.configurable(denylist=['num_kernels'])
class MultiHashEncoding(nn.Module):
  """Mulitresolution hash encoding from Instant NGP."""

  hash_map_size: int = 2**21  # parameter T in InstantNGP.
  num_kernels: int = 1  # number of distinct hash encodings.
  num_features: int = 2  # parameter F in InstantNGP.
  num_scales: int = 20  # parameter L in InstantNGP.
  min_grid_size: int = 16  # parameter N_min in InstantNGP.
  max_grid_size: int = 8192  # parameter N_max in InstantNGP.
  hash_init_range: float = 1e-4
  precondition_scaling: float = 10.0  # Modification to NGP made by hedman@.
  op_mode: str = 'jax'  # jax, cuda

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
  def __call__(self, idxs, xs):
    # Maps x ∈ [WORLD_MIN,WORLD_MAX]^3 to [0,1]^3.
    xs = math.normalize(xs, grid_utils.WORLD_MIN, grid_utils.WORLD_MAX)

    # Create a list of per-level features.
    features = []
    for grid_size in self.grid_sizes:
      if grid_size**3 <= self.hash_map_size:
        # For smaller levels (fewer cells), store entries in a dense grid.
        datastructure = 'grid'
        shape_prefix = [self.num_kernels] + [grid_size] * 3
      else:
        datastructure = 'hash'
        shape_prefix = [self.num_kernels, self.hash_map_size]

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
      feature_level = multi_trilerp(
          values, idxs, xs * grid_size, datastructure, self.op_mode
      )
      features.append(feature_level)

    # Aggregate into a single "neural feature" vector.
    features = jnp.concatenate(features, axis=-1)
    features *= self.precondition_scaling
    return features


# Initial proposal MLP with hash encoding used for generating densities
# with a different set of parameters than the main HashEncoding.
@gin.configurable(denylist=['num_kernels'])
class MultiPropHashEncoding(MultiHashEncoding):
  hash_map_size: int = 2**16
  num_scales: int = 10
  max_grid_size: int = 512


class Multi3DGrid(nn.Module):
  """3D grid features with trilinear interpolation."""

  num_kernels: int = 1  # number of distinct 3D grids.
  num_features: int = 16  # number of features per voxel.
  grid_size: int = 64  # the side length, i.e. the grid is [grid_size]^3
  init_range: float = 1e-1
  op_mode: str = 'cuda'  # jax, cuda
  quantize: bool = False

  # To retrieve the “neural” feature vector for a given 3D coordinate
  # x in the [0,1]^3 volume (which MipNeRF360 extends to an unbounded volume),
  # the voxels surrounding the coordinate are fetched from the 3S grid
  # and their corresponding feature vectors are then tri-linearly interpolated.

  @nn.compact
  def __call__(self, idxs, xs):
    datastructure = 'grid'
    shape_prefix = [self.num_kernels] + [self.grid_size] * 3

    # Initialize/grab the tensor of grid or hash values.
    maxval = self.init_range
    init_fn = functools.partial(
        random.uniform,
        shape=shape_prefix + [self.num_features],
        minval=-maxval,
        maxval=maxval,
    )
    values = self.param(f'{datastructure}_{self.grid_size}', init_fn)
    values = jax.nn.sigmoid(values)

    # Interpolate into `values` to get a per-coordinate feature vector.
    quant_values = (
        quantize.differentiable_byte_quantize(values) if quantize else values
    )
    features = multi_trilerp(
        quant_values, idxs, xs * self.grid_size, datastructure, self.op_mode
    )

    return features


@jax.named_scope('multi_gather_volume')
def multi_gather_volume(data, data_idxs, locations, coordinate_order='xyz'):
  """Gather from data at locations.

  Args:
    data: A f32[K, D, H, W, C] tensor.
    data_idxs: A i32[..., 1] tensor of indices for 'data'. Values in [0, K).
    locations: A i32[..., 3] tensor containing the locations to sample at.
    coordinate_order: Whether the sample locations are x,y,z or z,y,x.

  Returns:
    A [..., C] tensor containing the gathered locations.
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
    raise ValueError(f'coordinate_order {coordinate_order} not implemented.')

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
  return data[data_idxs[Ellipsis, 0], z_coordinate, y_coordinate, x_coordinate]

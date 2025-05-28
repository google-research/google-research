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

"""Voxel grid interpolation and Instant NGP hash encoding utility functions."""

# This Python/Jax program is a re-implementation of the multiresolution
# hash encoding structure described in Section 3 of the
# Instant Neural Graphics Primitives SIGGRAPH 2022 paper by
# Müller, Evans, Schied, and Keller.
#  see https://github.com/NVlabs/instant-ngp

import functools
from typing import Union

from flax import linen as nn
import gin
from internal import hash_resample
from internal import math
from internal import resample
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp


# A bounding box defined as a tuple containing (min_coord, max_coord).
BboxType = tuple[tuple[float, float, float], tuple[float, float, float]]



def trilerp(
    values,
    coordinates,
    datastructure,
):
  """Sample from a hash or 3D voxel grid `values` using `coordinates`.

  TODO(keunhong): Consider making datastructure an enum as well.

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
    fn = hash_resample.hash_resample_3d
  elif datastructure == 'grid':
    # Note: unlike hash_resample_3d, resample_3d expects integer coordinate
    # voxel centers, so we offset the coordinates by 0.5 here. We also
    # flip the input coordinates since the convention used in `resample_3d`
    # is for input point (x, y, z) to index grid_values[z, y, x]. We prefer the
    # grid axis order to align with the Cartesian coordinate axes.
    coordinates = jnp.flip(coordinates - 0.5, axis=-1)

    def fn(v, c):
      """Add and remove two extra dims at the front of coord/output tensors."""
      return resample.resample_3d(v, c[None, None])[0, 0]

  else:
    raise ValueError(
        'datastructure must be either `grid` or `hash` but '
        f'`{datastructure}` was given.'
    )

  coordinates_flat = coordinates.reshape(-1, coordinates.shape[-1])
  if values.dtype != coordinates_flat.dtype:
    coordinates_flat = coordinates_flat.astype(values.dtype)
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
class HashEncoding(nn.Module):  # TODO(barron): Rename this to just "NGP".
  """Multiresolution grid/hash encoding from Instant NGP."""

  hash_map_size: int = 2**19  # parameter T in InstantNGP
  num_features: int = 2  # parameter F in InstantNGP
  scale_supersample: float = 2.0  # The "supersampling" factor between scales.
  # == 0.25 scales sizes by 16x, like (16, 256).
  # == 0.5 scales sizes by 4x, like (16, 64, 256).
  # == 1 scales sizes by 2x, like (16, 32, 64, 128, 256).
  # == 2 scales sizes by sqrt(2)x, like (16, 23, 32, 45, 64, ..., 256).
  # If you want a ratio of R between adjacent grid scales, set
  #   scale_supersample = 1 / log2(R).
  # TODO(barron): Parameterize this as with R directly.
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
  append_scale: bool = True  # Append an explicit scale feature.
  use_triton_ops: bool = False  # Use Triton ops for hash resampling.
  jitter_coordinates: bool = False  # Randomly jitter coords by [-0.5, 0.5).
  # To retrieve the “neural” feature vector for a given 3D coordinate
  # x in the [0,1]^3 volume (which MipNeRF360 extends to an unbounded volume),
  # the voxels surrounding the coordinate are fetched from the hash table
  # and their corresponding feature vectors are then tri-linearly interpolated.
  # The feature vectors from each level are concatenated together,
  # and then returned for further processing by a following MLP.
  # This is summarized in Figure 3 of the paper InstantNGP paper.
  use_float16_hash: bool = False  # Whether to use float16 for the hashes.

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
      rng=None,
      min_grid_size=None,
      max_grid_size=None,
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
      if (min_grid_size is not None and grid_size < min_grid_size) or (
          max_grid_size is not None and grid_size > max_grid_size
      ):
        continue
      if grid_size**3 <= self.hash_map_size:
        # For smaller levels (fewer cells), store entries in a dense grid.
        datastructure = 'grid'
        shape_prefix = [grid_size] * 3
      else:
        datastructure = 'hash'
        shape_prefix = [self.hash_map_size]

      # Initialize/grab the tensor of grid or hash values.
      maxval = self.hash_init_range / self.precondition_scaling
      dtype_to_use = jnp.float32
      if self.use_float16_hash and datastructure == 'hash':
        dtype_to_use = jnp.float16
      init_fn = functools.partial(
          random.uniform,
          shape=shape_prefix + [self.num_features],
          minval=-maxval,
          maxval=maxval,
          dtype=dtype_to_use,
      )
      grid_size_str = self.get_grid_size_str(grid_size)
      values = self.param(f'{datastructure}_{grid_size_str}', init_fn)
      grid_values.append(values)
      grid_sizes.append(grid_size)
      grid_datastructures.append(datastructure)

    if self.use_triton_ops:
      if x_scale is not None:
        raise ValueError(
            f'Triton Ops do not support x_scale. Got x_scale: {x_scale}'
        )
      if per_level_fn != math.average_across_multisamples:
        raise ValueError(
            'Trion Ops require per_level_fn to be'
            f' math.average_across_multisamples. Got: {per_level_fn}'
        )
      total_grid = jnp.concatenate(
          [grid.reshape(-1, grid.shape[-1]) for grid in grid_values], axis=0
      )
      side_or_hash_table_sizes = [
          grid_size if datastructure == 'grid' else self.hash_map_size
          for grid_size, datastructure in zip(grid_sizes, grid_datastructures)
      ]
      voxel_grids_count = sum(int(x == 'grid') for x in grid_datastructures)
      warp_scales = jnp.array(grid_sizes, dtype=x.dtype)

      batch_size, rays_samples, spiral_samples, _ = x.shape

      # Triton kernel doesn't have the backward pass, so fallback to CUDA during
      # training.
      implementation = (
          triton_grids_interface.VoxelAndHashResampleImplementation.CUDA
          if train
          else triton_grids_interface.VoxelAndHashResampleImplementation.TRITON
      )

      result = triton_grids_interface.voxel_and_hash_resample(
          total_grid,
          side_or_hash_table_sizes,
          voxel_grids_count,
          warp_scales,
          x.reshape(batch_size * rays_samples, spiral_samples, 3),
          features_scale=self.precondition_scaling,
          implementation=implementation,
      )
      return result.reshape(
          batch_size, rays_samples, self.num_features * len(self.grid_sizes)
      )

    for values, grid_size, datastructure in zip(
        grid_values, grid_sizes, grid_datastructures
    ):
      # Scale `x` by the grid size to get the indices of the coordinates.
      x_scaled = x * grid_size

      # Optionally jitter the scaled coordinates by [-0.5, 0.5).
      if self.jitter_coordinates:
        if rng is not None:
          key, rng = random.split(rng)
          x_scaled += random.uniform(key, x_scaled.shape) - 0.5

      # Interpolate into `values` to get a per-coordinate feature vector.
      f = trilerp(values, x_scaled, datastructure)

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
    features = jnp.concatenate(features, axis=-1)

    features *= self.precondition_scaling

    return features

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

"""General math functions for NeRF."""

import jax
import jax.numpy as jnp

from nf_diffusion.models.utils import resample


# Grid/hash gathers with trilinear interpolation.
def up_sample_grid_trilerp(grid_values, coordinates):
  """Grid trilinear interpolation."""
  # Note: unlike hash_resample_3d, resample_3d expects integer coordinate voxel
  # centers, so we need to offset the coordinates by 0.5 here.
  coordinates_flat = coordinates.reshape((-1, 3)) - 0.5
  coordinates_flat = jnp.flip(coordinates_flat, axis=1)
  coordinates_3d = coordinates_flat.reshape([1, 1, -1, 3])
  result = resample.resample_3d(grid_values, coordinates_3d)
  num_channels = result.shape[-1]
  result = result.reshape(list(coordinates.shape[0:-1]) + [num_channels])
  return result


def upsample_vox_v0(vox, up_factor = 2):
  """Upsample voxel by [up_factor]."""
  bss = vox.shape[:-4]
  orig_res = vox.shape[-2]
  out_res = orig_res * up_factor
  dims = vox.shape[-1]
  pts_grid = (
      jnp.concatenate(
          [
              x[Ellipsis, None]
              for x in jnp.meshgrid(
                  jnp.arange(out_res), jnp.arange(out_res), jnp.arange(out_res)
              )
          ],
          axis=-1,
      )
      / float(out_res - 1)
      * orig_res
  )[Ellipsis, [1, 0, 2]]
  pts_grid = pts_grid.reshape(-1, 3)
  out_vox = up_sample_grid_trilerp(vox, pts_grid)
  out_vox = out_vox.reshape(*bss, out_res, out_res, out_res, dims)
  return out_vox


def upsample_vox(vox, up_factor = 2):
  """Upsample voxel by [up_factor]."""
  bss = vox.shape[:-4]
  orig_res = vox.shape[-2]
  out_res = orig_res * up_factor
  dims = vox.shape[-1]
  out_vox = jax.image.resize(
      vox,
      shape=(*bss, out_res, out_res, out_res, dims),
      method=jax.image.ResizeMethod.LINEAR,
  )
  out_vox = out_vox.reshape(*bss, out_res, out_res, out_res, dims)
  return out_vox

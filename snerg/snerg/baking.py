# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Functions to extract a grid/atlas of colors and features from a SNeRG MLP."""

import gc
import jax
import numpy as np

from snerg.snerg import model_utils


def build_3d_grid(min_xyz,
                  voxel_size,
                  grid_size,
                  worldspace_t_opengl=np.eye(4),
                  output_dtype=np.float32):
  """Builds a tensor containing a regular grid of 3D locations.

  Args:
    min_xyz: The minimum XYZ location of the grid.
    voxel_size: The side length of a voxel.
    grid_size: A numpy array containing the grid dimensions [H, W, D].
    worldspace_t_opengl: An optional 4x4 transformation matrix that maps the
      native coordinate space of the NeRF model to an OpenGL coordinate system,
      where y is down, and negative-z is pointing towards the scene.
    output_dtype: The data type of the resulting grid tensor.

  Returns:
    A [H, W, D, 3] numpy array for the XYZ coordinates of each grid cell center.
  """
  x_span = min_xyz[0] + voxel_size / 2 + np.arange(
      grid_size[0], dtype=output_dtype) * voxel_size
  y_span = min_xyz[1] + voxel_size / 2 + np.arange(
      grid_size[1], dtype=output_dtype) * voxel_size
  z_span = min_xyz[2] + voxel_size / 2 + np.arange(
      grid_size[2], dtype=output_dtype) * voxel_size
  xv, yv, zv = np.meshgrid(x_span, y_span, z_span, indexing='ij')
  positions_hom = np.stack([xv, yv, zv, np.ones_like(zv)], axis=-1)
  positions_hom = positions_hom.reshape((-1, 4)).dot(worldspace_t_opengl)
  return positions_hom[Ellipsis, 0:3].reshape(
      (xv.shape[0], xv.shape[1], xv.shape[2], -1))


def render_voxel_block(mlp_model, mlp_params, block_coordinates_world,
                       voxel_size, scene_params):
  """Extracts a grid of colors, features and alpha values from a SNeRG model.

  Args:
    mlp_model: A nerf.model_utils.MLP that predicts per-sample color, density,
      and the SNeRG feature vector.
    mlp_params: A dict containing the MLP parameters for the per-sample MLP.
    block_coordinates_world: A [H, W, D, 3] numpy array of XYZ coordinates (see
      build_3d_grid).
    voxel_size: The side length of a voxel.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    output_rgb_and_features: A [H, W, D, C] numpy array for the colors and
      computed features at each voxel.
    output_alpha: A [H, W, D, 1] numpy array for the alpha values at each voxel.
  """
  num_devices = jax.device_count()
  host_id = jax.host_id()

  chunk_size = scene_params['chunk_size']
  channels = scene_params['_channels']
  output_dtype = np.dtype(scene_params['dtype'])

  batch_size = chunk_size * num_devices
  actual_num_rays = int(np.array(block_coordinates_world.shape[0:3]).prod())
  rounded_num_rays = batch_size * (
      (actual_num_rays + batch_size - 1) // batch_size)

  origins = block_coordinates_world.reshape((-1, 3)).copy()
  origins.resize((rounded_num_rays, 3))
  origins = origins.reshape((-1, num_devices, batch_size // num_devices, 3))

  rgb_and_features = np.zeros(
      (origins.shape[0], origins.shape[1], origins.shape[2], channels),
      dtype=output_dtype)
  sigma = np.zeros_like(rgb_and_features[:, :, :, 0])
  for i in range(origins.shape[0]):
    batch_origins = origins[i]
    batch_origins = batch_origins.reshape(num_devices, 1, -1, 3)

    host_batch_origins = batch_origins[host_id *
                                       jax.local_device_count():(host_id + 1) *
                                       jax.local_device_count()]

    batch_rgb, batch_sigma = model_utils.pmap_model_fn(mlp_model, mlp_params,
                                                       host_batch_origins,
                                                       scene_params)
    rgb_and_features[i] = np.array(
        batch_rgb[0], dtype=output_dtype).reshape(rgb_and_features[i].shape)
    sigma[i] = np.array(
        batch_sigma[0], dtype=output_dtype).reshape(sigma[i].shape)

  rgb_and_features = rgb_and_features.reshape((-1, channels))
  sigma = sigma.reshape((-1))
  rgb_and_features = rgb_and_features[:actual_num_rays]
  sigma = sigma[:actual_num_rays]

  alpha = 1.0 - np.exp(-sigma * voxel_size)
  output_rgb_and_features = rgb_and_features.reshape(
      (block_coordinates_world.shape[0], block_coordinates_world.shape[1],
       block_coordinates_world.shape[2], channels))
  output_alpha = alpha.reshape(block_coordinates_world.shape[0:3])

  return output_rgb_and_features * np.expand_dims(output_alpha,
                                                  -1), output_alpha


def extract_3d_atlas(mlp_model, mlp_params, scene_params, render_params,
                     atlas_params, culling_params, alpha_grid_values,
                     visibility_grid_values):
  """The main baking function, this extracts the 3D texture atlas.

  Uses a precomputed culling grid, to avoid querying empty space, and packs
  the extracted colors, featurs, and alpa values into a 3D texture atlas of
  size (S, S, N, C), where
    S = atlas_params['atlas_slice_size'],
    N is computed on the fly based on the amount of free space in the scene, and
    C defaults to 7 (RGB + 4 features).

  Args:
    mlp_model: A nerf.model_utils.MLP that predicts per-sample color, density,
      and the SNeRG feature vector.
    mlp_params: A dict containing the MLP parameters for the per-sample MLP.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    render_params: A dict with parameters for high-res rendering.
    atlas_params: A dict with params for building the 3D texture atlas.
    culling_params: A dict for low-res rendering and opacity/visibility culling.
    alpha_grid_values: Used to avoid querying empty space --- a [bW, bH, bD, 1]
      numpy array for containing the maximum  alpha value found inside each
      macroblock.
    visibility_grid_values: Used to avoid querying voxels that are never visible
      during trainingspace --- a [bW, bH, bD, 1] numpy array for containing the
      maximum  visibility value found inside each macroblock.

  Returns:
    atlas: The SNeRG scene packed as a texture atlas in a [S, S, N, C] numpy
      array, where the channels C contain both RGB and features.
    atlas_block_indices: The indirection grid for the SNeRG scene, this is a
      numpy int32 array of size (bW, bH, bD, 3), This is a low resolution dense
      grid containing either -1 for free space, or the 3D index of the
      macroblock in the texture atlas that represent this region of space.
  """
  atlas_block_size = atlas_params['atlas_block_size']
  min_xyz = scene_params['min_xyz']
  voxel_size = render_params['_voxel_size']

  data_block_size = atlas_params['_data_block_size']
  atlas_grid_size = atlas_params['_atlas_grid_size']
  atlas_slice_size = atlas_params['atlas_slice_size']
  num_channels = scene_params['_channels']
  atlas_dtype = np.dtype(scene_params['dtype'])

  atlas_blocks = np.logical_and(
      visibility_grid_values >= culling_params['visibility_threshold'],
      alpha_grid_values >= culling_params['alpha_threshold']).sum()
  atlas_slices = (atlas_blocks * atlas_block_size**3) / (atlas_slice_size**2)
  atlas_slices = atlas_block_size * int(
      np.ceil(atlas_slices / atlas_block_size))

  atlas = np.zeros(
      (atlas_slice_size, atlas_slice_size, atlas_slices, num_channels + 1),
      dtype=atlas_dtype)
  atlas_blocks_x = int(atlas.shape[0] / atlas_block_size)
  atlas_blocks_y = int(atlas.shape[1] / atlas_block_size)

  atlas_valid_mask = np.zeros(
      (atlas_slice_size, atlas_slice_size, atlas_slices), dtype=atlas_dtype)
  atlas_world_coordinates = np.zeros(
      (atlas_slice_size, atlas_slice_size, atlas_slices, 3), dtype=atlas_dtype)
  atlas_block_indices = -1 * np.ones(
      (atlas_grid_size[0], atlas_grid_size[1], atlas_grid_size[2], 3), np.int16)

  atlas_index = 0
  for block_x in range(atlas_grid_size[0]):
    for block_y in range(atlas_grid_size[1]):
      for block_z in range(atlas_grid_size[2]):

        if alpha_grid_values[block_x, block_y,
                             block_z] < culling_params['alpha_threshold']:
          continue

        if visibility_grid_values[
            block_x, block_y, block_z] < culling_params['visibility_threshold']:
          continue

        atlas_min_voxel = np.array([block_x, block_y, block_z
                                   ]) * data_block_size - 1
        atlas_max_voxel = atlas_min_voxel + atlas_block_size
        atlas_block_size_3d = atlas_max_voxel - atlas_min_voxel

        atlas_i = atlas_index
        atlas_ix = int(atlas_i % atlas_blocks_x)
        atlas_i = int(atlas_i / atlas_blocks_x)
        atlas_iy = int(atlas_i % atlas_blocks_y)
        atlas_iz = int(atlas_i / atlas_blocks_y)

        atlas_x = int(atlas_ix * atlas_block_size)
        atlas_y = int(atlas_iy * atlas_block_size)
        atlas_z = int(atlas_iz * atlas_block_size)

        atlas_block_indices[block_x, block_y,
                            block_z] = np.array([atlas_ix, atlas_iy, atlas_iz])
        atlas_world_coordinates[atlas_x:(atlas_x + atlas_block_size),
                                atlas_y:(atlas_y + atlas_block_size),
                                atlas_z:(atlas_z + atlas_block_size),
                                0:3] = build_3d_grid(
                                    min_xyz + atlas_min_voxel * voxel_size,
                                    voxel_size, atlas_block_size_3d,
                                    scene_params['worldspace_T_opengl'])
        atlas_valid_mask[atlas_x:(atlas_x + atlas_block_size),
                         atlas_y:(atlas_y + atlas_block_size),
                         atlas_z:(atlas_z + atlas_block_size)] = 1.0

        atlas_index += 1
  num_samples = render_params['num_samples_per_voxel']
  for _ in range(num_samples):
    std_dev = render_params['voxel_filter_sigma']
    offset = np.random.normal(0, std_dev, (3)) * voxel_size
    # Use this inner loop to conserve CPU RAM.
    for z in range(atlas.shape[2]):
      atlas_rgb, atlas_alpha = render_voxel_block(
          mlp_model, mlp_params,
          atlas_world_coordinates[:, :, z:z + 1, :] + offset, voxel_size,
          scene_params)
      atlas[:, :, z:z + 1, 0:num_channels] += atlas_rgb
      atlas[:, :, z:z + 1, num_channels] += atlas_alpha
    gc.collect()  # Try this to mitigate out-of-memory crashes.

  atlas /= num_samples
  atlas[atlas_valid_mask < 1.0] = 0.0

  return atlas, atlas_block_indices

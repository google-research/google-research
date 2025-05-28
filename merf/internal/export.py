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

"""Functions that export a baked MERF model for viewing in the webviewer."""

import math
import multiprocessing
import os
from os import path

from internal import utils
import jax
import numpy as np
from PIL import Image


def save_8bit_png(img_and_path):
  """Save an 8bit numpy array as a PNG on disk.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit, [height, width,
      channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path
  with utils.open_file(pth, 'wb') as imgout:
    Image.fromarray(img).save(imgout, 'PNG')


def synchronize_jax_hosts():
  """Makes sure that the JAX hosts have all reached this point."""
  # Build an array containing the host_id.
  num_local_devices = jax.local_device_count()
  num_hosts = jax.process_count()
  host_id = jax.process_index()
  dummy_array = np.ones((num_local_devices, 1), dtype=np.int32) * host_id

  # Then broadcast it between all JAX hosts. This makes sure that all hosts are
  # in sync, and have reached this point in the code.
  gathered_array = jax.pmap(
      lambda x: jax.lax.all_gather(x, axis_name='i'), axis_name='i'
  )(dummy_array)
  gathered_array = np.reshape(
      gathered_array[0], (num_hosts, num_local_devices, 1)
  )

  # Finally, make sure that the data is exactly what we expect.
  for i in range(num_hosts):
    assert gathered_array[i][0] == i


def parallel_write_images(image_write_fn, img_and_path_list):
  """Parallelizes image writing over JAX hosts and CPU cores.

  Args:
    image_write_fn: A function that takes a tuple as input (path, image) and
      writes the result to disk.
    img_and_path_list: A list of tuples (image, path) containing all the images
      that should be written.
  """
  num_hosts = jax.process_count()
  host_id = jax.process_index()
  num_images = len(img_and_path_list)
  num_images_per_batch = math.ceil(num_images / num_hosts)

  # First shard the images onto each host.
  per_host_images_and_paths = []
  for i in range(num_images_per_batch):
    base_index = i * num_hosts
    global_index = base_index + host_id
    if global_index < num_images:
      per_host_images_and_paths.append(img_and_path_list[global_index])

  # Now within each JAX host, use multi-processing to save the sharded images.
  with multiprocessing.pool.ThreadPool() as pool:
    pool.map(image_write_fn, per_host_images_and_paths)
    pool.close()
    pool.join()


def export_scene(
    log_dir,
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_block_indices,
    planes_features,
    planes_density,
    deferred_mlp_vars,
    occupancy_grids,
    config,
    grid_config,
    data_block_size,
):
  """Exports the baked repr. into a format that can be read by the webviewer."""
  use_sparse_grid = sparse_grid_features is not None
  use_triplanes = planes_features is not None

  # Store one lossless copy of the scene as PNGs.
  baked_dir = path.join(log_dir, 'baked')
  utils.makedirs(baked_dir)

  output_paths = []
  output_images = []

  export_scene_params = {
      'triplane_resolution': config.triplane_resolution,
      'triplane_voxel_size': grid_config['triplane_voxel_size'],
      'sparse_grid_resolution': config.sparse_grid_resolution,
      'sparse_grid_voxel_size': grid_config['sparse_grid_voxel_size'],
      'range_features': config.range_features,
      'range_density': config.range_density,
  }

  if use_triplanes:
    for plane_idx, (plane_features, plane_density) in enumerate(
        zip(planes_features, planes_density)
    ):
      plane_rgb_and_density = np.concatenate(
          [plane_features[Ellipsis, :3], plane_density], axis=-1
      ).transpose([1, 0, 2])
      output_images.append(plane_rgb_and_density)
      output_paths.append(
          path.join(baked_dir, f'plane_rgb_and_density_{plane_idx}.png')
      )
      plane_features = plane_features[Ellipsis, 3:].transpose([1, 0, 2])
      output_images.append(plane_features)
      output_paths.append(
          path.join(baked_dir, f'plane_features_{plane_idx}.png')
      )

  if use_sparse_grid:
    sparse_grid_rgb_and_density = np.copy(
        np.concatenate(
            [sparse_grid_features[Ellipsis, :3], sparse_grid_density], axis=-1
        )
    )  # [..., 4]
    sparse_grid_features = sparse_grid_features[Ellipsis, 3:]  # [..., 4]

    # Subdivide volume into slices and store each slice in a seperate png
    # file. This is nessecary since the browser crashes when tryining to
    # decode a large png that holds the entire volume. Also multiple slices
    # enable parallel decoding in the browser.
    slice_depth = 1
    num_slices = sparse_grid_rgb_and_density.shape[2] // slice_depth

    def write_slices(x, prefix):
      for i in range(0, x.shape[2], slice_depth):
        stack = []
        for j in range(slice_depth):
          plane_index = i + j
          stack.append(x[:, :, plane_index, :].transpose([1, 0, 2]))
        output_path = path.join(baked_dir, f'{prefix}_{i:03d}.png')
        output_images.append(np.concatenate(stack, axis=0))
        output_paths.append(output_path)

    write_slices(sparse_grid_rgb_and_density, 'sparse_grid_rgb_and_density')
    write_slices(sparse_grid_features, 'sparse_grid_features')

    sparse_grid_block_indices_path = path.join(
        baked_dir, 'sparse_grid_block_indices.png'
    )
    sparse_grid_block_indices_image = (
        np.transpose(sparse_grid_block_indices, [2, 1, 0, 3])
        .reshape((-1, sparse_grid_block_indices.shape[0], 3))
        .astype(np.uint8)
    )
    output_paths.append(sparse_grid_block_indices_path)
    output_images.append(sparse_grid_block_indices_image)

    export_scene_params |= {
        'data_block_size': data_block_size,
        'atlas_width': sparse_grid_features.shape[0],
        'atlas_height': sparse_grid_features.shape[1],
        'atlas_depth': sparse_grid_features.shape[2],
        'num_slices': num_slices,
        'slice_depth': slice_depth,
    }

  for occupancy_grid_factor, occupancy_grid in occupancy_grids:
    occupancy_grid_path = path.join(
        baked_dir, f'occupancy_grid_{occupancy_grid_factor}.png'
    )
    # Occupancy grid only holds a single channel, so we create a RGBA image by
    # repeating the channel 4 times over.
    occupany_grid_image = (
        np.transpose(np.repeat(occupancy_grid[Ellipsis, None], 4, -1), [2, 1, 0, 3])
        .reshape((-1, occupancy_grid.shape[0], 4))
        .astype(np.uint8)
    )
    output_images.append(occupany_grid_image)
    output_paths.append(occupancy_grid_path)

  # Also include the network weights of the Deferred MLP in this dictionary.
  export_scene_params |= {
      '0_weights': deferred_mlp_vars['Dense_0']['kernel'].tolist(),
      '1_weights': deferred_mlp_vars['Dense_1']['kernel'].tolist(),
      '2_weights': deferred_mlp_vars['Dense_2']['kernel'].tolist(),
      '0_bias': deferred_mlp_vars['Dense_0']['bias'].tolist(),
      '1_bias': deferred_mlp_vars['Dense_1']['bias'].tolist(),
      '2_bias': deferred_mlp_vars['Dense_2']['bias'].tolist(),
  }

  parallel_write_images(save_8bit_png, list(zip(output_images, output_paths)))

  scene_params_path = path.join(baked_dir, 'scene_params.json')
  utils.save_json(export_scene_params, scene_params_path)

  # Calculate disk consumption.
  total_storage_in_mib = 0
  for output_path in output_paths:
    if 'occupancy_grid' not in output_path:
      storage_in_mib = os.stat(output_path).st_size / (1024**2)
      total_storage_in_mib += storage_in_mib
  storage_path = path.join(log_dir, 'storage.json')
  print(f'Disk consumption: {total_storage_in_mib:.2f} MiB')
  utils.save_json(total_storage_in_mib, storage_path)

  print(f'Exported scene to {baked_dir}')

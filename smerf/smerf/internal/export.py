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

import gzip
import io
import itertools
import math
import multiprocessing
import os

from absl import logging
import chex
from diffren.jax import camera
from etils import epath
import flax
import gin
import jax
import numpy as np
from PIL import Image
from smerf.internal import baking
from smerf.internal import utils


def save_8bit_png(img_and_path):
  """Save an 8bit numpy array as a PNG on disk.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit, [height, width,
      channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path
  with utils.open_file(pth, 'wb') as imgout:
    Image.fromarray(img).save(imgout, 'PNG')


def save_npy_gzip(img_and_path):
  """Save a numpy array with np.save.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit, [height, width,
      channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path

  # Write .npy file to in-memory buffer.
  bytes_io = io.BytesIO()
  np.save(bytes_io, img, allow_pickle=False)

  # Compress .npy file.
  content = gzip.compress(bytes_io.getvalue())

  # Write compressed bytes to disk.
  with utils.open_file(pth, 'wb') as imgout:
    imgout.write(content)


def save_bytes_gzip(img_and_path):
  """Save a numpy array as raw bytes with gzip.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit, [height, width,
      channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path
  with utils.open_file(pth, 'wb') as imgout:
    imgout.write(gzip.compress(img.tobytes()))


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


def create_slices(x, num_slices):
  """Constructs GPU-friendly slices of x.

  Args:
    x: dtype[n, ..., 3 or 4]. Array with values to store. Finals dims must
      correspond to RGBA channels. Values must fit in uint8. Leading dims
      correspond to spatial dimensions (e.g. height, width, depth, ...)
    num_slices: int. Number of slices to build. Must evenly divide number of
      rows to be written. Larger values splits x across a greater number of
      slices.

  Returns:
    list of uint8[?,n,4] of length num_slices. GPU-ready textures.
  """
  assert np.all((-1 <= x) & (x < 256)), 'array must be castable to uint8'
  assert x.shape[-1] in [1, 2, 3, 4], 'array must have 1, 2, 3, or 4 channels'

  x = reverse_spatial_dims(x)  # GPU-friendly dim order.
  x = np.reshape(x, (-1, *x.shape[-2:]))  # Merge all but one spatial dim.
  x = x.astype(np.uint8)  # dtype for GPU.

  if num_slices == 0:
    return []

  num_rows = x.shape[0]
  if num_rows % num_slices != 0:
    raise ValueError(
        f'{num_rows=} must be evenly divisible by {num_slices=}')

  slices = []
  num_rows_per_slice = num_rows // num_slices  # rows per image.
  for i in range(0, num_rows, num_rows_per_slice):
    slices.append(x[i:i+num_rows_per_slice, Ellipsis])
  return slices


def zdiff_slices(slices):
  """Set output[i] = slice[i] - slice[i-1]."""
  result = []
  for i in range(len(slices)):
    if i == 0:
      result.append(slices[i])
    else:
      result.append(slices[i] - slices[i - 1])
  return result


def reverse_spatial_dims(x):
  """Reverses all dims except for the last."""
  dims = list(range(len(x.shape)))
  *spatial_dims, channel_dim = dims
  spatial_dims = tuple(reversed(spatial_dims))
  return np.transpose(x, (*spatial_dims, channel_dim))


def create_slice_names(dirname, fmt):
  """Create filenames for slices.

  Args:
    dirname: epath.Path directory
    fmt: Format string with optional placeholder `i`. If the format string does
      not mention `i`, the same filepath will be yielded over and over.

  Yields:
    epath.Path instances with formatted filename and dir.
  """
  for i in itertools.count():
    name = dirname / fmt.format(i=i)
    yield name


def append_rgba_dim(x):
  """Adds a length-4 dim for RGBA."""
  return np.repeat(x[Ellipsis, None], 4, axis=-1)


def export_scene(
    baked_dir,
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_block_indices,
    planes_features,
    planes_density,
    deferred_mlp_vars,
    packed_occupancy_grids,
    distance_grids,
    sm_idx,
    config,
):
  """Exports the baked repr. into a format that can be read by the webviewer."""
  grid_config = config.grid_config

  use_sparse_grid = sparse_grid_features is not None
  use_triplanes = planes_features is not None

  # Store one lossless copy of the scene.
  baked_dir.mkdir(parents=True, exist_ok=True)

  output_paths = []
  output_images = []
  output_metadata = []

  # Choose which file format to use when writing arrays.
  assert config.export_array_format in ['png', 'raw.gz', 'npy.gz']
  ext = config.export_array_format
  save_array_fn = {
      'png': save_8bit_png,
      'raw.gz': save_bytes_gzip,
      'npy.gz': save_npy_gzip,
  }[ext]

  export_scene_params = {
      'triplane_resolution': config.triplane_resolution,
      # 'triplane_voxel_size': grid_config['triplane_voxel_size'],
      'sparse_grid_resolution': config.sparse_grid_resolution,
      # 'sparse_grid_voxel_size': grid_config['sparse_grid_voxel_size'],
      'range_features': config.range_features,
      'range_density': config.range_density,
      'merge_features_combine_op': config.merge_features_combine_op,
      'deferred_rendering_mode': config.deferred_rendering_mode,
      'submodel_idx': sm_idx,
      'export_apply_zdiff_to_slices': config.export_apply_zdiff_to_slices,
      'export_array_format': config.export_array_format,
      'export_slice_occupancy_and_distance_grids': (
          config.export_slice_occupancy_and_distance_grids
      ),
      'export_pad_occupancy_and_distance_grids': (
          config.export_pad_occupancy_and_distance_grids
      ),
      'export_store_deferred_mlp_separately': (
          config.export_store_deferred_mlp_separately
      ),
      'export_store_rgb_and_density_separately': (
          config.export_store_rgb_and_density_separately
      ),
      **grid_config,
      **config.exposure_config,
  }

  def write_slices(x, fmt, num_slices):
    """Helper function for preparing sliced image data."""
    slices = create_slices(x, num_slices)
    if config.export_apply_zdiff_to_slices:
      slices = zdiff_slices(slices)
    slice_names = create_slice_names(baked_dir, fmt)
    for slice_idx, image, output_path in zip(
        itertools.count(), slices, slice_names
    ):
      output_filename = f'{output_path.name}.{ext}'
      output_path = output_path.parent / output_filename
      output_images.append(image)
      output_paths.append(os.fspath(output_path))
      output_metadata.append({
          'dtype': image.dtype.name,
          'shape': image.shape,
          'format': ext,
          'filename': output_filename,
          'slice_idx': slice_idx,
          'slice_count': len(slices),
      })

  if use_triplanes:
    for plane_idx, (plane_features, plane_density) in enumerate(
        zip(planes_features, planes_density)
    ):
      plane_features_image = plane_features[Ellipsis, 3:]
      write_slices(
          x=plane_features_image,
          fmt=f'plane_features_{plane_idx}',
          num_slices=1,
      )

      if config.export_store_rgb_and_density_separately:
        # Store rgb and density separately.
        plane_rgb_image = plane_features[Ellipsis, :3]  # uint8[..., 3]
        write_slices(
            x=plane_rgb_image,
            fmt=f'plane_rgb_{plane_idx}',
            num_slices=1,
        )
        plane_density_image = plane_density  # uint8[..., 1]
        write_slices(
            x=plane_density_image,
            fmt=f'plane_density_{plane_idx}',
            num_slices=1,
        )
      else:
        # Store rgb and density together.
        plane_rgb_and_density_image = np.concatenate(
            [plane_features[Ellipsis, :3], plane_density], axis=-1
        )  # uint8[..., 4]
        write_slices(
            x=plane_rgb_and_density_image,
            fmt=f'plane_rgb_and_density_{plane_idx}',
            num_slices=1,
        )

  if use_sparse_grid:
    write_slices(
        x=sparse_grid_block_indices,
        fmt='sparse_grid_block_indices',
        num_slices=1,
    )

    # Subdivide volume into slices and store each slice in a seperate
    # file. This is necessary since the browser crashes when trying to
    # decode a large png that holds the entire volume. Also multiple slices
    # enable parallel decoding in the browser.
    slice_depth = 1
    num_slices = sparse_grid_features.shape[2] // slice_depth

    # The sparse grid density image may contain the last feature of
    # sparse_grid_features if it's used to weigh contributions from
    # the triplane representation. This information is NOT redundantly
    # stored in the "features" image.
    sparse_grid_features_image = sparse_grid_features[Ellipsis, 3:]  # uint8[..., 4]
    if config.export_store_rgb_and_density_separately:
      sparse_grid_features_image = sparse_grid_features_image[Ellipsis, :-1]
    write_slices(
        x=sparse_grid_features_image,
        fmt='sparse_grid_features_{i:03d}',
        num_slices=num_slices,
    )

    if config.export_store_rgb_and_density_separately:
      # Store rgb and density separately.
      sparse_grid_rgb_image = sparse_grid_features[Ellipsis, :3]  # uint8[..., 3]
      write_slices(
          x=sparse_grid_rgb_image,
          fmt='sparse_grid_rgb_{i:03d}',
          num_slices=num_slices,
      )

      # Density contains the last feature from features as well.
      sparse_grid_density_image = np.concatenate(
          [sparse_grid_density, sparse_grid_features[Ellipsis, -1:]], axis=-1
      )  # uint8[..., 2]
      write_slices(
          x=sparse_grid_density_image,
          fmt='sparse_grid_density_{i:03d}',
          num_slices=num_slices,
      )

    else:
      # Store rgb and density together.
      sparse_grid_rgb_and_density_image = np.concatenate(
          [sparse_grid_features[Ellipsis, :3], sparse_grid_density], axis=-1
      )  # uint8[..., 4]
      write_slices(
          x=sparse_grid_rgb_and_density_image,
          fmt='sparse_grid_rgb_and_density_{i:03d}',
          num_slices=num_slices,
      )

    export_scene_params |= {
        'data_block_size': config.data_block_size,
        'atlas_width': sparse_grid_features.shape[0],
        'atlas_height': sparse_grid_features.shape[1],
        'atlas_depth': sparse_grid_features.shape[2],
        'num_slices': num_slices,
        'slice_depth': slice_depth,
    }

  def num_slices_for_grid(x):
    """Calculates number of slices for a grid."""
    num_slices = 1
    if config.export_slice_occupancy_and_distance_grids and x.shape[0] > 256:
      num_slices = 8
    return num_slices

  def write_grid_slices(grids, grid_name):
    """Helper function for preparing sliced grid data."""
    for grid_factor, grid in grids:
      # Verify shape.
      chex.assert_rank(grid, 3)
      k = grid.shape[0]
      chex.assert_shape(grid, (k, k, k))

      # Only break into slices if the grid resolution exceeds 256^3.
      num_slices = num_slices_for_grid(grid)
      fmt = (
          f'{grid_name}_{grid_factor}_{{i:03d}}'
          if config.export_slice_occupancy_and_distance_grids
          else f'{grid_name}_{grid_factor}'
      )

      # Grids only holds a single channel, so we create a RGBA image by
      # repeating the channel 4 times over.
      if config.export_pad_occupancy_and_distance_grids:
        grid = append_rgba_dim(grid)
      else:
        grid = grid[Ellipsis, None]  # [k,k,k,1]

      write_slices(x=grid, fmt=fmt, num_slices=num_slices)

  # Write (possibly sliced) occupancy and distance grids.
  write_grid_slices(packed_occupancy_grids, 'occupancy_grid')
  write_grid_slices(distance_grids, 'distance_grid')

  # Also include the network weights of the Deferred MLP in this dictionary.
  # Each parameter has a name like "ResampleDense_0/kernel".
  exported_deferred_mlp_vars = _export_deferred_mlp_vars(deferred_mlp_vars)
  if config.export_store_deferred_mlp_separately:
    deferred_mlp_path = baked_dir / 'deferred_mlp.json.gz'
    utils.save_json_gz(exported_deferred_mlp_vars, deferred_mlp_path)
  else:
    export_scene_params['deferred_mlp'] = exported_deferred_mlp_vars

  # Store DeferredMLP activation function.
  deferred_mlp_activation = 'relu'
  gin_deferred_mlp_bindings = gin.get_bindings('DeferredMLP')
  if 'net_activation' in gin_deferred_mlp_bindings:
    deferred_mlp_activation = gin_deferred_mlp_bindings[
        'net_activation'
    ].__name__

  if deferred_mlp_activation not in ['elu', 'relu']:
    raise ValueError(f'Unsupported activation: {deferred_mlp_activation}')
  export_scene_params['activation'] = deferred_mlp_activation

  if (
      config.use_triplane_weights_for_density
      != config.use_low_res_features_as_weights
  ):
    raise ValueError(
        'The web viewer only supports use_low_res_features_as_weights and'
        ' use_triplane_weights_for_density together.'
    )
  export_scene_params['feature_gating'] = config.use_low_res_features_as_weights

  # Store metadata about each image.
  export_scene_params['asset_metadata'] = output_metadata

  parallel_write_images(save_array_fn, list(zip(output_images, output_paths)))

  scene_params_path = baked_dir / 'scene_params.json'
  utils.save_json(export_scene_params, scene_params_path)

  scene_params_path_gz = baked_dir / 'scene_params.json.gz'
  utils.save_json_gz(export_scene_params, scene_params_path_gz)

  # Calculate memory, disk consumption.
  storage_stats = jax.tree_util.tree_map(
      estimate_storage_stats,
      output_paths + [scene_params_path],
      output_images + [deferred_mlp_vars],
  )
  aggregate_stats = aggregate_storage_stats(storage_stats)
  logging.info(f'{aggregate_stats=}')  # pylint: disable=logging-fstring-interpolation

  storage_path = baked_dir / 'storage.json'
  utils.save_json(storage_stats, storage_path)

  print(f'Exported scene to {baked_dir}')


def export_test_cameras(baked_dir, test_dataset, sm_idx, config, grid_config):
  """Exports the test cameras to a format that can be read by the webviewer."""
  near = max(test_dataset.near, 0.02)
  far = min(test_dataset.far, 2500.0)
  pose_dict = {'test_frames': []}
  for index in range(test_dataset.images.shape[0]):
    # Determine if this image belongs to this submodel. If not, skip it.
    cam_sm_idx = baking.sm_idx_for_camera(
        test_dataset, index, config, grid_config
    )
    if cam_sm_idx != sm_idx:
      continue

    position, rotation, projection = _webviewer_camera_from_index(
        index, test_dataset, near, far
    )
    pose_dict['test_frames'].append({
        # dtype=np.float64, shape=(3,) flattened to a list of length 3.
        'position': position.tolist(),
        # dtype=np.float64, the shape=(4, 4) flattened to a list of length 16.
        'rotation': rotation.T.flatten().tolist(),
        # dtype=np.float64, shape=(4, 4) flattened to a list of length 16.
        'projection': projection.T.flatten().tolist(),
    })

  json_pose_path = baked_dir / 'test_frames.json'
  utils.save_json(pose_dict, json_pose_path)


def _webviewer_camera_from_index(index, dataset, near, far):
  """Returns a camera pos/rot/proj matrix following the OpenGL conventions."""
  webviewer_t_world = np.diag(np.array([-1, 1, 1]))
  webviewer_t_world[:, [1, 2]] = webviewer_t_world[:, [2, 1]]
  webviewer_position = webviewer_t_world @ dataset.camtoworlds[index][:3, 3]
  webviewer_rotation = np.identity(4)
  webviewer_rotation[:3, :3] = (
      webviewer_t_world @ dataset.camtoworlds[index][:3, :3]
  )
  intrinsics = np.linalg.inv(dataset.pixtocams[index])
  focal_x = intrinsics[0, 0]
  focal_y = intrinsics[1, 1]
  center_offset_x = intrinsics[0, 2] - dataset.width / 2
  center_offset_y = dataset.height / 2 - intrinsics[1, 2]

  webviewer_projection = camera.perspective_from_intrinsics(
      focal_x, focal_y, center_offset_x, center_offset_y, near, far,
      dataset.width, dataset.height)
  return webviewer_position, webviewer_rotation, webviewer_projection


def _export_deferred_mlp_vars(variables):
  """Export DeferredMLP variables."""
  # Remove nesting. Nested keys are separated by "/".
  variables = flax.traverse_util.flatten_dict(variables, sep='/')

  # Convert each array to a flat list of values and a shape.
  def format_array(x):
    return {'shape': x.shape, 'data': x.flatten().tolist()}

  variables = jax.tree_util.tree_map(format_array, variables)

  return variables


def estimate_storage_stats(filepath, representation):
  """Estimates storage statistics for a single file.

  Args:
    filepath: Path to serialized file.
    representation: PyTree of numpy arrays. In-memory representation of
      filepath.

  Returns:
    Statistics about file and its footprint in memory and on disk.
  """
  filename = epath.Path(filepath).name

  special_prefixes = ['distance_grid_', 'occupancy_grid_']
  if (
      any([filename.startswith(prefix) for prefix in special_prefixes])
  ):
    # Undo append_rgba_dim().
    assert representation.shape[-1] in [1, 4], representation.shape
    representation = representation[Ellipsis, 0]

  disk_mb = epath.Path(filepath).stat().length / 1e6
  memory_mb = jax.tree_util.tree_map(lambda x: x.nbytes / 1e6, representation)
  aggregate_memory_mb = jax.tree_util.tree_reduce(
      lambda x, y: x + y, memory_mb, initializer=0.0
  )
  return {
      'filename': filename,
      'memory_mb': aggregate_memory_mb,
      'disk_mb': disk_mb,
  }


def aggregate_storage_stats(storage_stats):
  """Aggregates all storage statistics."""
  sum_of_values = lambda key: sum(entry[key] for entry in storage_stats)
  return {
      'memory_mb': sum_of_values('memory_mb'),
      'disk_mb': sum_of_values('disk_mb'),
  }

# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Functions that export a baked SNeRG model for viewing in the web-viewer."""

import json
import math
import multiprocessing
import jax
import numpy as np
from PIL import Image
import tensorflow as tf

from snerg.nerf import utils


def save_8bit_png(img_and_path):
  """Save an 8bit numpy array as a PNG on disk.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit,
      [height, width, channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path
  with utils.open_file(pth, 'wb') as imgout:
    Image.fromarray(img).save(imgout, 'PNG')


def synchronize_jax_hosts():
  """Makes sure that the JAX hosts have all reached this point."""
  # Build an array containing the host_id.
  num_local_devices = jax.local_device_count()
  num_hosts = jax.host_count()
  host_id = jax.host_id()
  dummy_array = np.ones((num_local_devices, 1), dtype=np.int32) * host_id

  # Then broadcast it between all JAX hosts. This makes sure that all hosts are
  # in sync, and have reached this point in the code.
  gathered_array = jax.pmap(
      lambda x: jax.lax.all_gather(x, axis_name='i'), axis_name='i')(
          dummy_array)
  gathered_array = np.reshape(
      gathered_array[0], (num_hosts, num_local_devices, 1))

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
  num_hosts = jax.host_count()
  host_id = jax.host_id()
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

def export_snerg_scene_raw(output_directory, atlas, atlas_block_indices,
                       viewdir_mlp_params, render_params, atlas_params,
                       scene_params, input_height, input_width, input_focal):
  
  np.save(output_directory+'/indirection_grid.npy', atlas_block_indices)

  np.save(output_directory+'/atlas.npy', atlas)

  import pickle

  def pickle_file(output_directory, obj_to_pickle, filename):
    with open(output_directory+f'/{filename}.pickle', 'wb') as handle:
        pickle.dump(obj_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

  pickle_file(output_directory, viewdir_mlp_params, 'viewdir_mlp_params')

  pickle_file(output_directory, render_params, 'render_params')

  pickle_file(output_directory, atlas_params, 'atlas_params')

  pickle_file(output_directory, scene_params, 'scene_params')

  input_data = {
    'input_height':input_height,
    'input_width':input_width,
    'input_focal':input_focal,
  }
  pickle_file(output_directory, input_data, 'input_data')
    
def export_snerg_scene(output_directory, atlas, atlas_block_indices,
                       viewdir_mlp_params, render_params, atlas_params,
                       scene_params, input_height, input_width, input_focal):
  """Exports a scene to web-viewer format: a collection of PNGs and a JSON file.

  The scene gets exported to output_directory/png. Any previous results will
  be overwritten.

  Args:
    output_directory: The root directory where the scene gets written.
    atlas: The SNeRG scene packed as a texture atlas in a [S, S, N, C] numpy
      array, where the channels C contain both RGB and features.
    atlas_block_indices: The indirection grid of the SNeRG scene, represented as
      a numpy int32 array of size (bW, bH, bD, 3).
    viewdir_mlp_params: A dict containing the MLP parameters for the per-sample
      view-dependence MLP.
    render_params: A dict with parameters for high-res rendering.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    input_height: Height (pixels) of the NDC camera (i.e. the training cameras).
    input_width: Width (pixels) of the NDC camera (i.e. the training cameras).
    input_focal: Focal length (pixels) of the NDC camera (i.e. the
      training cameras).
  """
  # Slice the atlas into images.
  rgbs = []
  alphas = []
  for i in range(0, atlas.shape[2], 4):
    rgb_stack = []
    alpha_stack = []
    for j in range(4):
      plane_index = i + j
      rgb_stack.append(atlas[:, :, plane_index, :][Ellipsis,
                                                   0:3].transpose([1, 0, 2]))
      alpha_stack.append(
          atlas[:, :,
                plane_index, :][Ellipsis,
                                scene_params['_channels']].transpose([1, 0]))
    rgbs.append(np.concatenate(rgb_stack, axis=0))
    alphas.append(np.concatenate(alpha_stack, axis=0))

  atlas_index_image = np.transpose(atlas_block_indices, [2, 1, 0, 3]).reshape(
      (-1, atlas_block_indices.shape[0], 3)).astype(np.uint8)

  # Build a dictionary of the scene parameters, so we can export it as a json.
  export_scene_params = {}
  export_scene_params['voxel_size'] = float(render_params['_voxel_size'])
  export_scene_params['block_size'] = atlas_params['_data_block_size']
  export_scene_params['grid_width'] = int(render_params['_grid_size'][0])
  export_scene_params['grid_height'] = int(render_params['_grid_size'][1])
  export_scene_params['grid_depth'] = int(render_params['_grid_size'][2])
  export_scene_params['atlas_width'] = atlas.shape[0]
  export_scene_params['atlas_height'] = atlas.shape[1]
  export_scene_params['atlas_depth'] = atlas.shape[2]
  export_scene_params['num_slices'] = len(rgbs)

  export_scene_params['min_x'] = float(scene_params['min_xyz'][0])
  export_scene_params['min_y'] = float(scene_params['min_xyz'][1])
  export_scene_params['min_z'] = float(scene_params['min_xyz'][2])

  export_scene_params['atlas_blocks_x'] = int(atlas.shape[0] /
                                              atlas_params['atlas_block_size'])
  export_scene_params['atlas_blocks_y'] = int(atlas.shape[1] /
                                              atlas_params['atlas_block_size'])
  export_scene_params['atlas_blocks_z'] = int(atlas.shape[2] /
                                              atlas_params['atlas_block_size'])

  export_scene_params['input_height'] = float(input_height)
  export_scene_params['input_width'] = float(input_width)
  export_scene_params['input_focal'] = float(input_focal)

  export_scene_params['worldspace_T_opengl'] = scene_params[
      'worldspace_T_opengl'].tolist()
  export_scene_params['ndc'] = scene_params['ndc']

  # Also include the network weights in this dictionary.
  export_scene_params['0_weights'] = viewdir_mlp_params['params']['Dense_0'][
      'kernel'].tolist()
  export_scene_params['1_weights'] = viewdir_mlp_params['params']['Dense_1'][
      'kernel'].tolist()
  export_scene_params['2_weights'] = viewdir_mlp_params['params']['Dense_3'][
      'kernel'].tolist()
  export_scene_params['0_bias'] = viewdir_mlp_params['params']['Dense_0'][
      'bias'].tolist()
  export_scene_params['1_bias'] = viewdir_mlp_params['params']['Dense_1'][
      'bias'].tolist()
  export_scene_params['2_bias'] = viewdir_mlp_params['params']['Dense_3'][
      'bias'].tolist()

  # To avoid partial overwrites, first dump the scene to a temporary directory.
  output_tmp_directory = output_directory + '/temp'

  if jax.host_id() == 0:
    # Delete the folder if it already exists.
    if utils.isdir(output_tmp_directory):
      tf.io.gfile.rmtree(output_tmp_directory)
    utils.makedirs(output_tmp_directory)

  # Now store the indirection grid.
  atlas_indices_path = '%s/atlas_indices.png' % output_tmp_directory
  if jax.host_id() == 0:
    save_8bit_png((atlas_index_image, atlas_indices_path))

  # Make sure that all JAX hosts have reached this point in the code before we
  # proceed. Things will get tricky if output_tmp_directory doesn't yet exist.
  synchronize_jax_hosts()

  # Save the alpha values and RGB colors as one set of PNG images.
  output_images = []
  output_paths = []
  for i, rgb_and_alpha in enumerate(zip(rgbs, alphas)):
    rgb, alpha = rgb_and_alpha
    rgba = np.concatenate([rgb, np.expand_dims(alpha, -1)], axis=-1)
    uint_multiplier = 2.0**8 - 1.0
    rgba = np.minimum(uint_multiplier,
                      np.maximum(0.0, np.floor(uint_multiplier * rgba))).astype(
                          np.uint8)
    output_images.append(rgba)
    atlas_rgba_path = '%s/rgba_%03d.png' % (output_tmp_directory, i)
    output_paths.append(atlas_rgba_path)

  # Save the computed features a separate collection of PNGs.
  uint_multiplier = 2.0**8 - 1.0
  for i in range(0, atlas.shape[2], 4):
    feature_stack = []
    for j in range(4):
      plane_index = i + j
      feature_slice = atlas[:, :, plane_index, :][Ellipsis,
                                                  3:-1].transpose([1, 0, 2])
      feature_slice = np.minimum(
          uint_multiplier,
          np.maximum(0.0, np.floor(uint_multiplier * feature_slice))).astype(
              np.uint8)
      feature_stack.append(feature_slice)
    output_images.append(np.concatenate(feature_stack, axis=0))

  for i in range(len(rgbs)):
    output_paths.append('%s/feature_%03d.png' % (output_tmp_directory, i))

  parallel_write_images(save_8bit_png, list(zip(output_images, output_paths)))

  # Now export the scene parameters and the network weights as a JSON.
  export_scene_params['format'] = 'png'
  scene_params_path = '%s/scene_params.json' % output_tmp_directory
  if jax.host_id() == 0:
    with utils.open_file(scene_params_path, 'wb') as f:
      f.write(json.dumps(export_scene_params).encode('utf-8'))

  # Again, make sure that the JAX hosts are in sync. Don't delete
  # output_tmp_directory before all files have been written.
  synchronize_jax_hosts()

  # Finally move the scene to the appropriate output path.
  output_png_directory = output_directory + '/png'
  if jax.host_id() == 0:
    # Delete the folder if it already exists.
    if utils.isdir(output_png_directory):
      tf.io.gfile.rmtree(output_png_directory)
    tf.io.gfile.rename(output_tmp_directory, output_png_directory)


def compute_scene_size(output_directory, atlas_block_indices, atlas_params,
                       scene_params):
  """Computes the size of an exported SNeRG scene.

  Args:
    output_directory: The root directory where the SNeRG scene was written.
    atlas_block_indices: The indirection grid of the SNeRG scene.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    png_size_gb: The scene size (in GB) when stored as compressed 8-bit PNGs.
    byte_size_gb: The scene size (in GB), stored as uncompressed 8-bit integers.
    float_size_gb: The scene size (in GB), stored as uncompressed 32-bit floats.
  """

  output_png_directory = output_directory + '/png'
  png_files = [
      output_png_directory + '/' + f
      for f in sorted(utils.listdir(output_png_directory))
      if f.endswith('png')
  ]
  png_size_gb = sum(
      [tf.io.gfile.stat(f).length / (1000 * 1000 * 1000) for f in png_files])

  block_index_size_gb = np.array(
      atlas_block_indices.shape).prod() / (1000 * 1000 * 1000)

  active_atlas_blocks = (atlas_block_indices[Ellipsis, 0] >= 0).sum()
  active_atlas_voxels = (
      active_atlas_blocks * atlas_params['atlas_block_size']**3)
  active_atlas_channels = active_atlas_voxels * scene_params['_channels']

  byte_size_gb = active_atlas_channels / (1000 * 1000 *
                                          1000) + block_index_size_gb
  float_size_gb = active_atlas_channels * 4 / (1000 * 1000 *
                                               1000) + block_index_size_gb

  return png_size_gb, byte_size_gb, float_size_gb

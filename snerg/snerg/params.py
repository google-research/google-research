# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Parameters and parameter-related functions for SNeRG baking."""

import numpy as np


def initialize_params(args):
  """Initializes the SNeRG params from command line FLAGS.

  Args:
    args: A dictionary containing all the SNeRG parameters.

  Returns:
    render_params: A dict with parameters for high-res rendering.
    culling_params: A dict for low-res rendering and opacity/visibility culling.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
  """
  #
  # Rendering settings.
  #
  render_params = {}

  # To prevent aliasing, we prefilter the NeRF volume with a 3D Gaussian on XYZ.
  # This sigma controls how much to blur. Default: 1/sqrt(12).
  render_params['voxel_filter_sigma'] = args.voxel_filter_sigma

  # How many samples to use for the spatial prefilter and the direction samples.
  render_params['num_samples_per_voxel'] = args.num_samples_per_voxel

  # The resolution of the voxel grid along the longest edge of the volume.
  render_params['voxel_resolution'] = args.voxel_resolution

  #
  # Culling settings.
  #
  culling_params = {}

  # The block size used for visibility and alpha culling.
  culling_params['block_size'] = args.culling_block_size

  # We discard any atlas blocks where the max alpha is below this threshold.
  culling_params['alpha_threshold'] = args.alpha_threshold

  # We threshold on visiblity = max(1 - alpha_along_ray), to create a visibility
  # mask for all atlas blocks in the scene.
  culling_params['visibility_threshold'] = args.visibility_threshold

  # Speedup: Scale the images by this factor when computing visibilities.
  culling_params['visibility_image_factor'] = args.visibility_image_factor

  # Speedup: Only process every Nth image when computing visibilities.
  culling_params[
      'visibility_subsample_factor'] = args.visibility_subsample_factor

  # This makes the visibility grid conservative by dilating it slightly.
  culling_params['visibility_grid_dilation'] = args.visibility_grid_dilation

  # Atlas settings.
  atlas_params = {}

  # Make sure to use a multiple of 16, so this fits nicely within image/video
  # compression macroblocks.
  atlas_params['atlas_block_size'] = args.atlas_block_size

  # We store the atlas as a collection of atlas_size * atlas_size images.
  atlas_params['atlas_slice_size'] = args.atlas_slice_size

  #
  # Scene-specific settings.
  #
  scene_params = {}

  if args.flip_scene_coordinates:
    scene_params['worldspace_T_opengl'] = np.array([[-1.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0, 0.0],
                                                    [0.0, 1.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]])
  elif (args.dataset == 'llff' and not args.spherify):
    # Use a hard-coded flip for NDC scenes.
    scene_params['worldspace_T_opengl'] = np.array([[1.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0, 0.0],
                                                    [0.0, 0.0, -1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]])
  else:
    scene_params['worldspace_T_opengl'] = np.array([[1.0, 0.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]])

  scene_params['ndc'] = (args.dataset == 'llff' and not args.spherify)
  scene_params['voxel_resolution'] = args.voxel_resolution
  scene_params['white_bkgd'] = args.white_bkgd
  scene_params['near'] = args.near
  scene_params['far'] = args.far
  scene_params['min_xyz'] = np.array([-1.0, -1.0, -1.0]) * args.snerg_box_scale
  scene_params['max_xyz'] = np.array([1.0, 1.0, 1.0]) * args.snerg_box_scale
  scene_params['dtype'] = args.snerg_dtype

  # The number of network queries to perform at a time. Lower this number if
  # you run out of GPU or TPU memory.
  scene_params['chunk_size'] = args.snerg_chunk_size

  # Define internal parameters, which can be derived from the arguments passed
  # above. We denote internal parameters with a leading underscore.

  # Pack dataset related params into scene_params.
  scene_params['_use_pixel_centers'] = args.use_pixel_centers

  # Also incorporate the relevant params for the MLPs in scene_params.
  scene_params['_net_depth'] = args.net_depth
  scene_params['_net_width'] = args.net_width
  scene_params['_skip_layer'] = args.skip_layer
  scene_params['_num_sigma_channels'] = args.num_sigma_channels
  scene_params['_viewdir_net_depth'] = args.viewdir_net_depth
  scene_params['_viewdir_net_width'] = args.viewdir_net_width
  scene_params['_num_rgb_channels'] = args.num_rgb_channels
  scene_params['_min_deg_point'] = args.min_deg_point
  scene_params['_max_deg_point'] = args.max_deg_point
  scene_params['_legacy_posenc_order'] = args.legacy_posenc_order
  scene_params['_deg_view'] = args.deg_view

  # Derive the remaining internal params from the values we've already provided.
  post_process_params(render_params, culling_params, atlas_params, scene_params)

  return render_params, culling_params, atlas_params, scene_params


def post_process_params(render_params, culling_params, atlas_params,
                        scene_params):
  """Derives internal SNeRG parameters and adds them to the param dictionaries.

  Note that this modifies the dictionaries passed as arguments. The internal
  parameters are denoted by leading underscores.

  Args:
    render_params: A dict with parameters for high-res rendering.
    culling_params: A dict for low-res rendering and opacity/visibility culling.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
  """

  # We pack the atlas with a 1-voxel padding to avoid interpolation artifacts.
  atlas_params['_data_block_size'] = atlas_params['atlas_block_size'] - 2

  # Compute the grid size in world-space and the voxel size in world space.
  render_params['_voxel_size'] = (
      scene_params['max_xyz'] -
      scene_params['min_xyz']).max() / render_params['voxel_resolution']
  render_params['_grid_size'] = np.ceil(
      (scene_params['max_xyz'] - scene_params['min_xyz']) /
      render_params['_voxel_size']).astype(np.int32)

  # Now figure out how many atlas blocks are needed to cover the entire grid.
  atlas_params['_atlas_grid_size'] = np.ceil(
      render_params['_grid_size'] / atlas_params['_data_block_size']).astype(
          np.int32)
  atlas_params['_atlas_voxel_size'] = render_params[
      '_voxel_size'] * atlas_params['_data_block_size']

  # Finally, compute the grid size and voxel size for the smaller grid used
  # for visibility and alpha culling.
  culling_params['_voxel_size'] = render_params['_voxel_size'] * atlas_params[
      '_data_block_size'] / culling_params['block_size']
  culling_params['_grid_size'] = (
      (render_params['_grid_size'] * culling_params['block_size']) /
      atlas_params['_data_block_size']).astype(np.int32)

  # We use 7 channels for baking: RGBA + 3 features.
  scene_params['_channels'] = 7

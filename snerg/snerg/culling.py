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
"""Functions to cull empty space based on a low-res grid."""

import numpy as np

from snerg.nerf import datasets
from snerg.snerg import params


def crop_alpha_grid(render_params,
                    culling_params,
                    atlas_params,
                    scene_params,
                    culling_grid_alpha,
                    percentile_threshold=0.5,
                    dilation_factor=1.1):
  """Crops the grid to the smallest 3D bounding box which contains the scene.

  This function works by computing the total distribution of alpha mass along
  each axis. Then we cut the tails off the distribution (say, leaving 1% of the
  alpha mass outside the cropped region). To account for structures just outside
  this boundary, we then expand the cropped bounding box by dilation_factor.

  Args:
    render_params: A dict with parameters for high-res rendering.
    culling_params: A dict for low-res rendering and opacity/visibility culling.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    culling_grid_alpha: A [cW, cH, cD, 1] numpy array for the alpha values in
      the low-res culling grid.
    percentile_threshold: The alpha mass (in percent) we're leaving outside the
      cropped bounding box on each side.
    dilation_factor: The factor we're expanding the cropped bounding box by.

  Returns:
    render_params_cropped render_params after cropping the grid bounds.
    culling_params_cropped: culling_params after cropping the grid bounds.
    atlas_params_cropped: atlas_params after cropping the grid bounds.
    scene_params_cropped: scene_params after cropping the grid bounds.
  """

  cumsum_x = np.cumsum(np.sum(np.sum(culling_grid_alpha, axis=2), axis=1))
  cumsum_y = np.cumsum(np.sum(np.sum(culling_grid_alpha, axis=2), axis=0))
  cumsum_z = np.cumsum(np.sum(np.sum(culling_grid_alpha, axis=1), axis=0))

  cumsum_threshold = percentile_threshold / 100.0  # Convert to a fraction.
  active_voxels_x = np.logical_and(
      cumsum_x > cumsum_threshold * cumsum_x[-1],
      cumsum_x < cumsum_x[-1] * (1 - cumsum_threshold))
  active_voxels_y = np.logical_and(
      cumsum_y > cumsum_threshold * cumsum_y[-1],
      cumsum_y < cumsum_y[-1] * (1 - cumsum_threshold))
  active_voxels_z = np.logical_and(
      cumsum_z > cumsum_threshold * cumsum_z[-1],
      cumsum_z < cumsum_z[-1] * (1 - cumsum_threshold))

  # Early out if there are no active voxels.
  if not (np.max(active_voxels_x) and np.max(active_voxels_y) and
          np.max(active_voxels_y)):
    return render_params, culling_params, atlas_params, scene_params

  min_in_voxels = np.array([
      np.min(np.where(active_voxels_x)),
      np.min(np.where(active_voxels_y)),
      np.min(np.where(active_voxels_z))
  ])
  max_in_voxels = np.array([
      np.max(np.where(active_voxels_x)),
      np.max(np.where(active_voxels_y)),
      np.max(np.where(active_voxels_z))
  ])

  avg_in_voxels = min_in_voxels + 0.5 * (max_in_voxels - min_in_voxels)
  min_in_voxels = (min_in_voxels -
                   avg_in_voxels) * dilation_factor + avg_in_voxels
  max_in_voxels = (max_in_voxels -
                   avg_in_voxels) * dilation_factor + avg_in_voxels

  min_in_world = min_in_voxels * culling_params['_voxel_size'] + scene_params[
      'min_xyz']
  max_in_world = (1.0 + max_in_voxels
                 ) * culling_params['_voxel_size'] + scene_params['min_xyz']

  # Make sure that the cropped bounding box doesn't inflate beyond the initial
  # scene bounds.
  min_in_world = np.maximum(scene_params['min_xyz'], min_in_world)
  max_in_world = np.minimum(scene_params['max_xyz'], max_in_world)

  scene_params_cropped = scene_params.copy()
  render_params_cropped = render_params.copy()
  culling_params_cropped = culling_params.copy()
  atlas_params_cropped = atlas_params.copy()

  scene_params_cropped['min_xyz'] = min_in_world
  scene_params_cropped['max_xyz'] = max_in_world

  # Now update the internal SNeRG parameters to match the cropped grid.
  params.post_process_params(render_params_cropped, culling_params_cropped,
                             atlas_params_cropped, scene_params_cropped)

  return (render_params_cropped, culling_params_cropped, atlas_params_cropped,
          scene_params_cropped)


def rays_aabb_intersection(aabb_min, aabb_max, origins, inv_directions):
  """Intersects rays with axis aligned bounding boxes (AABBs).

  The bounding boxes are represented by their min/max coordinates. Note that
  each ray needs a separate bounding box.

  Args:
    aabb_min: A numpy array [..., 3] containing the coordinate of the corner
      closest to the origin for each AABB.
    aabb_max: A numpy arrayr [...,3] containing the coordinate of the corner
      furthest from the origin for each AABB.
    origins: A numpy array [..., 3] of the ray origins.
    inv_directions: A A numpy array [..., 3] of ray directions. However, each
      channel has been inverted, i.e. (1/dx, 1/dy, 1/dz).

  Returns:
    t_min: A [...] numpy array containing the smallest (signed) distance along
      the rays to the closest intersection point with each AABB.
    t_max: A [...] numpy array  containing the largest (signed) distance along
      the rays to the furthest intersection point with each AABB. Note that if
      t_max < t_min, the ray does not intersect the AABB.
  """
  t1 = (aabb_min - origins) * inv_directions
  t2 = (aabb_max - origins) * inv_directions
  t_min = np.minimum(t1, t2).max(axis=-1)
  t_max = np.maximum(t1, t2).min(axis=-1)
  return t_min, t_max


def integrate_visibility_from_rays(origins, directions, alpha_grid,
                                   visibility_grid, scene_params, grid_params):
  """Ray marches rays through a grid and marks visited voxels as visible.

  This function adds the visibility value (1-accumulated_alpha) into the
  visibility_grid passed as a parameter.

  Args:
    origins: An np.array [N, 3] of the ray origins.
    directions: An np.array [N, 3] of ray directions.
    alpha_grid: A [cW, cH, cD, 1] numpy array for the alpha values in the
       low-res culling grid.
    visibility_grid: A [cW, cH, cD, 1] numpy array for the visibility values in
      the low-res culling grid. Note that this function will be adding
      visibility values into this grid.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    grid_params: A dict with parameters describing the high-res voxel grid which
      the atlas is representing.
  """
  # Extract the relevant parameters from the dictionaries.
  min_xyz = scene_params['min_xyz']
  worldspace_t_opengl = scene_params['worldspace_T_opengl']
  voxel_size = grid_params['_voxel_size']
  grid_size = grid_params['_grid_size']

  # Set up the rays and transform them to the voxel grid coordinate space.
  num_rays = origins.shape[0]
  opengl_t_worldspace = np.linalg.inv(worldspace_t_opengl)

  directions_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
  directions /= directions_norm

  origins_hom = np.concatenate(
      [origins, np.ones_like(origins[Ellipsis, 0:1])], axis=-1)
  origins_hom = origins_hom.reshape((-1, 4)).dot(opengl_t_worldspace)
  origins_opengl = origins_hom[Ellipsis, 0:3].reshape(origins.shape)
  directions_opengl = directions.dot(opengl_t_worldspace[0:3, 0:3])

  origins_grid = (origins_opengl - min_xyz) / voxel_size
  directions_grid = directions_opengl
  inv_directions_grid = 1.0 / directions_grid

  # Now set the near and far distance of each ray to match the cube which
  # the voxel grid is defined in.
  min_distances, max_distances = rays_aabb_intersection(
      np.zeros_like(min_xyz), grid_size, origins_grid, inv_directions_grid)
  invalid_mask = min_distances > max_distances
  min_distances[invalid_mask] = 0
  max_distances[invalid_mask] = 0

  # The NeRF near/far bounds have been set for unnormalized ray directions, so
  # we need to scale our bounds here to compensate for normalizing.
  near_in_voxels = directions_norm[Ellipsis, 0] * scene_params['near'] / voxel_size
  far_in_voxels = directions_norm[Ellipsis, 0] * scene_params['far'] / voxel_size

  min_distances = np.maximum(near_in_voxels, min_distances)
  max_distances = np.maximum(near_in_voxels, max_distances)
  max_distances = np.minimum(far_in_voxels, max_distances)
  num_steps = int(0.5 + max_distances.max() - min_distances.min())

  # Finally, set up the accumulation buffers we need for ray marching.
  raveled_alpha = alpha_grid.ravel()
  total_visibility = np.ones((1, num_rays, 1), dtype=np.float32)
  min_distances = np.expand_dims(min_distances, -1)
  for i in range(num_steps):
    visibility_mask = (total_visibility >= 0.01)[0]
    if not visibility_mask.max():
      break

    current_distances = min_distances + 0.5 + i
    active_current_distances = current_distances[visibility_mask]
    active_max_distances = max_distances[visibility_mask[Ellipsis, 0]]
    if active_current_distances.min() >= active_max_distances.max():
      break

    positions_grid = origins_grid + directions_grid * current_distances

    epsilon = 0.1
    valid_mask = (positions_grid[Ellipsis, 0] < grid_size[Ellipsis, 0] - 0.5 - epsilon)
    valid_mask *= (positions_grid[Ellipsis, 1] < grid_size[Ellipsis, 1] - 0.5 - epsilon)
    valid_mask *= (positions_grid[Ellipsis, 2] < grid_size[Ellipsis, 2] - 0.5 - epsilon)
    valid_mask *= (positions_grid[Ellipsis, 0] > 0.5 + epsilon)
    valid_mask *= (positions_grid[Ellipsis, 1] > 0.5 + epsilon)
    valid_mask *= (positions_grid[Ellipsis, 2] > 0.5 + epsilon)
    invalid_mask = np.logical_not(valid_mask)
    if not valid_mask.max():
      continue

    invalid_mask = np.logical_not(valid_mask)
    positions_grid -= 0.5
    alpha = np.zeros((1, num_rays, 1), dtype=np.float32)

    # Use trilinear interpolation for smoother results.
    offsets_xyz = [(x, y, z)  # pylint: disable=g-complex-comprehension
                   for x in [0.0, 1.0]
                   for y in [0.0, 1.0]
                   for z in [0.0, 1.0]]
    for dx, dy, dz in offsets_xyz:
      grid_indices = np.floor(positions_grid + np.array([dx, dy, dz])).astype(
          np.int32)
      weights = 1 - np.abs(grid_indices - positions_grid)
      weights = weights.prod(axis=-1).reshape(alpha.shape)

      grid_indices[invalid_mask] = 0
      grid_indices = grid_indices.reshape(-1, 3).T

      trilinear_alpha = np.take(
          raveled_alpha, np.ravel_multi_index(grid_indices, alpha_grid.shape))
      alpha += weights * trilinear_alpha.reshape(alpha.shape)

      # Splat the visibility value into the visibility grid.
      weighted_visibilities = weights * total_visibility
      weighted_visibilities[0, invalid_mask] = 0.0
      visibility_grid.ravel()[np.ravel_multi_index(
          grid_indices, visibility_grid.shape)] += (
              weighted_visibilities[0, :, 0])

    alpha[0, invalid_mask] = 0.0
    total_visibility *= 1.0 - alpha


def integrate_visibility_from_image(h, w, focal, camtoworld, alpha_grid,
                                    visibility_grid, scene_params, grid_params):
  """Marks the voxels which are visible from the a given camera.

  A convenient wrapper function around integrate_visibility_from_rays.

  Args:
    h: The image height (pixels).
    w: The image width (pixels).
    focal: The image focal length (pixels).
    camtoworld: A numpy array of shape [4, 4] containing the camera-to-world
      transformation matrix for the camera.
    alpha_grid: A [cW, cH, cD, 1] numpy array for the alpha values in the
       low-res culling grid.
    visibility_grid: A [cW, cH, cD, 1] numpy array for the visibility values in
      the low-res culling grid. Note that this function will be adding
      visibility values into this grid.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    grid_params: A dict with parameters describing the high-res voxel grid which
      the atlas is representing.
  """
  origins, directions, _ = datasets.rays_from_camera(
      scene_params['_use_pixel_centers'], h, w, focal,
      np.expand_dims(camtoworld, 0))
  if scene_params['ndc']:
    origins, directions = datasets.convert_to_ndc(origins, directions, focal, w,
                                                  h)

  integrate_visibility_from_rays(
      origins.reshape(-1, 3), directions.reshape(-1, 3), alpha_grid,
      visibility_grid, scene_params, grid_params)


def max_downsample_grid(culling_params, atlas_params, culling_grid_values):
  """This function performs maximum downsampling for macroblocks in the scene.

  Essentially, it computes the maximum value for each macroblock inside the
  scene volume. The baking code later thresholds these values. We denote
  (cW, cH, cD) to be the 3D dimensions of the culling grid, and (bW, bH, bD) to
  be the 3D dimensions of the lower resolution macroblock grid. Note that
  cW = culling_params['block_size'] * bW.

  Args:
    culling_params: A dict for low-res rendering and opacity/visibility culling.
    atlas_params: A dict with params for building the 3D texture atlas.
    culling_grid_values: A [cW, cH, cD, 1] numpy array for the values in the
       low-res culling grid.

  Returns:
    A [bW, bH, bD, 1] numpy array for containing the maximum value found inide
    each macroblock.
  """
  atlas_grid_maximum = np.zeros(
      (atlas_params['_atlas_grid_size'][0], atlas_params['_atlas_grid_size'][1],
       atlas_params['_atlas_grid_size'][2]),
      dtype=np.float32)

  for block_x in range(atlas_params['_atlas_grid_size'][0]):
    for block_y in range(atlas_params['_atlas_grid_size'][1]):
      for block_z in range(atlas_params['_atlas_grid_size'][2]):
        min_voxel = np.array([block_x, block_y, block_z
                             ]) * culling_params['block_size']
        max_voxel = np.minimum(min_voxel + culling_params['block_size'],
                               culling_params['_grid_size'])
        data_block_size_3d = max_voxel - min_voxel
        if data_block_size_3d.min() == 0:
          continue

        block_values = culling_grid_values[min_voxel[0]:max_voxel[0],
                                           min_voxel[1]:max_voxel[1],
                                           min_voxel[2]:max_voxel[2]]
        atlas_grid_maximum[block_x, block_y, block_z] = block_values.max()

  return atlas_grid_maximum

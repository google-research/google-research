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

"""Functions that render a baked SNeRG model directly in python."""

import numpy as np
import tensorflow as tf

from snerg.nerf import datasets
from snerg.snerg import model_utils


#
# Fast raymarching functions to render on the CPU in parallel with tensorflow.
#


def rays_aabb_intersection_tf(aabb_min, aabb_max, origins, inv_directions):
    """Intersects rays with axis aligned bounding boxes (AABBs).
    The bounding boxes are represented by their min/max coordinates. Note that
    each ray needs a separate bounding box.
    Args:
      aabb_min: A tf.tensor [..., 3] containing the coordinate of the corner
        closest to the origin for each AABB.
      aabb_max: A tf.tensor [...,3] containing the coordinate of the corner
        furthest from the origin for each AABB.
      origins: A tf.tensor [..., 3] of the ray origins.
      inv_directions: A tf.tensor [..., 3] of ray directions. However, each
        channel has been inverted, i.e. (1/dx, 1/dy, 1/dz).
    Returns:
      t_min: A [...] tensor containing the smallest (signed) distance along the
        rays to the closest intersection point with each AABB.
      t_max: A [...] tensor containing the largest (signed) distance along the
        rays to the furthest intersection point with each AABB. Note that if
        t_max < t_min, the ray does not intersect the AABB.
    """
    t1 = (aabb_min - origins) * inv_directions
    t2 = (aabb_max - origins) * inv_directions
    t_min = tf.math.reduce_max(tf.minimum(t1, t2), axis=-1)
    t_max = tf.math.reduce_min(tf.maximum(t1, t2), axis=-1)
    return t_min, t_max


@tf.function
def atlas_raymarch_rays_tf(
    origins,
    directions,
    atlas_t,
    atlas_block_indices_t,
    atlas_params,
    scene_params,
    grid_params,
):
    """Ray marches rays through a SNeRG scene and returns accumulated RGBA.
    Args:
      origins: A tf.tensor [N, 3] of the ray origins.
      directions: A tf.tensor [N, 3] of ray directions.
      atlas_t: A tensorflow tensor  containing the texture atlas.
      atlas_block_indices_t: A tensorflow tensor containing the indirection grid.
      atlas_params: A dict with params for building and rendering with
        the 3D texture atlas.
      scene_params: A dict for scene specific params (bbox, rotation, resolution).
      grid_params: A dict with parameters describing the high-res voxel grid which
        the atlas is representing.
    Returns:
      rgb: An [N, C] tf.tensor with the colors and features accumulated along each
        ray.
      alpha: An [N, 1] tf.tensor with the alpha value accumuated along each ray.
    """

    # Extract the relevant parameters from the dictionaries.
    worldspace_t_opengl = tf.cast(scene_params["worldspace_T_opengl"], dtype=tf.float32)
    min_xyz = tf.cast(scene_params["min_xyz"], dtype=tf.float32)
    voxel_size = tf.cast(grid_params["_voxel_size"], dtype=tf.float32)

    atlas_block_size = atlas_params["atlas_block_size"]
    grid_size = tf.cast(grid_params["_grid_size"], dtype=tf.float32)
    data_block_size = atlas_params["_data_block_size"]
    num_channels = scene_params["_channels"]

    # Set up the rays and transform them to the voxel grid coordinate space.
    num_rays = origins.shape[0]
    opengl_t_worldspace = tf.linalg.inv(worldspace_t_opengl)

    origins_hom = tf.concat([origins, tf.ones_like(origins[Ellipsis, 0:1])], axis=-1)
    origins_hom = tf.reshape(origins_hom, (-1, 4))
    origins_hom = tf.matmul(origins_hom, opengl_t_worldspace)
    origins_opengl = tf.reshape(origins_hom[Ellipsis, 0:3], origins.shape)

    directions_norm = tf.linalg.norm(directions, axis=-1, keepdims=True)
    directions /= directions_norm
    directions_opengl = tf.matmul(directions, opengl_t_worldspace[0:3, 0:3])

    origins_grid = (origins_opengl - min_xyz) / voxel_size
    directions_grid = directions_opengl
    inv_directions_grid = 1.0 / directions_grid

    # Now set the near and far distance of each ray to match the cube which
    # the voxel grid is defined in.
    min_distances, max_distances = rays_aabb_intersection_tf(
        tf.zeros_like(min_xyz),
        tf.cast(grid_size, dtype=tf.float32),
        origins_grid,
        inv_directions_grid,
    )

    invalid_mask = min_distances > max_distances
    zero_fill = tf.zeros_like(min_distances)
    min_distances = tf.where(invalid_mask, zero_fill, min_distances)
    max_distances = tf.where(invalid_mask, zero_fill, max_distances)

    # The NeRF near/far bounds have been set for unnormalized ray directions, so
    # we need to scale our bounds here to compensate for normalizing.
    near_in_voxels = directions_norm[Ellipsis, 0] * scene_params["near"] / voxel_size
    far_in_voxels = directions_norm[Ellipsis, 0] * scene_params["far"] / voxel_size
    min_distances = tf.maximum(near_in_voxels, min_distances)
    max_distances = tf.maximum(near_in_voxels, max_distances)
    max_distances = tf.minimum(far_in_voxels, max_distances)

    current_distances = tf.expand_dims(min_distances, -1) + 0.5

    # Finally, set up the accumulation buffers we need for ray marching.
    total_rgb = tf.zeros((num_rays, num_channels), dtype=tf.float32)
    total_visibility = tf.ones((num_rays, 1), dtype=tf.float32)

    init_state = (0, current_distances, total_rgb, total_visibility)
    max_num_steps = tf.math.ceil(tf.linalg.norm(grid_size))

    def raymarch_condition(i, current_distances, _, total_visibility):
        """Proceed until each ray is fully opaque or has left the voxel grid.
        Args:
          i: A integer containing the iteration count.
          current_distances: A [N, 1] tensor containing the t values for each ray.
          _: Dummy parameter for the accumuated RGB color along each ray.
          total_visibility: A [N, 1] tensor with the  accumulated visibility
            (1 - alpha) along each ray.
        Returns:
          False if all of the rays have finished ray marching (exited the scene,
          or saturated alpha). Also returns false if a maximum iteration count
          has been reached (computed as the diagonal of the dense high-res voxel
          grid).
        """
        visibility_mask = total_visibility >= 1.0 / 256.0
        distance_mask = current_distances < tf.expand_dims(max_distances, -1)
        active_mask = tf.math.logical_and(visibility_mask, distance_mask)[Ellipsis, 0]
        return (
            tf.cast(i, tf.float32) < max_num_steps
            and tf.reduce_max(tf.cast(active_mask, dtype=tf.float32)) > 0.0
        )

    def raymarch_body(i, current_distances, total_rgb, total_visibility):
        """Performs a single ray marching step for every ray.
        This is the main body of the ray marching loop factored into a nested
        function, which allows it to be called iteratively inside a TF graph.
        Args:
          i: A integer containing the iteration count.
          current_distances: A [N, 1] tensor containing the t values for each ray.
          total_rgb: Dummy parameter for the accumuated RGB color along each ray.
          total_visibility: A [N, 1] tensor with the  accumulated visibility
            (1 - alpha) along each ray.
        Returns:
          A tuple containing the updated parameters (i, current_distances,
          total_rgb, total_visibility) after the ray marching step.
        """
        positions_grid = origins_grid + tf.math.multiply(
            directions_grid, current_distances
        )
        positions_atlas_grid = tf.cast(
            tf.floor(positions_grid / data_block_size), dtype=tf.int64
        )

        # Speedup: Only process rays that are inside the voxel grid.
        epsilon = 0.1
        valid_mask = (
            positions_grid[Ellipsis, 0] < grid_size[Ellipsis, 0] - 0.5 - epsilon
        )
        valid_mask = tf.math.logical_and(
            valid_mask,
            positions_grid[Ellipsis, 1] < grid_size[Ellipsis, 1] - 0.5 - epsilon,
        )
        valid_mask = tf.math.logical_and(
            valid_mask,
            positions_grid[Ellipsis, 2] < grid_size[Ellipsis, 2] - 0.5 - epsilon,
        )
        valid_mask = tf.math.logical_and(
            valid_mask, positions_grid[Ellipsis, 0] > 0.5 + epsilon
        )
        valid_mask = tf.math.logical_and(
            valid_mask, positions_grid[Ellipsis, 1] > 0.5 + epsilon
        )
        valid_mask = tf.math.logical_and(
            valid_mask, positions_grid[Ellipsis, 2] > 0.5 + epsilon
        )
        invalid_mask = tf.math.logical_not(valid_mask)

        # Fetch the atlas indices from the indirection grid.
        positions_atlas_grid = tf.where(
            tf.expand_dims(invalid_mask, -1),
            tf.zeros_like(positions_atlas_grid),
            positions_atlas_grid,
        )
        block_indices = tf.gather_nd(atlas_block_indices_t, positions_atlas_grid)
        empty_atlas_mask = block_indices[Ellipsis, 0] < 0

        # Compute where each ray intersects the current macroblock.
        min_aabb_positions = tf.cast(
            positions_atlas_grid * data_block_size, dtype=tf.float32
        )
        max_aabb_positions = min_aabb_positions + data_block_size
        _, max_distance_to_aabb = rays_aabb_intersection_tf(
            min_aabb_positions, max_aabb_positions, origins_grid, inv_directions_grid
        )

        # And then skip past empty macroblocks.
        skip_ahead_mask = tf.math.logical_and(empty_atlas_mask, valid_mask)
        skip_ahead_delta = tf.expand_dims(max_distance_to_aabb, -1) - current_distances
        skip_ahead_delta = tf.where(
            tf.expand_dims(tf.logical_not(skip_ahead_mask), -1),
            tf.zeros_like(skip_ahead_delta),
            skip_ahead_delta,
        )

        current_distances += skip_ahead_delta
        current_distances += 1.0

        # Early out if all rays are outside the voxel grid.
        if tf.reduce_max(tf.cast(valid_mask, dtype=tf.float32)) == 0.0:
            return (i + 1, current_distances, total_rgb, total_visibility)

        # For the rays that are 1) inside the voxel grid and 2) inside a non-empty
        # macroblock, we fetch RGB, Features and alpha from the texture atlas.
        block_indices = tf.where(
            tf.expand_dims(empty_atlas_mask, -1),
            tf.zeros_like(block_indices),
            block_indices,
        )
        block_indices = tf.where(
            tf.expand_dims(invalid_mask, -1),
            tf.zeros_like(block_indices),
            block_indices,
        )

        positions_atlas = positions_grid - min_aabb_positions
        positions_atlas += tf.cast(block_indices * atlas_block_size, dtype=tf.float32)
        positions_atlas += 1.0  # Account for the one-voxel padding in the atlas.

        positions_atlas -= 0.5
        rgb = tf.zeros((num_rays, num_channels), dtype=tf.float32)
        alpha = tf.zeros((num_rays, 1), dtype=tf.float32)

        # Use trilinear interpolation for smoother results.
        offsets_xyz = [
            (x, y, z)  # pylint: disable=g-complex-comprehension
            for x in [0.0, 1.0]
            for y in [0.0, 1.0]
            for z in [0.0, 1.0]
        ]
        for dx, dy, dz in offsets_xyz:
            offset_t = tf.convert_to_tensor([dx, dy, dz], dtype=tf.float32)
            atlas_indices = tf.cast(
                tf.floor(positions_atlas + offset_t), dtype=tf.int64
            )
            weights = 1 - tf.abs(
                tf.cast(atlas_indices, dtype=tf.float32) - positions_atlas
            )
            weights = tf.reshape(tf.math.reduce_prod(weights, axis=-1), alpha.shape)

            atlas_indices = tf.where(
                tf.expand_dims(invalid_mask, -1),
                tf.zeros_like(atlas_indices),
                atlas_indices,
            )

            gathered_features = tf.cast(
                tf.gather_nd(atlas_t, atlas_indices), tf.float32
            )
            trilinear_features = gathered_features[Ellipsis, 0:num_channels]
            trilinear_alpha = gathered_features[
                Ellipsis, num_channels : num_channels + 1
            ]

            rgb += weights * tf.reshape(trilinear_features, rgb.shape)
            alpha += weights * tf.reshape(trilinear_alpha, alpha.shape)

        # Finally, pad use dummy values for rays that were outside the voxel grid
        # or inside empty macroblocks.
        alpha = tf.where(tf.expand_dims(invalid_mask, -1), tf.zeros_like(alpha), alpha)
        alpha = tf.where(
            tf.expand_dims(empty_atlas_mask, -1), tf.zeros_like(alpha), alpha
        )
        rgb = tf.where(tf.expand_dims(invalid_mask, -1), tf.zeros_like(rgb), rgb)
        rgb = tf.where(tf.expand_dims(empty_atlas_mask, -1), tf.zeros_like(rgb), rgb)

        # Accumulate RGB, features and alpha, and loop again.
        total_rgb += total_visibility * rgb
        total_visibility *= 1.0 - alpha

        return (i + 1, current_distances, total_rgb, total_visibility)

    _, current_distances, total_rgb, total_visibility = tf.while_loop(
        raymarch_condition, raymarch_body, init_state, parallel_iterations=1
    )

    return total_rgb, 1.0 - total_visibility


@tf.function
def map_fn_wrapper(origins_t, directions_t, extra_info):
    # unpack the parameters needed for raymarching
    (
        atlas_t,
        atlas_block_indices_t,
        atlas_params,
        scene_params,
        grid_params,
    ) = extra_info

    def partial_raymarch_fn(rays_t):
        origins_t, directions_t = rays_t
        return atlas_raymarch_rays_tf(
            origins_t,
            directions_t,
            atlas_t,
            atlas_block_indices_t,
            atlas_params,
            scene_params,
            grid_params,
        )

    return tf.map_fn(
        partial_raymarch_fn, (origins_t, directions_t), parallel_iterations=32
    )


def atlas_raymarch_rays_parallel_tf(
    h,
    w,
    origins,
    directions,
    atlas_t,
    atlas_block_indices_t,
    atlas_params,
    scene_params,
    grid_params,
):
    """Block-parallel ray marching through a SNeRG scene for an image.
    While atlas_raymarch_rays_tf does vectorized ray marching, we can still
    gain more performance by distributing the rays over CPU cores. This function
    achieves this by splitting the image into square blocks, and ray marching
    each block in parallel.
    Args:
      h: The image height (pixels).
      w: The image width (pixels).
      origins: A tf.tensor [..., 3] of the ray origins.
      directions: A tf.tensor [.., 3] of ray directions.
      atlas_t: A tensorflow tensor  containing the texture atlas.
      atlas_block_indices_t: A tensorflow tensor containing the indirection grid.
      atlas_params: A dict with params for building and rendering with
        the 3D texture atlas.
      scene_params: A dict for scene specific params (bbox, rotation, resolution).
      grid_params: A dict with parameters describing the high-res voxel grid which
        the atlas is representing.
    Returns:
      rgb: A [h, w, C] np.array with the colors and features accumulated for each
        pixel.
      alpha: A [h, w, 1 ] np.array with the alpha value accumuated for each pixel.
    """
    num_channels = scene_params["_channels"]
    block_size = 32

    block_coordinates_list = []
    block_origins_list = []
    block_directions_list = []

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            max_y = min(y + block_size, h)
            max_x = min(x + block_size, w)
            block_coordinates_list.append((y, max_y, x, max_x))

            block_origins = np.zeros((block_size, block_size, 3), dtype=np.float32)
            block_origins[0 : max_y - y, 0 : max_x - x] = origins[
                :, y:max_y, x:max_x, :
            ]
            block_origins_list.append(
                tf.convert_to_tensor(block_origins.reshape(-1, 3))
            )

            block_directions = np.zeros((block_size, block_size, 3), dtype=np.float32)
            block_directions[0 : max_y - y, 0 : max_x - x] = directions[
                :, y:max_y, x:max_x, :
            ]
            block_directions_list.append(
                tf.convert_to_tensor(block_directions.reshape(-1, 3))
            )

    block_origins_t = tf.stack(block_origins_list)
    block_directions_t = tf.stack(block_directions_list)

    # pack the parameters needed for raymarching
    extra_info = (
        atlas_t,
        atlas_block_indices_t,
        atlas_params,
        scene_params,
        grid_params,
    )
    block_rgb_t, block_alpha_t = map_fn_wrapper(
        block_origins_t, block_directions_t, extra_info
    )
    block_rgb_list = list(block_rgb_t)
    block_alpha_list = list(block_alpha_t)

    rgb = np.zeros((h, w, num_channels), np.float32)
    alpha = np.zeros((h, w, 1), np.float32)

    for block_coordinates, block_rgb, block_alpha in zip(
        block_coordinates_list, block_rgb_list, block_alpha_list
    ):
        y, max_y, x, max_x = block_coordinates
        rgb[y:max_y, x:max_x] = block_rgb.numpy().reshape(
            block_size, block_size, num_channels
        )[0 : max_y - y, 0 : max_x - x]
        alpha[y:max_y, x:max_x] = block_alpha.numpy().reshape(
            block_size, block_size, 1
        )[0 : max_y - y, 0 : max_x - x]

    return rgb, alpha


def atlas_raymarch_image_tf(
    h,
    w,
    focal,
    camtoworld,
    atlas_t,
    atlas_block_indices_t,
    atlas_params,
    scene_params,
    grid_params,
):
    """Fast ray marching through a SNeRG scene for an image.
    A convenient wrapper function around atlas_raymarch_rays_parallel_tf.
    Args:
      h: The image height (pixels).
      w: The image width (pixels).
      focal: The image focal length (pixels).
      camtoworld: A numpy array of shape [4, 4] containing the camera-to-world
        transformation matrix for the camera.
      atlas_t: A tensorflow tensor containing the texture atlas.
      atlas_block_indices_t: A tensorflow tensor containing the indirection grid.
      atlas_params: A dict with params for building and rendering with
        the 3D texture atlas.
      scene_params: A dict for scene specific params (bbox, rotation, resolution).
      grid_params: A dict with parameters describing the high-res voxel grid which
        the atlas is representing.
    Returns:
      rgb: A [h, w, C] np.array with the colors and features accumulated for each
        pixel.
      alpha: A [h, w, 1 ] np.array with the alpha value accumuated for each pixel.
    """
    origins, directions, _ = datasets.rays_from_camera(
        scene_params["_use_pixel_centers"], h, w, focal, np.expand_dims(camtoworld, 0)
    )
    if scene_params["ndc"]:
        origins, directions = datasets.convert_to_ndc(origins, directions, focal, w, h)

    return atlas_raymarch_rays_parallel_tf(
        h,
        w,
        origins,
        directions,
        atlas_t,
        atlas_block_indices_t,
        atlas_params,
        scene_params,
        grid_params,
    )


def post_process_render(
    viewdir_mlp, viewdir_mlp_params, rgb, alpha, h, w, focal, camtoworld, scene_params
):
    """Post-processes a SNeRG render (background, view-dependence MLP).
    Composites the render onto the desired background color, then evaluates
    the view-dependence MLP for each pixel, and adds the specular residual.
    Args:
      viewdir_mlp: A nerf.model_utils.MLP that predicts the per-ray view-dependent
        residual color.
      viewdir_mlp_params: A dict containing the MLP parameters for the per-sample
        view-dependence MLP.
      rgb: A [H, W, 7] tensor containing the RGB and features accumulated at each
        pixel.
      alpha: A [H, W, 1] tensor containing the alpha accumulated at each pixel.
      h: The image height (pixels).
      w: The image width (pixels).
      focal: The image focal length (pixels).
      camtoworld: A numpy array of shape [4, 4] containing the camera-to-world
        transformation matrix for the camera.
      scene_params: A dict for scene specific params (bbox, rotation, resolution).
    Returns:
      A list containing post-processed images in the following order:
      the final output image (output_rgb), the alpha channel (alpha), the
      diffuse-only rgb image (rgb), the accumulated feature channels (features),
      and the specular residual from the view-dependence MLP (residual).
    """
    if scene_params["white_bkgd"]:
        rgb[Ellipsis, 0:3] = (
            np.ones_like(rgb[Ellipsis, 0:3]) * (1.0 - alpha) + rgb[Ellipsis, 0:3]
        )

    features = rgb[Ellipsis, 3 : scene_params["_channels"]]
    rgb = rgb[Ellipsis, 0:3]

    rgb_features = np.concatenate([rgb, features], -1)
    _, _, viewdirs = datasets.rays_from_camera(
        scene_params["_use_pixel_centers"], h, w, focal, np.expand_dims(camtoworld, 0)
    )
    viewdirs = viewdirs.reshape((rgb.shape[0], rgb.shape[1], 3))

    residual = model_utils.viewdir_fn(
        viewdir_mlp, viewdir_mlp_params, rgb_features, viewdirs, scene_params
    )
    output_rgb = np.minimum(1.0, rgb + residual)

    return_list = [output_rgb, alpha, rgb, features, residual]

    return return_list

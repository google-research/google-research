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

"""Depth visualization utilities."""

from matplotlib import cm
import numpy as np


def clip_depth_maps(
    stacked_depth_maps, lower_percentile=5, upper_percentile=95
):
  """Clip a collection of stacked depth maps using the given percentiles.

  Args:
    stacked_depth_maps: A numpy array representing a float-valued depth map.
    lower_percentile: The lower percentile of depths to keep. Defaults to 5 e.g.
      the 95 percentile of all depths. Depths below this will be clipped.
    upper_percentile: The upper percentile of depths to keep. Defaults to 95
      e.g. the 95 percentile of all depths. Depths above this will be clipped.

  Returns:
    Clipped depth maps, the same shape as stacked_depth_maps.
  """
  lower_depth_limit, upper_depth_limit = np.percentile(
      stacked_depth_maps, [lower_percentile, upper_percentile]
  )
  return np.clip(stacked_depth_maps, lower_depth_limit, upper_depth_limit)


def grayscale_depth_map(depth_map):
  """Turns a float-valued depth map into a grayscale depth map on [0, 255].

  Args:
    depth_map: A depth map with float values per pixel. [n, h, w, 1]

  Returns:
    An 8-bit grayscale depth map of the same width/height as the input depth
    map.
  """
  assert depth_map.ndim == 4
  min_depth = np.min(depth_map, axis=(-1, -2, -3), keepdims=True)
  max_depth = np.max(depth_map, axis=(-1, -2, -3), keepdims=True)

  # Project the depth from [min_depth, max_depth] to [0,255].
  # With nearby objects being white and far away objects being black.
  scaled_depth = (1 - (depth_map - min_depth) / (max_depth - min_depth)) * 255
  return scaled_depth


def colorize_depth_map(depth_map, min_value=0, max_value=255):
  """Turns a grayscale depth map into a colorized one for visualization.

  Args:
    depth_map: shape [n, h, w, 1].
    min_value: The min value of the depth map. Defaults to 0.
    max_value: The max value of the depth map. Default to 255.

  Returns:
    colorized_depths: A colorized depth map of the same width/height as the
    input depth map.
  """
  assert depth_map.ndim == 4
  depth_map = clip_depth_maps(depth_map)
  depth_map = grayscale_depth_map(depth_map)
  clipped_depths = np.clip(depth_map, min_value, max_value)
  scaled_depths = (clipped_depths - min_value) / (max_value - min_value)

  color_map = cm.get_cmap('turbo')
  colorized_depths = color_map(scaled_depths[Ellipsis, 0])[Ellipsis, :3]
  return colorized_depths

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
"""TensorFlow ops for the structured video representation model."""

import enum
import tensorflow.compat.v1 as tf

EPSILON = 1e-6  # Constant for numerical stability.


class Axis(enum.Enum):
  """Maps axes to image indices, assuming that 0th dimension is the batch."""
  y = 1
  x = 2


def maps_to_keypoints(heatmaps):
  """Turns feature-detector heatmaps into (x, y, scale) keypoints.

  This function takes a tensor of feature maps as input. Each map is normalized
  to a probability distribution and the location of the mean of the distribution
  (in image coordinates) is computed. This location is used as a low-dimensional
  representation of the heatmap (i.e. a keypoint).

  To model keypoint presence/absence, the mean intensity of each feature map is
  also computed, so that each keypoint is represented by an (x, y, scale)
  triplet.

  Args:
    heatmaps: [batch_size, H, W, num_keypoints] tensors.
  Returns:
    A [batch_size, num_keypoints, 3] tensor with (x, y, scale)-triplets for each
    keypoint. Coordinate range is [-1, 1] for x and y, and [0, 1] for scale.
  """

  # Check that maps are non-negative:
  map_min = tf.reduce_min(heatmaps)
  assert_nonneg = tf.Assert(tf.greater_equal(map_min, 0.0), [map_min])
  with tf.control_dependencies([assert_nonneg]):
    heatmaps = tf.identity(heatmaps)

  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)
  map_scales = tf.reduce_mean(heatmaps, axis=[1, 2])

  # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
  # degeneracy between the encoder and decoder heatmap scales and ensures that
  # the scales are in a reasonable range for the RNN:
  map_scales /= (EPSILON + tf.reduce_max(map_scales, axis=-1, keepdims=True))

  return tf.stack([x_coordinates, y_coordinates, map_scales], axis=-1)


def _maps_to_coordinates(maps, axis):
  """Reduces heatmaps to coordinates along one axis (x or y).

  Args:
    maps: [batch_size, H, W, num_keypoints] tensors.
    axis: Axis Enum.

  Returns:
    A [batch_size, num_keypoints, 2] tensor with (x, y)-coordinates.
  """

  width = maps.get_shape()[axis.value]
  grid = _get_pixel_grid(axis, width)
  shape = [1, 1, 1, 1]
  shape[axis.value] = -1
  grid = tf.reshape(grid, shape)

  if axis == Axis.x:
    marginalize_dim = 1
  elif axis == Axis.y:
    marginalize_dim = 2

  # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
  weights = tf.reduce_sum(maps + EPSILON, axis=marginalize_dim, keep_dims=True)
  weights /= tf.reduce_sum(weights, axis=axis.value, keep_dims=True)

  # Compute the center of mass of the marginalized maps to obtain scalar
  # coordinates:
  coordinates = tf.reduce_sum(weights * grid, axis=axis.value, keep_dims=True)

  return tf.squeeze(coordinates, axis=[1, 2])


def keypoints_to_maps(keypoints, sigma=1.0, heatmap_width=16):
  """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).

  Args:
    keypoints: [batch_size, num_keypoints, 3] tensor of keypoints where the last
      dimension contains (x, y, scale) triplets.
    sigma: Std. dev. of the Gaussian blob, in units of heatmap pixels.
    heatmap_width: Width of output heatmaps in pixels.

  Returns:
    A [batch_size, heatmap_width, heatmap_width, num_keypoints] tensor.
  """

  coordinates, map_scales = tf.split(keypoints, [2, 1], axis=-1)

  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return tf.reshape(grid, shape)

  # Expand to [batch_size, 1, 1, num_keypoints] for broadcasting later:
  x_coordinates = coordinates[:, tf.newaxis, tf.newaxis, :, 0]
  y_coordinates = coordinates[:, tf.newaxis, tf.newaxis, :, 1]

  # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
  sigma = tf.cast(sigma, tf.float32)
  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0
  x_vec = tf.exp(-tf.square(get_grid(Axis.x) - x_coordinates)/keypoint_width)
  y_vec = tf.exp(-tf.square(get_grid(Axis.y) - y_coordinates)/keypoint_width)
  maps = tf.multiply(x_vec, y_vec)

  return maps * map_scales[:, tf.newaxis, tf.newaxis, :, 0]


def _get_pixel_grid(axis, width):
  """Returns an array of length `width` containing pixel coordinates."""
  if axis == Axis.x:
    return tf.linspace(-1.0, 1.0, width)  # Left is negative, right is positive.
  elif axis == Axis.y:
    return tf.linspace(1.0, -1.0, width)  # Top is positive, bottom is negative.


def add_coord_channels(image_tensor):
  """Adds channels containing pixel indices (x and y coordinates) to an image.

  Note: This has nothing to do with keypoint coordinates. It is just a data
  augmentation to allow convolutional networks to learn non-translation-
  equivariant outputs. This is similar to the "CoordConv" layers:
  https://arxiv.org/abs/1603.09382.

  Args:
    image_tensor: [batch_size, H, W, C] tensor.

  Returns:
    [batch_size, H, W, C + 2] tensor with x and y coordinate channels.
  """

  batch_size = tf.shape(image_tensor)[0]
  x_size = tf.shape(image_tensor)[2]
  y_size = tf.shape(image_tensor)[1]

  x_grid = tf.lin_space(-1.0, 1.0, x_size)
  x_map = tf.tile(
      x_grid[tf.newaxis, tf.newaxis, :, tf.newaxis], (batch_size, y_size, 1, 1))

  y_grid = tf.lin_space(1.0, -1.0, y_size)
  y_map = tf.tile(
      y_grid[tf.newaxis, :, tf.newaxis, tf.newaxis], (batch_size, 1, x_size, 1))

  return tf.concat([image_tensor, x_map, y_map], axis=-1)

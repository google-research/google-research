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

"""Library that defines a panoramic transformer which rotates the world.

The panoramic transformer performs a yaw rotation about the camera's z-axis
by translating an equirectangular representation. We assume a camera convention
of z-axis up.

Below is a top-down view of the camera coordinate system. The +z-axis points
into this viewpoint. A positive yaw rotation corresponds to a CCW rotation about
the z-axis when looking down.

    y
    ^
    |
    |
    +-----> x

One thing to note is that a CCW rotation of the camera is equivalent to a
CW rotation of the scene. Therefore rotating the scene by an offset phi
is equivalent to rotating the camera by -phi.

Usage:
  pano_image = ... # a pano or a representation derived from a pano
  phi = ...        # a differentiable estimate of a yaw rotation that is
                   # predicted from pano_image. Units in radians.

  # Rotating the scene by phi is equivalent to rotating the camera by -phi.
  rotation_normalized_pano_image = shift_pano_by_rotation(pano_image, -phi)
  rotation_normalized_output = network(rotation_normalized_pano_image)

  # Optional inverse rotate to get restore pano_image's original orientation.
  output = shift_pano_by_rotation(rotation_normalized_output, alpha_phi)
"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import resampler


def rotate_pano_horizontally(input_feature_map, yaw_angle):
  """Rotates input_feature_map by yaw_angle by horizontally translating pixels.

  The layer is differentiable with respect to yaw_angle and  input_feature_map.
  yaw_angle is positive for CCW rotation about the z-axis where the coordinates
  are constructed with z-axis facing up.

  Args:
    input_feature_map: panoramic image or neural feature maps of shape [B, H, W,
      C].
    yaw_angle: A tensor of shape `[B]` which represents the desired rotation of
      input_feature_map. yaw_angle is in units of radians. A positive yaw_angle
      rolls pixels left.

  Returns:
    A rotated feature map with dimensions `[B, H, W, C]`

  Reference:
  [1]: 'Spatial Transformer Networks', Jaderberg et. al,
       (https://arxiv.org/abs/1506.02025)
  """

  # Number of input dimensions.
  tfshape = tf.shape(input_feature_map)
  batch_size = tfshape[0]
  height = tfshape[1]
  width = tfshape[2]

  float32_width = tf.cast(width, dtype=tf.float32)
  float32_height = tf.cast(height, dtype=tf.float32)

  x_offset = (yaw_angle / 2 / np.pi) * float32_width

  x_grid = tf.linspace(0., float32_width - 1, width)  # (W)
  # 0.5 * original_image_width to match the convention described in comment
  x_pixel_coord = x_grid[tf.newaxis] + x_offset[:, tf.newaxis]  # (B, W)

  x_pixel_coord = tf.tile(x_pixel_coord[:, tf.newaxis, :],
                          [1, height, 1])  # (B, H, W)
  y_pixel_coord = tf.linspace(0., float32_height - 1,
                              height)[tf.newaxis, :, tf.newaxis]  # (1, H, 1)
  y_pixel_coord = tf.tile(y_pixel_coord, [batch_size, 1, width])
  wrapped_x_pixel_coord = tf.floormod(x_pixel_coord, float32_width)

  # Because these are panoramas, we can concatenate the first column to the
  # right side. This allows us to interpolate values for coordinates that
  # correspond to pixels that connects the left and right edges of the
  # panorama.
  input_feature_map = tf.concat(
      [input_feature_map, input_feature_map[:, :, :1]], axis=2)

  return resampler.resampler(
      input_feature_map,
      tf.stack([wrapped_x_pixel_coord, y_pixel_coord], axis=-1))

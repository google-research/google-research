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

"""Spherical geometric functions.

We use half-integer pixel centre convention, e.g. (0.5, 0.5) indicates the top
left pixel.
The colatitude of the equirectangular grid ranges from 0 (the top row) to
pi (the bottom), not inclusively (the first row starts from pixel_height/2 and
the last row ends with pi-pixel_height/2 where pixel_height = pi/image_height).
The azimuth angle goes from 0 (the leftmost column) to 2*pi (the rightmost
column), not inclusively (the first column starts from pixel_width/2 and the
last column ends with 2*pi-pixel_width/2 where pixel_width = 2*pi/image_width).
"""
import math
import tensorflow.compat.v1 as tf
import tensorflow_graphics.math.math_helpers as tfg_math_helpers


def cartesian_to_equirectangular_coordinates(v, shape):
  """Convert Cartesian coordinates to pixel locations on equirectangular images.

  Args:
    v: [A1, A2, ..., An, 3] 3d Cartesian coordinates.
    shape: 2-d list represents the shape of an equirectangular image.

  Returns:
    [A1, A2, ..., An, 2] pixel coordinates on an equirectangular image.
  """
  height, width = shape[0], shape[1]
  axis_convert = tf.constant([[1., 0., 0.],
                              [0., 0., -1.],
                              [0., 1., 0.]])
  v = tf.squeeze(tf.matmul(axis_convert, tf.expand_dims(v, -1)))
  colatitude, azimuth = tf.split(cartesian_to_spherical(v), [1, 1], -1)
  x = width * (azimuth % (2 * math.pi)) / (2 * math.pi)
  y = height * (colatitude / math.pi)
  return tf.concat([x, y], -1)


def equirectangular_coordinates_to_cartesian(p, shape):
  """Convert pixel locations on equirectangular images to Cartesian coordinates.

  Args:
    p: [A1, A2, ..., An, 2] pixel coordinates on an equirectangular image.
    shape: 2-d list represents the shape of an equirectangular image.

  Returns:
    [A1, A2, ..., An, 3] 3d Cartesian coordinates on sphere.
  """
  height, width = shape[0], shape[1]
  x, y = tf.split(p, [1, 1], -1)
  azimuth = x * (2 * math.pi) / width
  colatitude = math.pi * (y / height)
  spherical = tf.concat([colatitude, azimuth], -1)
  cartesian = spherical_to_cartesian(spherical)
  axis_convert = tf.constant([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
  cartesian = tf.squeeze(tf.matmul(
      axis_convert, tf.expand_dims(cartesian, -1), transpose_a=True), -1)
  return cartesian


def generate_cartesian_grid(resolution, fov):
  """Get (x, y, z) coordinates of all pixel centres in the image.

  The image plane lies at z=-1 and the image center is (0, 0, -1).
  Args:
    resolution: a 2-D list containing the resolution (height, width)
                of the desired output.
    fov: (float) camera's horizontal field of view in degrees.

  Returns:
    3-D tensor of shape `[HEIGHT, WIDTH, 3]`

  Raises:
    ValueError: 'resolution' is not valid.
  """
  with tf.name_scope(None, 'generate_cartesian_grid', [resolution, fov]):
    if not isinstance(resolution, list) or len(resolution) != 2:
      raise ValueError("'resolution' is not valid.")

    fov = fov / 180 * math.pi
    width = 2 * tf.tan(fov / 2)
    height = width * resolution[0] / resolution[1]
    pixel_size = width / resolution[1]
    x_range = width-pixel_size
    y_range = height-pixel_size
    # x increases from left to right while y increases from bottom to top.
    # Use half-integer pixel centre convention, and generate the coordinates
    # for the centres of the pixels.
    # For example, a 2x3 grid with pixel_size=1 (height=2, width=3) should have
    # [(-1.0,  0.5), (0.0,  0.5), (1.0,  0.5),
    #  (-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)]
    xx, yy = tf.meshgrid(
        tf.lin_space(-x_range / 2, x_range / 2, resolution[1]),
        tf.lin_space(y_range / 2, -y_range / 2, resolution[0]))
    grid = tf.stack([xx, yy, -tf.ones_like(xx)], axis=-1)
    return grid


def generate_equirectangular_grid(shape):
  """Get spherical coordinates of an equirectangular grid.

  Args:
    shape: a list represents the (height, width) of the output.

  Returns:
    3-D tensor of shape `[HEIGHT, WIDTH, 2]`

  Raises:
    ValueError: 'resolution' is not valid.
  """
  with tf.name_scope(None, 'generate_equirectangular_grid', [shape]):
    if not isinstance(shape, list) or len(shape) != 2:
      raise ValueError("'shape' is not valid.")

    height, width = shape[0], shape[1]
    pixel_w = 2 * math.pi / float(width)
    pixel_h = math.pi / float(height)
    azimuth, colatitude = tf.meshgrid(
        tf.lin_space(
            pixel_w / 2, 2 * math.pi - pixel_w / 2, width),
        tf.lin_space(
            pixel_h / 2, math.pi - pixel_h / 2, height))
    return tf.stack([colatitude, azimuth], axis=-1)


def spherical_to_cartesian(spherical):
  """Convert spherical coordinates to Cartesian coordinates.

  Args:
    spherical: [..., 2] tensor containing (colatitude, azimuth).

  Returns:
    a Tensor with shape [..., 3] containing (x, y, z).
  """
  colatitude, azimuth = tf.split(spherical, [1, 1], -1)
  return tfg_math_helpers.spherical_to_cartesian_coordinates(
      tf.concat([tf.ones_like(colatitude), colatitude, azimuth], -1))


def cartesian_to_spherical(cartesian):
  """Convert Cartesian coordinates to spherical coordinates.

  Args:
    cartesian: [..., 3] tensor containing (x, y, z).

  Returns:
    a Tensor with shape [..., 2] containing (colatitude, azimuth).
  """
  _, spherical = tf.split(
      tfg_math_helpers.cartesian_to_spherical_coordinates(cartesian),
      [1, 2], -1)
  return spherical


def equirectangular_padding(images, num_paddings):
  """Pad equirectangular panorama images.

  Args:
    images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
    num_paddings: a 2x2 integer list [[n_top, n_bottom], [n_left, n_right]]
      representing the number of rows or columns to pad around the images.

  Returns:
    a 4-D tensor representing padded images with a shape of
    `[BATCH, n_top+HEIGHT+n_bottom, n_left+WIDTH+n_right, CHANNELS]`.

  Raises:
    ValueError: 'images' has the wrong dimensions.
                The number of paddings exceeds the height or width dimension.
  """
  with tf.name_scope(None, 'equirectangular_padding', [images, num_paddings]):
    if len(images.shape) != 4:
      raise ValueError("'images' has the wrong dimensions.")

    shape = images.shape.as_list()
    height, width = shape[1], shape[2]
    top, down = num_paddings[0][0], num_paddings[0][1]
    left, right = num_paddings[1][0], num_paddings[1][1]
    if top > height or down > height:
      raise ValueError('The number of paddings exceeds the height dimension.')
    if left > width or right > width:
      raise ValueError('The number of paddings exceeds the width dimension.')

    semicircle = tf.cast(width/2, tf.int32)
    # The padded rows should be symmetric to 'images', but they should be
    # shifted by 180 degrees. Copy the rightmost column (2*pi-w) as the padded
    # colomn on the left and copy the leftmost column (0+w) as the padded colomn
    # on the right.
    top_padding = tf.reverse(
        tf.roll(images[:, :top, :, :], axis=2, shift=semicircle), axis=[1])
    bottom_padding = tf.roll(
        tf.reverse(images, axis=[1])[:, :down, :, :], axis=2, shift=semicircle)
    padded_images = tf.concat([top_padding, images, bottom_padding], 1)
    left_padding = tf.reverse(
        tf.reverse(padded_images, axis=[2])[:, :, :left, :], axis=[2])
    right_padding = padded_images[:, :, :right, :]
    padded_images = tf.concat([left_padding, padded_images, right_padding], 2)
    return padded_images

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

"""Transformations for equirectangular and perspective images.

The coordinate system is the same as OpenGL's, where -Z is the camera looking
direction, +Y points up and +X points right.
Rotations are applied as pre-multiplication in all cases.
"""
import math

from pano_utils import geometry
from pano_utils import math_utils
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa


def equirectangular_sampler(images, spherical_coordinates):
  """Sample panorama images using a grid of spherical coordinates.

  Args:
    images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
    spherical_coordinates: a float32 tensor with shape
      [BATCH, sampling_height, sampling_width, 2] representing spherical
      coordinates (colatitude, azimuth) of the sampling grids.

  Returns:
    a 4-D tensor of shape `[BATCH, sampling_height, sampling_width, CHANNELS]`
    representing resampled images.

  Raises:
    ValueError: 'images' or 'spherical_coordinates' has the wrong dimensions.
  """
  with tf.name_scope(
      None, 'equirectangular_sampler', [images, spherical_coordinates]):
    if len(images.shape) != 4:
      raise ValueError("'images' has the wrong dimensions.")
    if spherical_coordinates.shape[-1] != 2:
      raise ValueError("'spherical_coordinates' has the wrong dimensions.")

    shape = images.shape.as_list()
    height, width = shape[1], shape[2]
    padded_images = geometry.equirectangular_padding(images, [[1, 1], [1, 1]])
    colatitude, azimuth = tf.split(spherical_coordinates, [1, 1], -1)
    # The colatitude of the equirectangular image goes from 0 (the top row)
    # to pi (the bottom), not inclusively. The azimuth goes from 0
    # (the leftmost column) to 2*pi (the rightmost column).
    # For example, azimuth-colatitude (0, pi/2) is the mid pixel in the first
    # column of the equirect image.
    # Convert spherical coordinates to equirectangular coordinates on images.
    # +1 in the end because of the padding.
    x_pano = (tf.mod(azimuth / math.pi, 2) * width / 2.0 - 0.5) + 1
    y_pano = ((colatitude / math.pi) * height - 0.5) + 1
    pano_coordinates = tf.concat([x_pano, y_pano], -1)
    remapped = tfa.image.resampler(padded_images, pano_coordinates)
    return remapped


def rectilinear_projection(images,
                           resolution,
                           fov,
                           rotations):
  """Convert equirectangular panoramic images to perspective images.

  First, the panorama images are rotated by the input parameter "rotations".
  Then, the region with the field of view "fov" centered at camera's look-at -Z
  axis is projected into perspective images. The -Z axis corresponds to the
  spherical coordinates (pi/2, pi/2) which is (HEIGHT/2, WIDTH/4) on the pano.

  Args:
    images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
    resolution: a 2-D tuple or list containing the resolution of desired output.
    fov: (float) camera's horizontal field of view in degrees.
    rotations: [BATCH, 3, 3] rotation matrices.

  Returns:
    4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`

  Raises:
    ValueError: 'images' has the wrong dimensions.
    ValueError: 'images' is not a float tensor.
    ValueError: 'rotations' has the wrong dimensions.
  """
  with tf.name_scope(None, 'rectilinear_projection',
                     [images, resolution, fov, rotations]):
    if len(images.shape) != 4:
      raise ValueError("'images' has the wrong dimensions.")

    if images.dtype != tf.float32 and images.dtype != tf.float64:
      raise ValueError("'images' must be a float tensor.")

    if rotations.shape[-2:] != [3, 3]:
      raise ValueError("'rotations' has the wrong dimensions.")

    shape = images.shape.as_list()
    batch = shape[0]

    cartesian_coordinates = geometry.generate_cartesian_grid(resolution, fov)
    # create batch -> [batch, height, width, 3]
    cartesian_coordinates = tf.tile(
        tf.expand_dims(cartesian_coordinates, axis=0), [batch, 1, 1, 1])
    # The rotation matrices have to be [batch, height, width, 3, 3].
    flip_x = tf.constant([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rotations = tf.matmul(flip_x,
                          tf.matmul(rotations, flip_x, transpose_a=True))
    rotated_coordinates = tf.matmul(
        rotations[:, tf.newaxis, tf.newaxis],
        tf.expand_dims(cartesian_coordinates, -1), transpose_a=True)
    axis_convert = tf.constant([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    rotated_coordinates = tf.matmul(axis_convert, rotated_coordinates)
    rotated_coordinates = tf.squeeze(rotated_coordinates, -1)
    spherical_coordinates = geometry.cartesian_to_spherical(rotated_coordinates)
    # The azimuth of 'spherical_coordinates' decreases from left to right but
    # the x should increase from left to right.
    spherical_coordinates = tf.reverse(spherical_coordinates, [2])
    return equirectangular_sampler(images, spherical_coordinates)


def rotate_pano(images, rotations):
  """Rotate Panoramic images.

  Convert the spherical coordinates (colatitude, azimuth) to Cartesian (x, y, z)
  then apply SO(3) rotation matrices. Finally, convert them back to spherical
  coordinates and remap the equirectangular images.
  Note1: The rotations are applied to the sampling sphere instead of the camera.
  The camera actually rotates R^T. I_out(x) = I_in(R * x), x are points in the
  camera frame.

  Note2: It uses a simple linear interpolation for now instead of slerp, so the
  pixel values are not accurate but visually plausible.

  Args:
    images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
    rotations: [BATCH, 3, 3] rotation matrices.

  Returns:
    4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.

  Raises:
    ValueError: if the `images` or 'rotations' has the wrong dimensions.
  """
  with tf.name_scope(None, 'rotate_pano', [images, rotations]):
    if len(images.shape) != 4:
      raise ValueError("'images' has the wrong dimensions.")
    if rotations.shape[-2:] != [3, 3]:
      raise ValueError("'rotations' must have 3x3 dimensions.")

    shape = images.shape.as_list()
    batch, height, width = shape[0], shape[1], shape[2]
    spherical = tf.expand_dims(
        geometry.generate_equirectangular_grid([height, width]), 0)
    spherical = tf.tile(spherical, [batch, 1, 1, 1])
    cartesian = geometry.spherical_to_cartesian(spherical)
    axis_convert = tf.constant([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
    cartesian = tf.matmul(axis_convert, tf.expand_dims(cartesian, -1))
    rotated_cartesian = tf.matmul(
        rotations[:, tf.newaxis, tf.newaxis], cartesian)
    rotated_cartesian = tf.squeeze(
        tf.matmul(axis_convert, rotated_cartesian, transpose_a=True), -1)
    rotated_spherical = geometry.cartesian_to_spherical(rotated_cartesian)
    return equirectangular_sampler(images, rotated_spherical)


def rotate_image_in_3d(images,
                       input_rotations,
                       input_fov,
                       output_fov,
                       output_shape):
  """Return reprojected perspective view images given a rotated camera.

  This function applies a homography H = K_output * R^T * K_input' where
  K_output and K_input are the output and input camera intrinsics, R is the
  rotation from the input images' frame to the target frame.

  Args:
    images: [BATCH, HEIGHT, WIDTH, CHANNEL] perspective view images.
    input_rotations: [BATCH, 3, 3] rotations matrices from current camera frame
      to target camera frame.
    input_fov: [BATCH] a 1-D tensor (float32) of input field of view in degrees.
    output_fov: (float) output field of view in degrees.
    output_shape: a 2-D list of output dimension [height, width].

  Returns:
    reprojected images [BATCH, height, width, CHANNELS].
  """
  with tf.name_scope(
      None, 'rotate_image_in_3d',
      [images, input_rotations, input_fov, output_fov, output_shape]):
    if len(images.shape) != 4:
      raise ValueError("'images' has the wrong dimensions.")
    if input_rotations.shape[-2:] != [3, 3]:
      raise ValueError("'input_rotations' must have 3x3 dimensions.")

    shape = images.shape.as_list()
    batch, height, width = shape[0], shape[1], shape[2]
    cartesian = geometry.generate_cartesian_grid(output_shape, output_fov)
    cartesian = tf.tile(
        cartesian[tf.newaxis, :, :, :, tf.newaxis], [batch, 1, 1, 1, 1])
    input_rotations = tf.tile(input_rotations[:, tf.newaxis, tf.newaxis, :],
                              [1]+output_shape+[1, 1])
    cartesian = tf.squeeze(
        tf.matmul(input_rotations, cartesian, transpose_a=True), -1)
    image_coordinates = -cartesian[:, :, :, :2] / cartesian[:, :, :, -1:]
    x, y = tf.split(image_coordinates, [1, 1], -1)
    w = 2 * tf.tan(math_utils.degrees_to_radians(input_fov / 2))
    h = 2 * tf.tan(math_utils.degrees_to_radians(input_fov / 2))
    w = w[:, tf.newaxis, tf.newaxis, tf.newaxis]
    h = h[:, tf.newaxis, tf.newaxis, tf.newaxis]
    nx = x*width / w + width / 2 - 0.5
    ny = -y * height / h + height / 2 - 0.5
    return tfa.image.resampler(images, tf.concat([nx, ny], -1))


def rotate_image_on_pano(images, rotations, fov, output_shape):
  """Transform perspective images to equirectangular images after rotations.

  Return equirectangular panoramic images in which the input perspective images
  embedded in after the rotation R from the input images' frame to the target
  frame. The image with the field of view "fov" centered at camera's look-at -Z
  axis is projected onto the pano. The -Z axis corresponds to the spherical
  coordinates (pi/2, pi/2) which is (HEIGHT/2, WIDTH/4) on the pano.

  Args:
    images: [BATCH, HEIGHT, WIDTH, CHANNEL] perspective view images.
    rotations: [BATCH, 3, 3] rotations matrices.
    fov: (float) images' field of view in degrees.
    output_shape: a 2-D list of output dimension [height, width].

  Returns:
    equirectangular images [BATCH, height, width, CHANNELS].
  """
  with tf.name_scope(None, 'rotate_image_on_pano',
                     [images, rotations, fov, output_shape]):
    if len(images.shape) != 4:
      raise ValueError("'images' has the wrong dimensions.")
    if rotations.shape[-2:] != [3, 3]:
      raise ValueError("'rotations' must have 3x3 dimensions.")

    shape = images.shape.as_list()
    batch, height, width = shape[0], shape[1], shape[2]
    # Generate a mesh grid on a sphere.
    spherical = geometry.generate_equirectangular_grid(output_shape)
    cartesian = geometry.spherical_to_cartesian(spherical)
    cartesian = tf.tile(
        cartesian[tf.newaxis, :, :, :, tf.newaxis], [batch, 1, 1, 1, 1])
    axis_convert = tf.constant([[0., -1., 0.], [0., 0., 1.], [1., 0., 0.]])
    cartesian = tf.matmul(axis_convert, cartesian)
    cartesian = tf.squeeze(
        tf.matmul(rotations[:, tf.newaxis, tf.newaxis], cartesian), -1)
    # Only take one hemisphere. (camera lookat direction)
    hemisphere_mask = tf.cast(cartesian[:, :, :, -1:] < 0, tf.float32)
    image_coordinates = cartesian[:, :, :, :2] / cartesian[:, :, :, -1:]
    x, y = tf.split(image_coordinates, [1, 1], -1)
    # Map pixels on equirectangular pano to perspective image.
    nx = -x * width / (2 * tf.tan(
        math_utils.degrees_to_radians(fov / 2))) + width / 2 - 0.5
    ny = y * height / (2 * tf.tan(
        math_utils.degrees_to_radians(fov / 2))) + height / 2 - 0.5
    transformed = hemisphere_mask * tfa.image.resampler(
        images, tf.concat([nx, ny], -1))
    return transformed

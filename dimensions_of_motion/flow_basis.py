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

"""Functions to work with flow and flow bases."""
import geometry
import tensorflow as tf


def camera_rotation_basis(height,
                          width,
                          principal_point):
  """Camera motion basis, rotation only, unknown focal length.

  Args:
    height: the image height, H.
    width: the image width, W.
    principal_point: [..., 2] principal points (cx cy).
  Returns:
    rotation_basis: [..., 5, H, W, 2]
  """
  pixel_centers = geometry.pixel_center_grid(height, width)

  cx, cy = tf.unstack(principal_point[Ellipsis, tf.newaxis, tf.newaxis, :], axis=-1)

  # Assume square pixels
  aspect_ratio = width / height

  u = pixel_centers[Ellipsis, 0] - cx
  v = pixel_centers[Ellipsis, 1] - cy

  one = tf.ones_like(u)
  zero = tf.zeros_like(u)

  # Five flow dimensions for camera rotation. These don't depend on disparity.
  # Convention: field corresponds to positive rotation of camera about the axis.
  rotate_z = tf.stack([-v / aspect_ratio, u * aspect_ratio],
                      axis=-1)
  rotate_x_base = tf.stack([zero, one], axis=-1)
  rotate_x_secondary = tf.stack([u * v, v * v], axis=-1)
  rotate_y_base = tf.stack([-one, zero], axis=-1)
  rotate_y_secondary = tf.stack([-u * u, -u * v], axis=-1)

  elements = [rotate_x_base, rotate_x_secondary,
              rotate_y_base, rotate_y_secondary,
              rotate_z]
  rotation = tf.stack(elements, axis=-4)

  # Normalize basis flows
  rotation = rotation / tf.sqrt(
      tf.reduce_sum(tf.square(rotation), axis=(-1, -2, -3), keepdims=True))

  return rotation


def camera_translation_basis(height, width, principal_point, disparity):
  """Camera motion basis, translation only, given disparity.

  Args:
    height: the image height, H.
    width: the image width, W.
    principal_point: [..., 2] principal points (cx cy).
    disparity: [..., H, W, 1] disparity (i.e. inverse depth), or None.
  Returns:
    translation_basis: [..., 3, H, W, 2]
  """
  pixel_centers = geometry.pixel_center_grid(height, width)

  cx, cy = tf.unstack(principal_point[Ellipsis, tf.newaxis, tf.newaxis, :], axis=-1)

  u = pixel_centers[Ellipsis, 0] - cx
  v = pixel_centers[Ellipsis, 1] - cy

  one = tf.ones_like(u)
  zero = tf.zeros_like(u)

  assert height == disparity.shape[-3]
  assert width == disparity.shape[-2]

  # Three flow dimensions for camera translation. These depend on disparity
  # and correspond to positive movement of the camera along the three axes.
  translate_x = tf.stack([-one, zero], axis=-1)
  translate_y = tf.stack([zero, -one], axis=-1)
  translate_z = tf.stack([u, v], axis=-1)

  translation = tf.stack([translate_x, translate_y, translate_z], axis=-4)
  # Scale to be roughly normalized if average disparity is 0.5
  translation = 2 * translation / tf.sqrt(
      tf.reduce_sum(
          tf.square(translation), axis=(-1, -2, -3), keepdims=True))
  translation = translation * disparity[Ellipsis, tf.newaxis, :, :, :]

  return translation


def camera_motion_basis(height, width,
                        principal_point,
                        disparity):
  """Return rotation and translation parts of basis from camera motion.

  Args:
    height: the image height, H.
    width: the image width, W.
    principal_point: [..., 2] principal points (cx cy).
    disparity: [..., H, W, 1] disparity (i.e. inverse depth).
  Returns:
    rotation_basis: [..., 5, H, W, 2]
    translation_basis: [..., 3, H, W, 2]
  """
  rotation = camera_rotation_basis(height, width, principal_point)
  translation = camera_translation_basis(
      height, width, principal_point, disparity)

  return rotation, translation

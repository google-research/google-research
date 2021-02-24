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

"""Utility functions provided for renderer."""
import numpy as np
import tensorflow as tf


def create_vertices_intrinsics(disparity, intrinsics):
  """3D mesh vertices from a given disparity and intrinsics.

  Args:
     disparity: [B, H, W] inverse depth
     intrinsics: [B, 4] reference intrinsics

  Returns:
     [B, L, H*W, 3] vertex coordinates.
  """
  # Focal lengths
  fx = intrinsics[:, 0]
  fy = intrinsics[:, 1]
  fx = fx[Ellipsis, tf.newaxis, tf.newaxis]
  fy = fy[Ellipsis, tf.newaxis, tf.newaxis]

  # Centers
  cx = intrinsics[:, 2]
  cy = intrinsics[:, 3]
  cx = cx[Ellipsis, tf.newaxis]
  cy = cy[Ellipsis, tf.newaxis]

  batch_size, height, width = disparity.shape.as_list()
  vertex_count = height * width

  i, j = tf.meshgrid(tf.range(width), tf.range(height))
  i = tf.cast(i, tf.float32)
  j = tf.cast(j, tf.float32)
  width = tf.cast(width, tf.float32)
  height = tf.cast(height, tf.float32)
  # 0.5 is added to get the position of the pixel centers.
  i = (i + 0.5) / width
  j = (j + 0.5) / height
  i = i[tf.newaxis]
  j = j[tf.newaxis]

  depths = 1.0 / tf.clip_by_value(disparity, 0.01, 1.0)
  mx = depths / fx
  my = depths / fy
  px = (i-cx) * mx
  py = (j-cy) * my

  vertices = tf.stack([px, py, depths], axis=-1)
  vertices = tf.reshape(vertices, (batch_size, vertex_count, 3))
  return vertices


def create_triangles(h, w):
  """Creates mesh triangle indices from a given pixel grid size.

     This function is not and need not be differentiable as triangle indices are
     fixed.

  Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.

  Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
  """
  x, y = np.meshgrid(range(w - 1), range(h - 1))
  tl = y * w + x
  tr = y * w + x + 1
  bl = (y + 1) * w + x
  br = (y + 1) * w + x + 1
  triangles = np.array([tl, bl, tr, br, tr, bl])
  triangles = np.transpose(triangles, (1, 2, 0)).reshape(
      ((w - 1) * (h - 1) * 2, 3))
  return triangles


def perspective_from_intrinsics(intrinsics):
  """Computes a perspective matrix from camera intrinsics.

  The matrix maps camera-space to clip-space (x, y, z, w) where (x/w, y/w, z/w)
  ranges from -1 to 1 in each axis. It's a standard OpenGL-stye perspective
  matrix, except that we use positive Z for the viewing direction (instead of
  negative) so there are sign differences.

  Args:
    intrinsics: [B, 4] Source camera intrinsics tensor (f_x, f_y, c_x, c_y)

  Returns:
    A [B, 4, 4] float32 Tensor that maps from right-handed camera space
    to left-handed clip space.
  """
  intrinsics = tf.convert_to_tensor(intrinsics)
  focal_x = intrinsics[:, 0]
  focal_y = intrinsics[:, 1]
  principal_x = intrinsics[:, 2]
  principal_y = intrinsics[:, 3]
  zero = tf.zeros_like(focal_x)
  one = tf.ones_like(focal_x)
  near_z = 0.00001 * one
  far_z = 10000.0 * one

  a = (near_z + far_z) / (far_z - near_z)
  b = -2.0 * near_z * far_z / (far_z - near_z)

  matrix = [
      [2.0 * focal_x, zero, 2.0 * principal_x - 1.0, zero],
      [zero, 2.0 * focal_y, 2.0 * principal_y - 1.0, zero],
      [zero, zero, a, b],
      [zero, zero, one, zero]]
  return tf.stack([tf.stack(row, axis=-1) for row in matrix], axis=-2)

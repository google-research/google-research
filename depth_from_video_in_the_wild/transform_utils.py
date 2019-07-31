# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Helper functions for geometric transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def matrix_from_angles(rot):
  """Create a rotation matrix from a triplet of rotation angles.

  Args:
    rot: a tf.Tensor of shape [..., 3], where the last dimension is the rotation
      angles, along x, y, and z.

  Returns:
    A tf.tensor of shape [..., 3, 3], where the last two dimensions are the
    rotation matrix.

  This function mimics _euler2mat from struct2depth/project.py, for backward
  compatibility, but wraps tensorflow_graphics instead of reimplementing it.
  The negation and transposition are needed to bridge the differences between
  the two.
  """
  rank = tf.rank(rot)
  # Swap the two last dimensions
  perm = tf.concat([tf.range(rank - 1), [rank], [rank - 1]], axis=0)
  return tf.transpose(rotation_matrix_3d.from_euler(-rot), perm)


def invert_rot_and_trans(rot, trans):
  """Inverts a transform comprised of a rotation and a translation.

  Args:
    rot: a tf.Tensor of shape [..., 3] representing rotatation angles.
    trans: a tf.Tensor of shape [..., 3] representing translation vectors.

  Returns:
    a tuple (inv_rot, inv_trans), representing rotation angles and translation
    vectors, such that applting rot, transm inv_rot, inv_trans, in succession
    results in identity.
  """
  inv_rot = inverse_euler(rot)  # inv_rot = -rot  for small angles
  inv_rot_mat = matrix_from_angles(inv_rot)
  inv_trans = -tf.matmul(inv_rot_mat, tf.expand_dims(trans, -1))
  inv_trans = tf.squeeze(inv_trans, -1)
  return inv_rot, inv_trans


def inverse_euler(angles):
  """Returns the euler angles that are the inverse of the input.

  Args:
    angles: a tf.Tensor of shape [..., 3]

  Returns:
    A tensor of the same shape, representing the inverse rotation.
  """
  sin_angles = tf.sin(angles)
  cos_angles = tf.cos(angles)
  sz, sy, sx = tf.unstack(-sin_angles, axis=-1)
  cz, _, cx = tf.unstack(cos_angles, axis=-1)
  y = tf.asin((cx * sy * cz) + (sx * sz))
  x = -tf.asin((sx * sy * cz) - (cx * sz)) / tf.cos(y)
  z = -tf.asin((cx * sy * sz) - (sx * cz)) / tf.cos(y)
  return tf.stack([x, y, z], axis=-1)


def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
  """Composes two transformations, each has a rotation and a translation.

  Args:
    rot_mat1: A tf.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec1: A tf.tensor of shape [..., 3] representing translation vectors.
    rot_mat2: A tf.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec2: A tf.tensor of shape [..., 3] representing translation vectors.

  Returns:
    A tuple of 2 tf.Tensors, representing rotation matrices and translation
    vectors, of the same shapes as the input, representing the result of
    applying rot1, trans1, rot2, trans2, in succession.
  """
  # Building a 4D transform matrix from each rotation and translation, and
  # multiplying the two, we'd get:
  #
  # (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
  # (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
  #
  # Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
  # a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
  # total translation is R2*t1 + t2.
  r2r1 = tf.matmul(rot_mat2, rot_mat1)
  r2t1 = tf.matmul(rot_mat2, tf.expand_dims(trans_vec1, -1))
  r2t1 = tf.squeeze(r2t1, axis=-1)
  return r2r1, r2t1 + trans_vec2

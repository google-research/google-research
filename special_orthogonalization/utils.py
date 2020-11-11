# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utility functions."""
import numpy as np
from scipy.stats import special_ortho_group
import tensorflow.compat.v1 as tf


def relative_angle(r1, r2):
  """Relative angle (radians) between 3D rotation matrices."""
  rel_rot = tf.matmul(tf.transpose(r1, perm=[0, 2, 1]), r2)
  trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
  cos_theta = (trace - 1.0) / 2.0
  cos_theta = tf.minimum(cos_theta, tf.ones_like(cos_theta))
  cos_theta = tf.maximum(cos_theta, (-1.0) * tf.ones_like(cos_theta))
  theta = tf.acos(cos_theta)
  return theta


def random_rotation_benchmark_np(n):
  """Sample a random 3D rotation by method used in Zhou et al, CVPR19.

  This numpy function is a copy of the PyTorch function
  get_sampled_rotation_matrices_by_axisAngle() in the code made available
  for Zhou et al, CVPR19, at https://github.com/papagina/RotationContinuity/.

  Args:
    n: the number of rotation matrices to return.

  Returns:
    [n, 3, 3] np array.
  """
  theta = np.random.uniform(-1, 1, n) * np.pi
  sin = np.sin(theta)
  axis = np.random.randn(n, 3)
  axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-7)
  qw = np.cos(theta)
  qx = axis[:, 0] * sin
  qy = axis[:, 1] * sin
  qz = axis[:, 2] * sin

  xx = qx*qx
  yy = qy*qy
  zz = qz*qz
  xy = qx*qy
  xz = qx*qz
  yz = qy*qz
  xw = qx*qw
  yw = qy*qw
  zw = qz*qw

  row0 = np.stack((1-2*yy-2*zz, 2*xy-2*zw, 2*xz+2*yw), axis=-1)
  row1 = np.stack((2*xy+2*zw, 1-2*xx-2*zz, 2*yz-2*xw), axis=-1)
  row2 = np.stack((2*xz-2*yw, 2*yz+2*xw, 1-2*xx-2*yy), axis=-1)
  matrix = np.stack((row0, row1, row2), axis=1)

  return matrix


def random_rotation_benchmark(n):
  """A TF wrapper for random_rotation_benchmark_np()."""
  mat = tf.compat.v1.py_func(
      func=lambda t: np.float32(random_rotation_benchmark_np(t)),
      inp=[n],
      Tout=tf.float32,
      stateful=True)
  return tf.reshape(mat, (n, 3, 3))


def random_rotation(n):
  """Sample rotations from a uniform distribution on SO(3)."""
  mat = tf.compat.v1.py_func(
      func=lambda t: np.float32(special_ortho_group.rvs(3, size=t)),
      inp=[n],
      Tout=tf.float32,
      stateful=True)
  return tf.reshape(mat, (n, 3, 3))


def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization."""
  # Innner dimensions of the input should be 3x3 matrices.
  m = tf.reshape(x, (-1, 3, 3))
  _, u, v = tf.svd(m)
  det = tf.linalg.det(tf.matmul(u, v, transpose_b=True))
  r = tf.matmul(
      tf.concat([u[:, :, :-1], u[:, :, -1:] * tf.reshape(det, [-1, 1, 1])], 2),
      v, transpose_b=True)
  return r


def gs_orthogonalization(p6):
  """Gram-Schmidt orthogonalization from 6D input."""
  # Input should be [batch_size, 6]
  x = p6[:, 0:3]
  y = p6[:, 3:6]
  xn = tf.math.l2_normalize(x, axis=-1)
  z = tf.linalg.cross(xn, y)
  zn = tf.math.l2_normalize(z, axis=-1)
  y = tf.linalg.cross(zn, xn)
  r = tf.stack([xn, y, zn], -1)
  return r

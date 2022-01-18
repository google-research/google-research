# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Utilities for mathematical operations."""
import math
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def degrees_to_radians(degree):
  """Convert degrees to radians."""
  return math.pi * degree / 180.0


def radians_to_degrees(radians):
  """Convert radians to degrees."""
  return 180.0 * radians / math.pi


def safe_sqrt(x):
  return tf.sqrt(tf.maximum(x, 1e-20))


def argmax2d(tensor):
  """Find the indices where the peaks value is on 2d maps.

  If there are multiple locations with the same value, return the top left one.

  Args:
    tensor: a 4-d tensor [BATCH, HEIGHT, WIDTH, N].

  Returns:
    [BATCH, N, 2] locations (row, col) for each channel and batch.
  """
  with tf.name_scope(None, 'argmax2d', [tensor]):
    shape = tf.shape(tensor)
    batch, width, channels = shape[0], shape[2], shape[3]
    flat_tensor = tf.reshape(tensor, (batch, -1, channels))
    index = tf.cast(tf.arg_max(flat_tensor, 1), tf.int32)
    y = index // width
    x = index % width
    locations = tf.stack([y, x], -1)
    return locations


def lookat_matrix(up, lookat_direction):
  """Return rotation matrices given camera's lookat directions and up vectors.

  Using OpenGL's coordinate system: -Z is the camera's lookat and +Y is up.
  Args:
    up: [BATCH, 3] the up vectors.
    lookat_direction:  [BATCH, 3] the lookat directions.

  Returns:
    [BATCH, 3, 3] the rotation matrices (from camera to world frame).
  """
  z = tf.linalg.l2_normalize(-lookat_direction, axis=-1)
  x = tf.linalg.l2_normalize(tf.cross(up, z), axis=-1)
  y = tf.cross(z, x)
  # Stack x, y, z basis vectors by colomn.
  return tf.stack([x, y, z], axis=-1)


def skew_symmetric(v):
  """Return skew symmetric matrices of 3d vectors.

  Args:
    v: [BATCH, 3] 3d vectors.

  Returns:
    [BATCH, 3, 3] skew symmetric matrices.

  Raises:
    Input has the wrong dimensions.
  """
  with tf.name_scope(None, 'skew_symmetric', [v]):
    if v.shape[-1] != 3:
      raise ValueError('Input has the wrong dimensions.')

    batch = v.shape.as_list()[0]
    v1, v2, v3 = tf.split(v, [1, 1, 1], -1)
    zeros = tf.zeros([batch, 1])
    lower = tf.concat([zeros, v1, -v2, zeros, zeros, v3], -1)
    upper = tf.concat([zeros, -v3, v2, zeros, zeros, -v1], -1)
    return tfp.math.fill_triangular(lower) + \
            tfp.math.fill_triangular(upper, upper=True)


def random_vector_on_sphere(batch, limits):
  """Randomly sample a point on a unit sphere within a range."""
  min_y, max_y = limits[0][0], limits[0][1]
  min_theta, max_theta = limits[1][0], limits[1][1]
  y = tf.random_uniform([batch, 1], minval=min_y, maxval=max_y)
  theta = tf.random_uniform([batch, 1], minval=min_theta, maxval=max_theta)
  cos_phi = tf.sqrt(1-tf.square(y))
  x = cos_phi*tf.cos(theta)
  z = -cos_phi*tf.sin(theta)
  return tf.concat([x, y, z], -1)


def uniform_sampled_vector_within_cone(axis, angle):
  """Return uniformly sampled random unit vector within a cone.

  Args:
    axis: [BATCH, 3] the axis of the cone.
    angle: (float) the radian angle from the axis to the generatrix of the cone.

  Returns:
    [BATCH, 3] unit vectors.
  """
  if angle >= math.pi/2 or angle < 0:
    raise ValueError("'angle' must be within (0, pi/2).")

  batch = axis.shape.as_list()[0]
  y = tf.cos(angle * tf.random_uniform([batch, 1], minval=0., maxval=1.0))
  phi = tf.random_uniform([batch, 1], minval=0., maxval=2*math.pi)
  r = safe_sqrt(1 - y ** 2)
  x = r * tf.cos(phi)
  z = r * tf.sin(phi)
  v = tf.concat([x, y, z], -1)
  y_axis = tf.tile(tf.constant([[0., 1., 0.]]), [batch, 1])
  rot = rotation_between_vectors(y_axis, axis)
  return tf.squeeze(tf.matmul(rot, tf.expand_dims(v, -1)), -1)


def normal_sampled_vector_within_cone(axis, angle, std=1.):
  """Return normally sampled random unit vector within a cone.

  Args:
    axis: [BATCH, 3] the axis of the cone.
    angle: (float) the radian angle from the axis to the generatrix of the cone.
    std: (float) standard deviation of the normal distribution.

  Returns:
    [BATCH, 3] unit vectors.
  """
  if angle >= math.pi/2 or angle < 0:
    raise ValueError("'angle' must be within (0, pi/2).")

  batch = axis.shape.as_list()[0]
  y = tf.cos(angle * tf.truncated_normal([batch, 1], stddev=std))
  phi = tf.random_uniform([batch, 1], minval=0., maxval=2 * math.pi)
  r = safe_sqrt(1 - y ** 2)
  x = r * tf.cos(phi)
  z = r * tf.sin(phi)
  v = tf.concat([x, y, z], -1)
  y_axis = tf.tile(tf.constant([[0., 1., 0.]]), [batch, 1])
  rot = rotation_between_vectors(y_axis, axis)
  return tf.squeeze(tf.matmul(rot, tf.expand_dims(v, -1)), -1)


def rotation_between_vectors(v1, v2):
  """Get the rotation matrix to align v1 with v2 (v2 = R * v1).

  The rotation is computed by Rodrigues' rotation formula given an axis-angle.
  The function returns an identity matrix when v1 = v2. When v1 = -v2, this
  gives an 180-degree rotation around any axis perpendicular to v1 or v2 . The
  function is discontinuous in this case.

  Args:
    v1: [BATCH, 3] 3d vectors.
    v2: [BATCH, 3] 3d vectors.

  Returns:
    [BATCH, 3, 3] rotation matrices.

  Raises:
    Input has the wrong dimensions.
  """
  with tf.name_scope(None, 'rotation_between_vectors', [v1, v2]):
    if v1.shape[-1] != 3 or v2.shape[-1] != 3:
      raise ValueError('Input has the wrong dimensions.')

    batch = v1.shape.as_list()[0]
    v1 = tf.linalg.l2_normalize(v1, -1)
    v2 = tf.linalg.l2_normalize(v2, -1)
    cross = tf.cross(v1, v2)
    cos_angle = tf.reduce_sum(v1 * v2, -1, keepdims=True)
    sin_angle = tf.norm(cross, axis=-1)
    x_axis = tf.tile(tf.constant([[1., 0., 0.]]), [batch, 1])
    y_axis = tf.tile(tf.constant([[0., 1., 0.]]), [batch, 1])
    z_axis = tf.tile(tf.constant([[0., 0., 1.]]), [batch, 1])
    identity = tf.eye(3, batch_shape=[batch])
    rotation_axis = tf.where(
        tf.abs(tf.tile(cos_angle, [1, 3]) - (-tf.ones([batch, 3]))) < 1e-6,
        tf.cross(v1, x_axis) + tf.cross(v1, y_axis) + tf.cross(v1, z_axis),
        cross)
    rotation_axis = tf.linalg.l2_normalize(rotation_axis, -1)
    ss = skew_symmetric(rotation_axis)
    sin_angle = sin_angle[:, tf.newaxis, tf.newaxis]
    cos_angle = cos_angle[:, tf.newaxis]
    rotation_matrix = identity + sin_angle * ss + (1 - cos_angle) * tf.matmul(
        ss, ss)
    return rotation_matrix

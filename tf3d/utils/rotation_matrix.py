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

"""Utility functions for rotation matrix."""

import tensorflow as tf


def from_euler_cos_sin(cos_x, sin_x, cos_y, sin_y, cos_z, sin_z):
  """Get rotation matrix from cos and sin of angles.

  Args:
    cos_x: rank N tensor of shape [..., 1] if not None.
    sin_x: rank N tensor of shape [..., 1] if not None.
    cos_y: rank N tensor of shape [..., 1] if not None.
    sin_y: rank N tensor of shape [..., 1] if not None.
    cos_z: rank N tensor of shape [..., 1] if not None.
    sin_z: rank N tensor of shape [..., 1] if not None.

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.
  """
  if cos_x is not None and sin_x is not None:
    rotation_x = from_rotation_around_x_cos_sin(cos_x, sin_x)
  else:
    rotation_x = None
  if cos_y is not None and sin_y is not None:
    rotation_y = from_rotation_around_y_cos_sin(cos_y, sin_y)
  else:
    rotation_y = None
  if cos_z is not None and sin_z is not None:
    rotation_z = from_rotation_around_z_cos_sin(cos_z, sin_z)
  else:
    rotation_z = None
  rotation_zy = rotation_y
  if rotation_z is not None:
    if rotation_y is not None:
      rotation_zy = tf.matmul(rotation_z, rotation_y)
    else:
      rotation_zy = rotation_z
  rotation_zyx = rotation_x
  if rotation_zy is not None:
    if rotation_x is not None:
      rotation_zyx = tf.matmul(rotation_zy, rotation_x)
    else:
      rotation_zyx = rotation_zy
  return rotation_zyx


def from_euler(angles):
  """Get rotation matrix from euler angles.

  Currently the function assumes the 'ZYX' convention. So the resulting rotation
  matrix is a product of R_z * Ry * R_x.

  Args:
    angles: rank N tensor of shape [..., 3].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of angle is invalid.
  """
  if angles.shape[-1] not in [3, None]:
    raise ValueError('angle must be a tensor of shape [..., 3]')
  theta_x, theta_y, theta_z = tf.unstack(angles, axis=-1)
  rotation_x = from_rotation_around_x(theta_x)
  rotation_y = from_rotation_around_y(theta_y)
  rotation_z = from_rotation_around_z(theta_z)
  return tf.matmul(tf.matmul(rotation_z, rotation_y), rotation_x)


def from_rotation_around_x_cos_sin(cos, sin):
  """Get the rotation matrix for rotation around x given its cos and sin.

  Args:
    cos: rank N tensor of shape [..., 1].
    sin: rank N tensor of shape [..., 1].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of angle is invalid.
  """
  cos = tf.convert_to_tensor(cos, dtype=tf.float32)
  sin = tf.convert_to_tensor(sin, dtype=tf.float32)
  if len(cos.shape) <= 1:
    cos = tf.expand_dims(cos, -1)
    sin = tf.expand_dims(sin, -1)
  if cos.shape[-1] not in [1, None]:
    raise ValueError('cos must be a tensor of shape [..., 1]')
  if sin.shape[-1] not in [1, None]:
    raise ValueError('sin must be a tensor of shape [..., 1]')
  ones = tf.ones_like(cos)
  zeros = tf.zeros_like(sin)
  # pyformat: disable
  rotation = tf.stack([ones, zeros, zeros,  #  1  0  0
                       zeros, cos, -sin,    #  0  c -s
                       zeros, sin, cos,],   #  0  s  c
                      axis=-1)
  # pyformat: enable
  output_shape = tf.concat((tf.shape(cos)[:-1], (3, 3)), axis=0)
  return tf.reshape(rotation, shape=output_shape)


def from_rotation_around_x(angle):
  """Get the rotation matrix for rotation around X.

  Args:
    angle: rank N tensor of shape [..., 1].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of angle is invalid.
  """
  angle = tf.convert_to_tensor(angle, dtype=tf.float32)
  cos = tf.cos(angle)
  sin = tf.sin(angle)
  return from_rotation_around_x_cos_sin(cos, sin)


def from_rotation_around_y_cos_sin(cos, sin):
  """Get the rotation matrix for rotation around Y given its cos and sin.

  Args:
    cos: rank N tensor of shape [..., 1].
    sin: rank N tensor of shape [..., 1].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of cos or sin is invalid.
  """
  cos = tf.convert_to_tensor(cos, dtype=tf.float32)
  sin = tf.convert_to_tensor(sin, dtype=tf.float32)
  if len(cos.shape) <= 1:
    cos = tf.expand_dims(cos, -1)
  if len(sin.shape) <= 1:
    sin = tf.expand_dims(sin, -1)
  if cos.shape[-1] not in [1, None]:
    raise ValueError('cos must be a tensor of shape [..., 1]')
  if sin.shape[-1] not in [1, None]:
    raise ValueError('sin must be a tensor of shape [..., 1]')
  ones = tf.ones_like(cos)
  zeros = tf.zeros_like(sin)
  # pyformat: disable
  rotation = tf.stack([cos, zeros, sin,     #  c  0  s
                       zeros, ones, zeros,  #  0  1  0
                       -sin, zeros, cos],   # -s  0  c
                      axis=-1)
  # pyformat: enable
  output_shape = tf.concat((tf.shape(cos)[:-1], (3, 3)), axis=0)
  return tf.reshape(rotation, shape=output_shape)


def from_rotation_around_y(angle):
  """Get the rotation matrix for rotation around Y.

  Args:
    angle: rank N tensor of shape [..., 1].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of angle is invalid.
  """
  angle = tf.convert_to_tensor(angle, dtype=tf.float32)
  cos = tf.cos(angle)
  sin = tf.sin(angle)
  return from_rotation_around_y_cos_sin(cos=cos, sin=sin)


def from_rotation_around_z_cos_sin(cos, sin):
  """Get the rotation matrix for rotation around z given its cos and sin.

  Args:
    cos: rank N tensor of shape [..., 1].
    sin: rank N tensor of shape [..., 1].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of angle is invalid.
  """
  cos = tf.convert_to_tensor(cos, dtype=tf.float32)
  sin = tf.convert_to_tensor(sin, dtype=tf.float32)
  if len(cos.shape) <= 1:
    cos = tf.expand_dims(cos, -1)
  if len(sin.shape) <= 1:
    sin = tf.expand_dims(sin, -1)
  if cos.shape[-1] not in [1, None]:
    raise ValueError('cos must be a tensor of shape [..., 1]')
  if sin.shape[-1] not in [1, None]:
    raise ValueError('sin must be a tensor of shape [..., 1]')
  ones = tf.ones_like(cos)
  zeros = tf.zeros_like(sin)
  # pyformat: disable
  rotation = tf.stack([cos, -sin, zeros,     #  c -s  0
                       sin, cos, zeros,      #  s  c  0
                       zeros, zeros, ones],  #  0  0  1
                      axis=-1)
  # pyformat: enable
  output_shape = tf.concat((tf.shape(cos)[:-1], (3, 3)), axis=0)
  return tf.reshape(rotation, shape=output_shape)


def from_rotation_around_z(angle):
  """Get the rotation matrix for rotation around z.

  Args:
    angle: rank N tensor of shape [..., 1].

  Returns:
    rank N+1 tensor of shape [..., 3, 3] containing rotation matrices.

  Raises:
    ValueError: if the shape of angle is invalid.
  """
  angle = tf.convert_to_tensor(angle, dtype=tf.float32)
  cos = tf.cos(angle)
  sin = tf.sin(angle)
  return from_rotation_around_z_cos_sin(cos, sin)

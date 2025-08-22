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

"""Quaternion math.

This module assumes the xyzw quaternion format where xyz is the imaginary part
and w is the real part.

Functions in this module support both batched and unbatched quaternions.

Some parts have been adapted from Ceres.
"""

from internal import spin_math
from jax import numpy as jnp
from jax.numpy import linalg


def _safe_sqrt(x):
  """safe_sqrt with the value at zero set to eps to avoid divide by zero."""
  return spin_math.safe_sqrt(x, value_at_zero=jnp.finfo(jnp.float32).eps)


def im(q):
  """Fetch the imaginary part of the quaternion."""
  return q[Ellipsis, :3]


def re(q):
  """Fetch the real part of the quaternion."""
  return q[Ellipsis, 3:]


def identity():
  return jnp.array([0.0, 0.0, 0.0, 1.0])


def conjugate(q):
  """Compute the conjugate of a quaternion."""
  return jnp.concatenate([-im(q), re(q)], axis=-1)


def inverse(q):
  """Compute the inverse of a quaternion."""
  return normalize(conjugate(q))


def normalize(q):
  """Normalize a quaternion."""
  return q / norm(q)


def norm(q):
  return linalg.norm(q, axis=-1, keepdims=True)


def multiply(q1, q2):
  """Multiply two quaternions."""
  c = re(q1) * im(q2) + re(q2) * im(q1) + jnp.cross(im(q1), im(q2))
  w = re(q1) * re(q2) - jnp.dot(im(q1), im(q2))
  return jnp.concatenate([c, w], axis=-1)


def rotate(q, v):
  """Rotate a vector using a quaternion."""
  # Create the quaternion representation of the vector.
  q_v = jnp.concatenate([v, jnp.zeros_like(v[Ellipsis, :1])], axis=-1)
  return im(multiply(multiply(q, q_v), conjugate(q)))


def log(q, eps = 1e-8):
  """Computes the quaternion logarithm.

  References:
    https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

  Args:
    q: the quaternion in (x,y,z,w) format.
    eps: an epsilon value for numerical stability.

  Returns:
    The logarithm of q.
  """
  mag = linalg.norm(q, axis=-1, keepdims=True)
  v = im(q)
  s = re(q)
  w = jnp.log(mag)
  denom = jnp.maximum(
      linalg.norm(v, axis=-1, keepdims=True), eps * jnp.ones_like(v)
  )
  xyz = v / denom * spin_math.safe_acos(s / eps)
  return jnp.concatenate((xyz, w), axis=-1)


def exp(q, eps = 1e-8):
  """Computes the quaternion exponential.

  References:
    https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions

  Args:
    q: the quaternion in (x,y,z,w) format or (x,y,z) if is_pure is True.
    eps: an epsilon value for numerical stability.

  Returns:
    The exponential of q.
  """
  is_pure = q.shape[-1] == 3
  if is_pure:
    s = jnp.zeros_like(q[Ellipsis, -1:])
    v = q
  else:
    v = im(q)
    s = re(q)

  norm_v = linalg.norm(v, axis=-1, keepdims=True)
  exp_s = jnp.exp(s)
  w = jnp.cos(norm_v)
  xyz = jnp.sin(norm_v) * v / jnp.maximum(norm_v, eps * jnp.ones_like(norm_v))
  return exp_s * jnp.concatenate((xyz, w), axis=-1)


def to_rotation_matrix(q):
  """Constructs a rotation matrix from a quaternion.

  Args:
    q: a (*,4) array containing quaternions.

  Returns:
    A (*,3,3) array containing rotation matrices.
  """
  x, y, z, w = jnp.split(q, 4, axis=-1)
  s = 1.0 / jnp.sum(q**2, axis=-1)
  return jnp.stack(
      [
          jnp.stack(
              [
                  1 - 2 * s * (y**2 + z**2),
                  2 * s * (x * y - z * w),
                  2 * s * (x * z + y * w),
              ],
              axis=0,
          ),
          jnp.stack(
              [
                  2 * s * (x * y + z * w),
                  1 - s * 2 * (x**2 + z**2),
                  2 * s * (y * z - x * w),
              ],
              axis=0,
          ),
          jnp.stack(
              [
                  2 * s * (x * z - y * w),
                  2 * s * (y * z + x * w),
                  1 - 2 * s * (x**2 + y**2),
              ],
              axis=0,
          ),
      ],
      axis=0,
  )


def from_rotation_matrix(m, eps = 1e-9):
  """Construct quaternion from a rotation matrix.

  Args:
    m: a (*,3,3) array containing rotation matrices.
    eps: a small number for numerical stability.

  Returns:
    A (*,4) array containing quaternions.
  """
  trace = jnp.trace(m)
  m00 = m[Ellipsis, 0, 0]
  m01 = m[Ellipsis, 0, 1]
  m02 = m[Ellipsis, 0, 2]
  m10 = m[Ellipsis, 1, 0]
  m11 = m[Ellipsis, 1, 1]
  m12 = m[Ellipsis, 1, 2]
  m20 = m[Ellipsis, 2, 0]
  m21 = m[Ellipsis, 2, 1]
  m22 = m[Ellipsis, 2, 2]

  def tr_positive():
    sq = _safe_sqrt(trace + 1.0) * 2.0  # sq = 4 * w.
    w = 0.25 * sq
    x = jnp.divide(m21 - m12, sq)
    y = jnp.divide(m02 - m20, sq)
    z = jnp.divide(m10 - m01, sq)
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_1():
    sq = _safe_sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * x.
    w = jnp.divide(m21 - m12, sq)
    x = 0.25 * sq
    y = jnp.divide(m01 + m10, sq)
    z = jnp.divide(m02 + m20, sq)
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_2():
    sq = _safe_sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * y.
    w = jnp.divide(m02 - m20, sq)
    x = jnp.divide(m01 + m10, sq)
    y = 0.25 * sq
    z = jnp.divide(m12 + m21, sq)
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_3():
    sq = _safe_sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * z.
    w = jnp.divide(m10 - m01, sq)
    x = jnp.divide(m02 + m20, sq)
    y = jnp.divide(m12 + m21, sq)
    z = 0.25 * sq
    return jnp.stack((x, y, z, w), axis=-1)

  def cond_idx(cond):
    cond = jnp.expand_dims(cond, -1)
    cond = jnp.tile(cond, [1] * (len(m.shape) - 2) + [4])
    return cond

  where_2 = jnp.where(cond_idx(m11 > m22), cond_2(), cond_3())
  where_1 = jnp.where(cond_idx((m00 > m11) & (m00 > m22)), cond_1(), where_2)
  return jnp.where(cond_idx(trace > 0), tr_positive(), where_1)


def from_axis_angle(
    axis_angle, eps = jnp.finfo(jnp.float32).eps
):
  """Constructs a quaternion for the given axis/angle rotation.

  Args:
    axis_angle: A 3-vector where the direction is the axis of rotation and the
      magnitude is the angle of rotation.
    eps: A small number used for numerical stability around zero rotations.

  Returns:
    A quaternion encoding the same rotation.
  """
  theta_squared = jnp.sum(axis_angle**2, axis=-1)
  theta = _safe_sqrt(theta_squared)
  half_theta = theta / 2.0
  k = jnp.sin(half_theta) / theta
  # Avoid evaluating sqrt when theta is close to zero.
  k = jnp.where(theta_squared > eps**2, k, 0.5)
  qw = jnp.where(theta_squared > eps**2, jnp.cos(half_theta), 1.0)
  qx = axis_angle[0] * k
  qy = axis_angle[1] * k
  qz = axis_angle[2] * k

  return jnp.squeeze(jnp.array([qx, qy, qz, qw]))


def to_axis_angle(
    q, eps = jnp.finfo(jnp.float32).eps
):
  """Converts a quaternion to an axis-angle representation.

  Args:
    q: a 4-vector representing a unit quaternion.
    eps: A small number used for numerical stability around zero rotations.

  Returns:
    A 3-vector where the direction is the axis of rotation and the magnitude
      is the angle of rotation.
  """
  sin_sq_theta = jnp.sum(im(q) ** 2, axis=-1)

  sin_theta = _safe_sqrt(sin_sq_theta)
  cos_theta = re(q)

  # If cos_theta is negative, theta is greater than pi/2, which
  # means that angle for the angle_axis vector which is 2 * theta
  # would be greater than pi.
  #
  # While this will result in the correct rotation, it does not
  # result in a normalized angle-axis vector.
  #
  # In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
  # which is equivalent saying
  #
  #   theta - pi = atan(sin(theta - pi), cos(theta - pi))
  #              = atan(-sin(theta), -cos(theta))
  two_theta = 2.0 * jnp.where(
      cos_theta < 0.0,
      jnp.arctan2(-sin_theta, -cos_theta),
      jnp.arctan2(sin_theta, cos_theta),
  )

  # For zero rotation, sqrt() will produce NaN in the derivative since
  # the argument is zero. We avoid this by directly returning the value in
  # such cases.
  k = jnp.where(sin_sq_theta > eps**2, two_theta / sin_theta, 2.0)

  return im(q) * k

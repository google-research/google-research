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

"""Functions and transforms for rigid body dynamics.

Many equations are from the Modern Robotics textbook available online at:
  http://hades.mech.northwestern.edu/index.php/Modern_Robotics

Note that many operations here use a `jnp.where` to avoid evaluating at
numerically unstable or undefined regions of the domain. In addition, to avoid
NaNs accumulating through `jnp.where` expressions of unsafe math operations,
we also wrap the argument of those operations in another `jnp.where` call

See:
  https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
"""

from typing import Tuple

from internal import quaternion as quat_lib
from internal import spin_math
import jax
from jax import numpy as jnp
import optax


def _safe_sqrt(x):
  """safe_sqrt with the value at zero set to eps to avoid divide by zero."""
  return spin_math.safe_sqrt(x, value_at_zero=jnp.finfo(jnp.float32).eps)


@jax.jit
def skew(w):
  """Build a skew matrix ("cross product matrix") for vector w.

  Modern Robotics Eqn 3.30.

  Args:
    w: (3,) A 3-vector

  Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v
  """
  w = jnp.reshape(w, (3))
  return jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def unskew(W):
  """Convert a skew matrix to a vector w.

  See `skew()` for documentation.

  Args:
    W: (3, 3) A skew matrix.

  Returns:
    w: (3,) A 3-vector corresponding to the skew matrix.
  """
  return jnp.stack([W[2, 1], W[0, 2], W[1, 0]], axis=-1)


def rp_to_se3(R, p):
  """Rotation and translation to homogeneous transform.

  Args:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.

  Returns:
    X: (4, 4) The homogeneous transformation matrix described by rotating by R
      and translating by p.
  """
  p = jnp.reshape(p, (3, 1))
  return jnp.block([[R, p], [jnp.array([[0.0, 0.0, 0.0, 1.0]])]])


def se3_to_rp(X):
  """Converts a homogeneous transform to a rotation and translation.

  Args:
    X: (4, 4) A homogeneous transformation matrix.

  Returns:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.
  """
  R = X[Ellipsis, :3, :3]
  p = X[Ellipsis, :3, 3]
  return R, p


def exp_so3(
    axis_angle, eps=jnp.finfo(jnp.float32).eps
):
  """Exponential map from Lie algebra so3 to Lie group SO3.

  Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

  Args:
    axis_angle: A 3-vector where the direction is the axis of rotation and the
      magnitude is the angle of rotation.
    eps: an epsilon value for numerical stability.

  Returns:
    R: (3, 3) An orthonormal rotation matrix representing the same rotation.
  """
  theta_squared = jnp.sum(axis_angle**2, axis=-1)
  theta = _safe_sqrt(theta_squared)

  # Near zero, we switch to using the first order Taylor expansion.
  R_taylor = jnp.eye(3) + skew(axis_angle)

  # Prevent bad gradients from propagating back when theta is small.
  axis_angle_safe = jnp.where(theta_squared > eps**2, axis_angle, 0.0)
  theta_safe = jnp.where(theta_squared > eps**2, theta, 1.0)
  axis = axis_angle_safe / theta_safe
  W = skew(axis)
  R = (
      jnp.eye(3)
      + jnp.sin(theta_safe) * W
      + (1.0 - jnp.cos(theta_safe)) * spin_math.matmul(W, W)
  )

  return jnp.where(theta_squared > eps**2, R, R_taylor)


def log_so3(R, eps=jnp.finfo(jnp.float32).eps):
  """Matrix logarithm from the Lie group SO3 to the Lie algebra so3.

  Modern Robotics Eqn 3.53.

  Args:
    R: (3, 3) An orthonormal rotation matrix.
    eps: an epsilon value for numerical stability.

  Returns:
    w: (3,) The unit vector representing the axis of rotation.
    theta: The angle of rotation.
  """
  q = quat_lib.from_rotation_matrix(R, eps)
  axis_angle = quat_lib.to_axis_angle(q, eps)
  return axis_angle


def exp_se3(
    screw_axis, eps=jnp.finfo(jnp.float32).eps
):
  """Exponential map from Lie algebra so3 to Lie group SO3.

  Modern Robotics Eqn 3.88.

  Args:
    screw_axis: A 6-vector encoding a screw axis of motion. This can be broken
      down into [w, v] where w is an angle-axis rotation and v represents a
      translation. ||w|| corresponds to the magnitude of motion.
    eps: an epsilon value for numerical stability.

  Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
      motion of magnitude theta about S for one second.
  """
  w, v = jnp.split(screw_axis, 2)
  R = exp_so3(w)
  theta_squared = jnp.sum(w**2, axis=-1)
  theta = _safe_sqrt(theta_squared)
  W = skew(w / theta)
  # Note that p = 0 when theta = 0.
  p = spin_math.matmul(
      (
          theta * jnp.eye(3)
          + (1.0 - jnp.cos(theta)) * W
          + (theta - jnp.sin(theta)) * spin_math.matmul(W, W)
      ),
      v / theta,
  )
  # If theta^2 is close to 0 it means this is a pure translation so p = v.
  p = jnp.where(theta_squared > eps**2, p, v)
  return rp_to_se3(R, p)


def log_se3(a_X_b, eps=jnp.finfo(jnp.float32).eps):
  """Matrix logarithm from the Lie group SE3 to the Lie algebra se3.

  Modern Robotics Eqn 3.91-3.92.

  Args:
    a_X_b: (4,4) A homogeneous transformation matrix.
    eps: an epsilon value for numerical stability.

  Returns:
    screw_axis: A 6-vector encoding a screw axis of motion. This can be broken
      down into [w, v] where w is an angle-axis rotation and v represents a
      translation. The ||w|| and ||v|| both correspond to the magnitude of
      motion.
  """
  R, p = se3_to_rp(a_X_b)
  w = log_so3(R, eps)
  theta_squared = jnp.sum(w**2, axis=-1)
  theta = spin_math.safe_sqrt(theta_squared)
  W = skew(w / theta)

  G_inv1 = jnp.eye(3)
  G_inv2 = theta * -W / 2.0
  G_inv3 = (1.0 - 0.5 * theta / jnp.tan(theta / 2.0)) * spin_math.matmul(W, W)
  G_inv = G_inv1 + G_inv2 + G_inv3

  v = spin_math.matmul(G_inv, p[Ellipsis, jnp.newaxis]).squeeze(-1)
  # If theta = 0 then the transformation is a pure translation and v = p.
  # This avoids using the numerically unstable G matrix when theta is near zero.
  v = jnp.where(theta_squared > eps, v, p)
  S = jnp.concatenate([w, v], axis=-1)
  return S


def rts_to_sim3(
    rotation, translation, scale
):
  """Converts a rotation, translation and scale to a homogeneous transform.

  Args:
    rotation: (3, 3) An orthonormal rotation matrix.
    translation: (3,) A 3-vector representing a translation.
    scale: A scalar factor.

  Returns:
    (4, 4) A homogeneous transformation matrix.
  """

  transform = jnp.eye(4)
  transform = transform.at[:3, :3].set(rotation * scale)
  transform = transform.at[:3, 3].set(translation)

  return transform


def sim3_to_rts(
    transform,
):
  """Converts a homogeneous transform to rotation, translation and scale.

  Args:
    transform: (4, 4) A homogeneous transformation matrix.

  Returns:
    rotation: (3, 3) An orthonormal rotation matrix.
    translation: (3,) A 3-vector representing a translation.
    scale: A scalar factor.
  """

  eps = jnp.float32(jnp.finfo(jnp.float32).tiny)
  rotation_scale = transform[Ellipsis, :3, :3]
  # Assumes rotation is an orthonormal transform, thus taking norm of first row.
  scale = optax.safe_norm(rotation_scale, min_norm=eps, axis=1)[0]
  rotation = rotation_scale / scale
  translation = transform[Ellipsis, :3, 3]
  return rotation, translation, scale


def ortho6d_from_rotation_matrix(rotation_matrix):
  """Converts a matrix to an ortho6d by taking the first two columns."""
  return rotation_matrix[Ellipsis, :2, :].reshape(*rotation_matrix.shape[:-2], 6)


def rotation_matrix_from_ortho6d(ortho6d):
  """Computes the 3D rotation matrix from the 6D representation.

  Zhou et al. have proposed a novel 6D representation for the rotation in
  SO(3) which is completely continuous. This is highly benificial and produces
  better results than most standard rotation representations for many tasks,
  especially when the predicted value is close to the discontinuity of the
  utilized rotation represantation. This function converts from the proposed 6
  dimensional representation to the classic 3x3 rotation matrix.

  See https://arxiv.org/pdf/1812.07035.pdf for more information.

  Args:
    ortho6d: 6D represantion for the rotation according Zhou et al. of shape
      [6].

  Returns:
    (3, 3) The associated 3x3 rotation matrices.
  """
  if ortho6d.ndim != 1 or ortho6d.shape[0] != 6:
    raise ValueError('The shape of the input ortho 6D vector needs to be (6).')

  a1, a2 = ortho6d[Ellipsis, :3], ortho6d[Ellipsis, 3:]
  b1 = spin_math.normalize(a1)
  b2 = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
  b2 = spin_math.normalize(b2)
  b3 = jnp.cross(b1, b2)
  return jnp.stack((b1, b2, b3), axis=-2)
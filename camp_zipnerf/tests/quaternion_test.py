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

"""Unit tests for quaternions."""

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
from internal import quaternion
from jax import random
import jax.numpy as jnp
import numpy as np


TEST_BATCH_SIZE = 128
TEST_ROTATION_AXES = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-0.20223016, 0.6677665, -0.7163734],
    [0.71292967, 0.53064775, 0.45841497],
    [-0.35238215, 0.81467855, -0.4605711],
    [0.00712328, -0.9661464, 0.25789577],
    [0.7036228, 0.44212067, 0.55627716],
    [0.0729339, -0.19503504, 0.97808075],
    [0.16014354, -0.902658, -0.39945287],
    [0.10118368, 0.60621494, -0.78883797],
    [-0.642572, -0.6796316, -0.35383916],
    [-0.13103311, -0.3223685, -0.9375014],
]
TEST_ANGLES = [
    0.0,
    1.0,
    math.pi / 2,
    math.pi / 4,
    1e-1,
    1e-4,
    1e-6,
    1e-8,  # Angle below eps (~1.19e-7).
]

_assert_allclose = functools.partial(
    np.testing.assert_allclose, rtol=1e-5, atol=1e-5
)


class QuaternionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = 42
    self._key = random.PRNGKey(self._seed)

  def test_identity(self):
    identity = quaternion.identity()
    self.assertLen(identity, 4)
    np.testing.assert_equal(identity.tolist(), [0.0, 0.0, 0.0, 1.0])

  @parameterized.named_parameters(
      ('single', (4,)), ('batched', (TEST_BATCH_SIZE, 4))
  )
  def test_real_imaginary_part(self, shape):
    if len(shape) > 1:
      num_quaternions = shape[0]
    else:
      num_quaternions = 1
    random_quat = random.uniform(self._key, shape=shape)
    imaginary = quaternion.im(random_quat)
    real = quaternion.re(random_quat)

    # The first three components are imaginary and the fourth is real.
    np.testing.assert_array_equal(
        jnp.prod(jnp.array(imaginary.shape)), num_quaternions * 3
    )
    np.testing.assert_array_equal(
        jnp.prod(jnp.array(real.shape)), num_quaternions
    )
    np.testing.assert_array_equal(
        random_quat[Ellipsis, :3].tolist(), imaginary[Ellipsis, :].tolist()
    )
    np.testing.assert_array_equal(
        random_quat[Ellipsis, 3:].tolist(), real[Ellipsis, :].tolist()
    )

  @parameterized.named_parameters(
      ('single', None), ('batched', TEST_BATCH_SIZE)
  )
  def test_conjugate(self, batch):
    if batch:
      shape = (batch, 4)
    else:
      shape = (4,)
    quat = random.uniform(self._key, shape=shape)
    conjugate = quaternion.conjugate(quat)
    self.assertTrue(jnp.all(-1 * quat[Ellipsis, :3] == conjugate[Ellipsis, :3]))
    self.assertTrue(jnp.all(quat[Ellipsis, 3:] == conjugate[Ellipsis, 3:]))

  @parameterized.named_parameters(
      ('single', None), ('batched', TEST_BATCH_SIZE)
  )
  def test_normalize(self, batch):
    eps = 1e-6
    if batch:
      shape = (batch, 4)
    else:
      shape = (4,)
    q = random.uniform(self._key, shape=shape)
    self.assertTrue(jnp.all(jnp.abs(quaternion.norm(q) - 1) > eps))
    q_norm = quaternion.normalize(q)
    self.assertTrue(jnp.all(jnp.abs(quaternion.norm(q_norm) - 1) < eps))

  @parameterized.product(axis=TEST_ROTATION_AXES, angle=TEST_ANGLES)
  def test_quaternion_axis_angle_round_trip(self, axis, angle):
    axis_angle = jnp.array(axis) * angle
    q = quaternion.from_axis_angle(axis_angle)

    axis_angle_rt = quaternion.to_axis_angle(q)
    angle_rt = jnp.linalg.norm(axis_angle_rt, axis=-1)

    # You cannot recover the axis if the angle is zero.
    if angle != 0:
      axis_rt = axis_angle_rt / angle_rt
      _assert_allclose(axis_rt, axis)

    _assert_allclose(angle_rt, angle)
    _assert_allclose(axis_angle, axis_angle_rt)

    q_rt = quaternion.from_axis_angle(axis_angle_rt)
    _assert_allclose(q, q_rt)


if __name__ == '__main__':
  absltest.main()
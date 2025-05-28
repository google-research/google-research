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

# pylint: disable=invalid-name
# pytype: disable=attribute-error
import functools
import math

from jax import numpy as jnp
from jax import random
import numpy as np

from internal import spin_math
from internal import quaternion
from internal import rigid_body
from absl.testing import absltest
from absl.testing import parameterized

TEST_BATCH_SIZE = 128
SAMPLE_POINTS = [
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]


_assert_allclose = functools.partial(
    np.testing.assert_allclose, rtol=1e-5, atol=1e-5
)


class RigidBodyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = 42
    self._key = random.PRNGKey(self._seed)

  @staticmethod
  def _process_parameters(batch, vector_size=4):
    if batch:
      shape = (batch, vector_size)
      num_vectors = batch
    else:
      shape = (vector_size,)
      num_vectors = 1

    return shape, num_vectors

  def get_random_vector(self, func, shape):
    if func == random.uniform:
      self._key, _ = random.split(self._key)
      return func(shape=shape, key=self._key)
    else:
      return func(shape=shape)

  @parameterized.product(
      func=[random.uniform, jnp.ones, jnp.zeros], sign1=[-1, 1], sign2=[-1, 1]
  )
  def test_skew_matrix(self, func, sign1, sign2):
    # The skew function does not support batched operation.
    shape, _ = self._process_parameters(None, 3)
    w = sign1 * self.get_random_vector(func, shape=shape)
    v = sign2 * self.get_random_vector(func, shape=shape)
    skew_matrix = rigid_body.skew(w)

    # Properties of a skew symmetric matrix.
    np.testing.assert_array_equal(jnp.trace(skew_matrix), 0)
    np.testing.assert_array_equal(-1 * jnp.transpose(skew_matrix), skew_matrix)

    # Does the matrix approximate the actual cross product?
    expected_cross_product = jnp.cross(w, v)
    predicted_cross_product = jnp.matmul(skew_matrix, v)
    _assert_allclose(expected_cross_product, predicted_cross_product)

  @parameterized.product(
      func=[random.uniform, jnp.ones], sign1=[-1, 1], sign2=[-1, 1]
  )
  def test_exp_so3(self, func, sign1, sign2):
    shape, num_vectors = self._process_parameters(None, 3)

    # Generate a normalized axis of rotation and the angle of rotation.
    w = sign1 * self.get_random_vector(func, shape=shape)
    w = w / jnp.linalg.norm(w)

    theta = sign2 * self.get_random_vector(func, shape=(num_vectors, 1))
    output = rigid_body.exp_so3(w * theta)

    # Verify orthonormality.
    _assert_allclose(jnp.matmul(jnp.transpose(output), output), jnp.eye(3))
    _assert_allclose(jnp.matmul(output, jnp.transpose(output)), jnp.eye(3))

  @parameterized.product(
      axis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      theta=[x * math.pi / 4 for x in range(8)],
      sign=[-1, 1],
      pt_input=SAMPLE_POINTS,
  )
  def test_exp_so3_rotation(self, axis, theta, sign, pt_input):
    axis = jnp.array(axis)
    theta = jnp.array(sign * theta)
    pt_input = jnp.array(pt_input)
    theta = jnp.expand_dims(theta, 0)

    axis = axis / jnp.linalg.norm(axis)
    rotation_matrix = rigid_body.exp_so3(axis * theta)
    predicted_output = jnp.matmul(rotation_matrix, pt_input)

    # Use a quaternion to compute the rotation and use it as a comparison.
    quat = quaternion.from_axis_angle(axis * theta)
    quaternion_output = quaternion.rotate(quat, pt_input)
    _assert_allclose(predicted_output, quaternion_output)

  @parameterized.product(
      func=[random.uniform, jnp.ones, jnp.zeros], sign1=[-1, 1], sign2=[-1, 1]
  )
  def test_rp_to_se3(self, func, sign1, sign2):
    shape, num_vectors = self._process_parameters(None, 3)
    w = sign1 * self.get_random_vector(func, shape=shape)
    r = rigid_body.exp_so3(w)

    p = sign2 * self.get_random_vector(func, shape=(num_vectors, 3))
    output = rigid_body.rp_to_se3(r, p)
    np.testing.assert_array_equal(output.shape, (4, 4))
    np.testing.assert_array_equal(jnp.squeeze(r), jnp.squeeze(output[0:3, 0:3]))
    np.testing.assert_array_equal(jnp.squeeze(p), jnp.squeeze(output[0:3, 3]))
    np.testing.assert_array_equal(
        jnp.squeeze(jnp.array([0.0, 0.0, 0.0, 1.0])), jnp.squeeze(output[3, :])
    )

  @parameterized.product(
      func=[random.uniform, jnp.ones, jnp.zeros], sign=[-1, 1], pt=SAMPLE_POINTS
  )
  def test_exp_se3_only_rotation(self, func, sign, pt):
    shape, _ = self._process_parameters(None, 3)
    pt = jnp.array(pt)
    w = sign * self.get_random_vector(func, shape=shape)
    v = jnp.zeros(shape=shape)
    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis)

    quat = quaternion.from_axis_angle(w)
    pt_rotated = quaternion.rotate(quat, pt)

    np.testing.assert_equal(transform.shape, (4, 4))
    pt_rotated_tf = spin_math.apply_homogeneous_transform(transform, pt)
    _assert_allclose(pt_rotated_tf, pt_rotated)

  @parameterized.product(
      func=[random.uniform, jnp.ones, jnp.zeros], sign=[-1, 1], pt=SAMPLE_POINTS
  )
  def test_exp_se3_only_translation(self, func, sign, pt):
    shape, _ = self._process_parameters(None, 3)
    w = jnp.zeros(shape=shape)
    v = sign * self.get_random_vector(func, shape=shape)
    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis)

    pt = jnp.array(pt)
    pt_translated = pt + v

    np.testing.assert_array_equal(transform.shape, (4, 4))
    pt_translated_tf = spin_math.apply_homogeneous_transform(transform, pt)
    _assert_allclose(pt_translated_tf, pt_translated)

  @parameterized.product(
      func=[random.uniform, jnp.ones], sign=[-1, 1], pt=SAMPLE_POINTS
  )
  def test_exp_se3_pure_rotation(self, func, sign, pt):
    shape, _ = self._process_parameters(None, 3)
    w = sign * self.get_random_vector(func, shape=shape)
    v = jnp.zeros(shape)

    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis)

    pt = np.array(pt)
    q = quaternion.from_axis_angle(w)
    pt_transformed = quaternion.rotate(q, pt)  # pytype: disable=wrong-arg-types  # jax-ndarray
    pt_transformed_tf = spin_math.apply_homogeneous_transform(transform, pt)  # pytype: disable=wrong-arg-types  # jax-ndarray
    np.testing.assert_array_equal(pt_transformed.shape, (3,))
    np.testing.assert_array_equal(pt_transformed_tf.shape, (3,))
    _assert_allclose(pt_transformed_tf, pt_transformed)

  @parameterized.product(
      func=[random.uniform, jnp.ones], sign=[-1, 1], pt=SAMPLE_POINTS
  )
  def test_exp_se3_pure_translation(self, func, sign, pt):
    shape, _ = self._process_parameters(None, 3)
    w = jnp.zeros(shape)
    v = sign * self.get_random_vector(func, shape=shape)

    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid_body.exp_se3(screw_axis)

    pt = np.array(pt)
    pt_transformed = pt + v
    pt_transformed_tf = spin_math.apply_homogeneous_transform(transform, pt)  # pytype: disable=wrong-arg-types  # jax-ndarray
    np.testing.assert_array_equal(pt_transformed.shape, (3,))
    np.testing.assert_array_equal(pt_transformed_tf.shape, (3,))
    _assert_allclose(pt_transformed_tf, pt_transformed)

  @parameterized.product(
      func=[random.uniform, jnp.ones], sign=[-1, 1]  # jnp.zeroes doesn't work.
  )
  def test_so3_round_trip(self, func, sign):
    shape, _ = self._process_parameters(None, 3)
    w = sign * self.get_random_vector(func, shape=shape)

    R = rigid_body.exp_so3(w)
    w_rt = rigid_body.log_so3(R)
    _assert_allclose(w, w_rt)

    R_rt = rigid_body.exp_so3(w_rt)
    _assert_allclose(R, R_rt)

  @parameterized.product(
      func=[random.uniform, jnp.ones], sign1=[-1, 1], sign2=[-1, 1]
  )
  def test_se3_round_trip(self, func, sign1, sign2):
    shape, _ = self._process_parameters(None, 3)
    w = sign1 * self.get_random_vector(func, shape=shape)
    v = sign2 * self.get_random_vector(func, shape=shape)
    S = jnp.concatenate([w, v], axis=-1)

    Rp = rigid_body.exp_se3(S)
    S_rt = rigid_body.log_se3(Rp)
    _assert_allclose(S, S_rt)

    Rp_rt = rigid_body.exp_se3(S_rt)
    _assert_allclose(Rp, Rp_rt)

  @parameterized.product(func=[random.uniform, jnp.ones])
  def test_rps_to_sim3_round_trip(self, func):
    angle_axis = self.get_random_vector(func, shape=(3,))
    R = rigid_body.exp_so3(angle_axis)
    t = self.get_random_vector(func, shape=(3,))
    s = self.get_random_vector(func, shape=(1,))

    transform_sim3 = rigid_body.rts_to_sim3(R, t, s)
    R_round_trip, t_round_trip, s_round_trip = rigid_body.sim3_to_rts(
        transform_sim3
    )

    _assert_allclose(R, R_round_trip)
    _assert_allclose(t, t_round_trip)
    _assert_allclose(s, s_round_trip)


if __name__ == "__main__":
  absltest.main()
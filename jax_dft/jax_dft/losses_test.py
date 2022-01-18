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

# Lint as: python3
"""Tests for jax_dft.losses."""

from absl.testing import absltest
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import losses


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class LossesTest(absltest.TestCase):

  def test_trajectory_mse_wrong_predict_ndim(self):
    with self.assertRaisesRegex(
        ValueError,
        'The size of the shape of predict should be '
        'greater or equal to 2, got 1'):
      losses.trajectory_mse(
          target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
          predict=jnp.array([0.6, 0.6, 0.6, 0.6]),
          discount=1.)

  def test_trajectory_mse_wrong_predict_target_ndim_difference(self):
    with self.assertRaisesRegex(
        ValueError,
        'The size of the shape of predict should be greater than '
        'the size of the shape of target by 1, '
        r'but got predict \(2\) and target \(2\)'):
      losses.trajectory_mse(
          target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
          predict=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
          discount=1.)

  def test_density_mse(self):
    self.assertAlmostEqual(
        float(losses.mean_square_error(
            target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
            predict=jnp.array([[0.4, 0.5, 0.2, 0.3], [0.6, 0.6, 0.6, 0.6]]))),
        # ((
        #   (0.4 - 0.2) ** 2 + (0.5 - 0.2) ** 2
        #   + (0.2 - 0.2) ** 2 + (0.3 - 0.2) ** 2
        # ) / 4 + 0) / 2 = 0.0175
        0.0175)

  def test_energy_mse(self):
    self.assertAlmostEqual(
        float(losses.mean_square_error(
            target=jnp.array([[0.2, 0.6]]),
            predict=jnp.array([[0.4, 0.7]]))),
        # ((0.4 - 0.2) ** 2 + (0.7 - 0.6) ** 2) / 2 = 0.025
        0.025)

  def test_get_discount_coefficients(self):
    np.testing.assert_allclose(
        losses._get_discount_coefficients(num_steps=4, discount=0.8),
        [0.512, 0.64, 0.8, 1.])

  def test_trajectory_mse_on_density(self):
    self.assertAlmostEqual(
        float(losses.trajectory_mse(
            target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
            predict=jnp.array([
                [[0.4, 0.5, 0.2, 0.3],
                 [0.3, 0.3, 0.2, 0.2],
                 [0.3, 0.3, 0.3, 0.2]],
                [[0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.5],
                 [0.6, 0.6, 0.6, 0.6]]]),
            discount=0.6)),
        # First sample in the batch:
        # (
        #   (0.4 - 0.2) ** 2 + (0.5 - 0.2) ** 2
        #   + (0.2 - 0.2) ** 2 + (0.3 - 0.2) ** 2
        # ) / 4 * 0.6 * 0.6
        # + (
        #   (0.3 - 0.2) ** 2 + (0.3 - 0.2) ** 2
        #   + (0.2 - 0.2) ** 2 + (0.2 - 0.2) ** 2
        # ) / 4 * 0.6
        # + (
        #   (0.3 - 0.2) ** 2 + (0.3 - 0.2) ** 2
        #   + (0.3 - 0.2) ** 2 + (0.2 - 0.2) ** 2
        # ) / 4 = 0.0231
        # Second sample in the batch:
        # (
        #   (0.6 - 0.6) ** 2 + (0.6 - 0.6) ** 2
        #   + (0.6 - 0.6) ** 2 + (0.6 - 0.6) ** 2
        # ) / 4 * 0.6 * 0.6
        # + (
        #   (0.6 - 0.6) ** 2 + (0.6 - 0.6) ** 2
        #   + (0.6 - 0.6) ** 2 + (0.5 - 0.6) ** 2
        # ) / 4 * 0.6
        # + (
        #   (0.6 - 0.6) ** 2 + (0.6 - 0.6) ** 2
        #   + (0.6 - 0.6) ** 2 + (0.6 - 0.6) ** 2
        # ) / 4 = 0.0015
        # Loss:
        # (0.0231 + 0.0015) / 2 = 0.0123
        0.0123)

  def test_trajectory_mse_on_energy(self):
    self.assertAlmostEqual(
        float(losses.trajectory_mse(
            target=jnp.array([0.2, 0.6]),
            predict=jnp.array([[0.4, 0.3, 0.2], [0.7, 0.7, 0.7]]),
            discount=0.6)),
        # First sample in the batch:
        # ((0.4 - 0.2) ** 2 * 0.6 * 0.6
        #  + (0.3 - 0.2) ** 2 * 0.6 + (0.2 - 0.2) ** 2) = 0.0204
        # Second sample in the batch:
        # ((0.7 - 0.6) ** 2 * 0.6 * 0.6
        #  + (0.7 - 0.6) ** 2 * 0.6 + (0.7 - 0.6) ** 2) = 0.0196
        # Loss:
        # (0.0204 + 0.0196) / 2 = 0.02
        0.02)


if __name__ == '__main__':
  absltest.main()

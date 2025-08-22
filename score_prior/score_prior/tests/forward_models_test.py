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

"""Tests for forward_models."""
import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from score_prior import forward_models


class DummyForwardModel(forward_models.IndependentGaussianLikelihood):
  """A dummy likelihood with a 2x3 measurement matrix."""

  def __init__(self):
    pass

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def apply_forward_operator(self, x):
    A = jnp.array([[1, 1, 2], [0, 2, 3]])  # pylint:disable=invalid-name
    return A @ x

  @property
  def sigmas(self):
    return jnp.array([2, 1])


def test_cases():
  image_shape = (16, 16, 3)
  dim = image_shape[0] * image_shape[1] * image_shape[2]
  return [
      dict(
          likelihood=forward_models.Denoising(
              scale=0.2, image_shape=image_shape),
          expected_y_dim=dim,
          seed=0
      ),
      dict(
          likelihood=forward_models.Deblurring(
              n_freqs_per_direction=8, sigmas=np.ones(8 * 8),
              image_shape=(16, 16, 1)),
          expected_y_dim=8 * 8 * 2,
          seed=1
      ),
  ]


class ForwardModelsTest(unittest.TestCase):
  """Exercises code for `IndependentGaussianLikelihood` classes."""

  def setUp(self):
    super().setUp()
    # Set up simple test case.
    self.test_likelihood = DummyForwardModel()
    self.test_x = jnp.array([[0., 1., 2.]])
    self.test_y = jnp.array([[6., 7.5]])

  def test_get_measurement_shape(self):
    """Checks shape of measurement value for various classes."""
    batch_size = 8
    for params in test_cases():
      likelihood = params['likelihood']
      expected_y_dim = params['expected_y_dim']
      seed = params['seed']
      with self.subTest():
        rng = jax.random.PRNGKey(seed)

        x = jax.random.normal(
            rng, shape=(batch_size, *likelihood.image_shape))

        rng, next_rng = jax.random.split(rng)
        y = likelihood.get_measurement(next_rng, x)
        expected_meas_shape = (batch_size, expected_y_dim)
        self.assertTupleEqual(y.shape, expected_meas_shape)

  def test_get_noiseless_measurement(self):
    """Checks forward operator."""
    Ax = self.test_likelihood.apply_forward_operator(self.test_x)  # pylint:disable=invalid-name
    ref_Ax = np.array([[5, 8]])  # pylint:disable=invalid-name
    np.testing.assert_allclose(Ax, ref_Ax)

  def test_get_measurement(self):
    """Checks measurement value using test likelihood and data."""
    rng = jax.random.PRNGKey(0)
    y = self.test_likelihood.get_measurement(rng, self.test_x)

    ref_meas_noise = jax.random.normal(
        rng, shape=(1, 2)) * self.test_likelihood.sigmas
    ref_measurement = np.array([[5, 8]]) + ref_meas_noise
    np.testing.assert_allclose(y, ref_measurement)

  def test_unnormalized_log_likelihood(self):
    """Checks unnormalized log-likelihood value using test likelihood and data."""
    unnormalized_log_llh = self.test_likelihood.unnormalized_log_likelihood(
        self.test_x, self.test_y)
    ref_unnormalized_log_llh = -0.25
    self.assertAlmostEqual(unnormalized_log_llh[0], ref_unnormalized_log_llh)

  def test_likelihood_score(self):
    """Checks gradient of the unnormalized log-likelihood using test likelihood and data."""
    llh_score = self.test_likelihood.likelihood_score(self.test_x, self.test_y)
    ref_llh_score = np.array([[0.25, -0.75, -1]])
    np.testing.assert_allclose(llh_score, ref_llh_score)


if __name__ == '__main__':
  unittest.main()

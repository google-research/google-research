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
"""Tests for jax_dft.xc."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import constants
from jax_dft import xc

# Set the default dtype as float64
config.update('jax_enable_x64', True)


class XcTest(parameterized.TestCase):

  def setUp(self):
    super(XcTest, self).setUp()
    self.amplitude = constants.EXPONENTIAL_COULOMB_AMPLITUDE
    self.kappa = constants.EXPONENTIAL_COULOMB_KAPPA

  def test_exponential_coulomb_uniform_exchange_density(self):
    np.testing.assert_allclose(
        xc.exponential_coulomb_uniform_exchange_density(
            density=jnp.array([1e-15, 1e-10, 1e-5, 1., 2., 3., 20.])),
        [0., 0., -0.000013, -0.3983580525,
         -0.4512822906, -0.4732600516, -0.521973703], atol=1e-6)

  def test_exponential_coulomb_uniform_exchange_density_low_density_limit(self):
    density = jnp.linspace(0, 0.01, 5)
    y = jnp.pi * density / self.kappa
    np.testing.assert_allclose(
        xc.exponential_coulomb_uniform_exchange_density(density=density),
        self.amplitude / (2 * jnp.pi) * (-y + y ** 3 / 6), atol=1e-6)

  def test_exponential_coulomb_uniform_exchange_density_high_density_limit(
      self):
    limit_value = -self.amplitude / 2
    np.testing.assert_allclose(
        xc.exponential_coulomb_uniform_exchange_density(
            density=jnp.array([1000., 10000.])),
        [limit_value, limit_value], atol=1e-3)

  def test_exponential_coulomb_uniform_exchange_density_gradient(self):
    # The derivative of exchange energy density is
    # d(amplitude / (2 * pi) * (ln(1 + y ** 2) / y - 2 * arctan(y)))/d(density)
    # = amplitude / (2 * kappa) * d(ln(1 + y ** 2) / y - 2 * arctan(y))/dy
    # = -amplitude / (2 * kappa) * ln(1 + y ** 2) / y ** 2
    grad_fn = jax.vmap(
        jax.grad(xc.exponential_coulomb_uniform_exchange_density), in_axes=(0,))
    density = jnp.linspace(0, 3, 11)
    y = jnp.pi * density / self.kappa
    np.testing.assert_allclose(
        grad_fn(density),
        -self.amplitude / (2 * self.kappa) * jnp.log(1 + y ** 2) / y ** 2)

  def test_exponential_coulomb_uniform_correlation_density_low_density_limit(
      self):
    density = jnp.linspace(0, 0.005, 5)
    y = jnp.pi * density / self.kappa
    alpha = 2.
    beta = -1.00077
    np.testing.assert_allclose(
        xc.exponential_coulomb_uniform_correlation_density(density=density),
        -self.amplitude / (jnp.pi * alpha) * y * (
            1 - beta / alpha * jnp.sqrt(y)),
        atol=1e-3)

  def test_exponential_coulomb_uniform_correlation_density_high_density_limit(
      self):
    np.testing.assert_allclose(
        xc.exponential_coulomb_uniform_correlation_density(
            density=jnp.array([1000., 10000.])),
        [0., 0.], atol=1e-3)

  def test_exponential_coulomb_uniform_correlation_density_gradient(self):
    grad_fn = jax.vmap(
        jax.grad(xc.exponential_coulomb_uniform_correlation_density),
        in_axes=(0,))
    density = jnp.linspace(0, 3, 11)
    y = jnp.pi * density / self.kappa
    alpha = 2.
    beta = -1.00077
    gamma = 6.26099
    delta = -11.9041
    eta = 9.62614
    sigma = -1.48334
    nu = 1.
    denominator = (
        alpha + beta * jnp.sqrt(y)
        + gamma * y + delta * jnp.sqrt(y ** 3) + eta * y ** 2
        + sigma * jnp.sqrt(y ** 5)
        + nu * jnp.pi * self.kappa ** 2 / self.amplitude * y ** 3)
    y_ddenominator = (
        beta * 0.5 * y ** 0.5 + gamma * y + delta * 1.5 * y ** 1.5
        + eta * 2 * y ** 2 + sigma * 2.5 * y ** 2.5
        + nu * jnp.pi * self.kappa ** 2 / self.amplitude * 3 * y ** 3)
    analytical_gradient = -self.amplitude / self.kappa * (
        denominator - y_ddenominator) / denominator ** 2
    np.testing.assert_allclose(grad_fn(density), analytical_gradient)
    # Check gradient value at zero is -amplitude / kappa / alpha.
    self.assertAlmostEqual(
        jax.grad(xc.exponential_coulomb_uniform_correlation_density)(0.),
        -self.amplitude / self.kappa / alpha)

  @parameterized.parameters(1., 2, None, np.array([[1., 2., 3.]]))
  def test_get_lda_xc_energy_density_fn(self, params):
    # Different values of params shouldn't change the output of
    # lda_xc_energy_density_fn since params is a dummy argument.
    lda_xc_energy_density_fn = xc.get_lda_xc_energy_density_fn()
    np.testing.assert_allclose(
        lda_xc_energy_density_fn(density=jnp.array([0., 1.]), params=params),
        [0., -0.406069], atol=1e-5)


if __name__ == '__main__':
  absltest.main()

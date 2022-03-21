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

# pylint: disable=invalid-name
"""Tests for the orthogonal polynomials package."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import test_util as jtu
from scipy import integrate
from scipy import special

from sobolev import chebyshev
from sobolev import sobolev


class TestChebyshev(jtu.JaxTestCase):
  """Check the implementation of Chebyshev polynomials."""

  def test_chebyshev(self):
    xx = jnp.linspace(-5, 5)
    for i in range(10):
      poly0 = chebyshev.eval_chebyt(i, xx)
      poly1 = special.eval_chebyt(i, xx)
      self.assertAllClose(poly0, poly1, check_dtypes=True)

  def test_chebyshev_matrices(self):
    # test that it works on matrices too
    key = jax.random.PRNGKey(0)
    xx = jax.random.normal(key, shape=(5, 10))
    for i in range(10):
      poly0 = chebyshev.eval_chebyt(i, xx)
      poly1 = special.eval_chebyt(i, xx)
      self.assertAllClose(poly0, poly1, check_dtypes=True)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_2', 1, 2),
      ('2_3', 2, 3),
      ('3_4', 3, 4),
  )
  def test_chebyshev_orthogonality(self, i, j):
    low = -1
    high = 1

    def chebyshev_product(f, g):
      mu = lambda x: 1. / jnp.sqrt(1 - x**2)
      fun0 = lambda x: f(x) * g(x) * mu(x)
      val0 = integrate.quad(fun0, low, high, epsabs=1e-4)[0]
      return val0

    Si = lambda x: jnp.squeeze(chebyshev.eval_chebyt(i, x))
    Sj = lambda x: jnp.squeeze(chebyshev.eval_chebyt(j, x))
    self.assertAlmostEqual(chebyshev_product(Si, Sj), 0., places=3)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_2', 1, 2),
      ('2_3', 2, 3),
      ('3_4', 3, 4),
  )
  def test_chebyshev_residual_orthogonality(self, i, j):
    """Test orthogonality for Chebyshev residual polynomials."""
    low, high = 0.1, 2.0

    def sigma(x):
      return 2 * x / (high - low) - (high + low) / (high - low)

    dsigma = 2 / (high - low)

    def chebyshev_product(f, g):
      mu = lambda x: dsigma / jnp.sqrt(1 - sigma(x)**2)
      fun0 = lambda x: f(x) * g(x) * mu(x)
      val0 = integrate.quad(fun0, low, high, epsabs=1e-4)[0]
      return val0

    def Si(x):
      z = chebyshev.eval_chebyt(i, x, low, high, normalization='residual')
      return jnp.squeeze(z)

    def Sj(x):
      z = chebyshev.eval_chebyt(j, x, low, high, normalization='residual')
      return jnp.squeeze(z)

    self.assertAlmostEqual(chebyshev_product(Si, Sj), 0., places=3)


class TestSobolevChebyshev(jtu.JaxTestCase):
  """Tests for Sobolev-Chebyshev polynomials."""

  def test_sobolev_residual(self):
    low, high = 0.1, 2.0
    for degree in range(10):
      out = jnp.sum(sobolev.eval_sobolev_chebyt(degree, 0., low, high))
      self.assertAlmostEqual(jax.device_get(out), 1.0, places=6)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_2', 1, 2),
      ('2_3', 2, 3),
      ('3_4', 3, 4),
  )
  def test_sobolev_orthogonality(self, i, j):
    low, high = 0.1, 2.0

    def sigma(x):
      return 2 * x / (high - low) - (high + low) / (high - low)

    dsigma = 2 / (high - low)

    def sobolev_product(f, g):
      mu = lambda x: dsigma / jnp.sqrt(1 - sigma(x)**2)
      fun0 = lambda x: f(x) * g(x) * mu(x)
      fun1 = lambda x: jax.grad(f)(x) * jax.grad(g)(x) * mu(x)
      val0 = integrate.quad(fun0, low, high, epsabs=1e-4)[0]
      val1 = integrate.quad(fun1, low, high, epsabs=1e-4)[0]
      return val0 + val1

    Si = lambda x: jnp.squeeze(sobolev.eval_sobolev_chebyt(i, x, low, high))
    Sj = lambda x: jnp.squeeze(sobolev.eval_sobolev_chebyt(j, x, low, high))
    self.assertAlmostEqual(sobolev_product(Si, Sj), 0., places=3)


if __name__ == '__main__':
  absltest.main()

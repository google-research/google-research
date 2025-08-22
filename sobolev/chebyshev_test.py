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

"""Tests for the orthogonal polynomials package."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from scipy import integrate
from scipy import special

from sobolev import chebyshev


class TestChebyshev(parameterized.TestCase):
  """Check the implementation of Chebyshev polynomials."""

  def test_chebyshev(self):
    xx = jnp.linspace(-5, 5)
    for i in range(10):
      poly0 = chebyshev.eval_chebyt(i, xx)
      poly1 = special.eval_chebyt(i, xx)
      np.testing.assert_allclose(poly0, poly1, rtol=1E-5)

  def test_chebyshev_recurrence(self):
    # test default values and make sure it gives the same result
    # with float and integer argument
    recurrence = chebyshev.recurrence_chebyt()
    recurrence2 = chebyshev.recurrence_chebyt(low=-1, high=1)
    np.testing.assert_allclose(next(recurrence), next(recurrence2))

  def test_chebyshev_recurrence_api(self):
    # test that it raises an exception for shifted Chebyshev polynomials
    bad_limits = ((-1, 2), (-2, 1), (-1, 1.1))
    for (low, high) in bad_limits:
      with np.testing.assert_raises(NotImplementedError):
        recurrence = chebyshev.recurrence_chebyt(low=low, high=high)
        next(recurrence)

  def test_chebyshev_matrices(self):
    # test that it works on matrices too
    key = jax.random.PRNGKey(0)
    xx = jax.random.normal(key, shape=(5, 10))
    for i in range(10):
      poly0 = chebyshev.eval_chebyt(i, xx)
      poly1 = special.eval_chebyt(i, xx)
      np.testing.assert_allclose(poly0, poly1, rtol=1E-4)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_2', 1, 2),
      ('2_3', 2, 3),
      ('3_4', 3, 4),
  )
  def test_chebyshev_orthogonality(self, i, j):
    """Test orthogonality for Chebyshev polynomials."""
    low = -1
    high = 1

    def chebyshev_product(f, g):
      mu = lambda x: 1. / jnp.sqrt(1 - x**2)
      fun0 = lambda x: f(x) * g(x) * mu(x)
      val0 = integrate.quad(fun0, low, high, epsabs=1e-4)[0]
      return val0

    chebyt_i = lambda x: jnp.squeeze(chebyshev.eval_chebyt(i, x))
    chebyt_j = lambda x: jnp.squeeze(chebyshev.eval_chebyt(j, x))

    self.assertAlmostEqual(chebyshev_product(chebyt_i, chebyt_j), 0., places=3)

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

    def chebyt_i(x):
      image = chebyshev.eval_chebyt(i, x, low, high, normalization='residual')
      return jnp.squeeze(image)
    def chebyt_j(x):
      image = chebyshev.eval_chebyt(j, x, low, high, normalization='residual')
      return np.squeeze(image)

    self.assertAlmostEqual(chebyshev_product(chebyt_i, chebyt_j), 0., places=3)


if __name__ == '__main__':
  absltest.main()

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
from scipy import integrate

from sobolev import sobolev


class TestSobolevChebyshev(parameterized.TestCase):
  """Tests for Sobolev-Chebyshev polynomials."""

  def test_schebyt_residual(self):
    low, high = 0.1, 2.0
    for degree in range(10):
      out = jnp.sum(sobolev.eval_schebyt(degree, 0., low, high))
      self.assertAlmostEqual(jax.device_get(out), 1.0, places=6)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_2', 1, 2),
      ('2_3', 2, 3),
      ('3_4', 3, 4),
  )
  def test_schebyt_orthogonality(self, i, j):
    """Test orthogonality for Sobolev-Chebyshev polynomials."""
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

    schebyt_i = lambda x: jnp.squeeze(sobolev.eval_schebyt(i, x, low, high))
    schebyt_j = lambda x: jnp.squeeze(sobolev.eval_schebyt(j, x, low, high))
    self.assertAlmostEqual(sobolev_product(schebyt_i, schebyt_j), 0., places=3)


if __name__ == '__main__':
  absltest.main()

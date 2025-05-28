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

"""Tests for isotonic_pav."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from sparse_soft_topk._src import isotonic_pav

LS = (1e-2, 1e-1, 1.0)
PS = (4 / 3, 3 / 2, 2.0)


def proj(u, _):
  # Project all coordinates on the non-negative orthant, except the first one!
  u_new = jax.nn.relu(u)
  u_new = u_new.at[0].set(u[0])
  return u_new


def isotonic_pg(y, w, l=1.0, p=2.0, increasing=False, maxiter=1000, tol=1e-5):
  q = p / (p - 1)
  sign = 1 if increasing else -1

  def fun(z, y):
    x = jnp.cumsum(z) * sign
    return jnp.sum(((jnp.absolute(y - x)) ** q) / q + w * x * l ** (q - 1))

  pg = jaxopt.ProjectedGradient(
      fun=fun, projection=proj, maxiter=maxiter, tol=tol
  )
  init = jnp.zeros_like(y)
  sol = pg.run(init, y=y).params
  return jnp.cumsum(sol) * sign


def isotonic_mag_pg(
    y, w, l=1.0, p=2.0, increasing=False, maxiter=1000, tol=1e-5
):
  q = p / (p - 1)
  sign = 1 if increasing else -1

  def fun(z, y):
    x = jnp.cumsum(z) * sign
    return jnp.sum(
        ((jnp.absolute(y - x)) ** q) / q + 0.5 * w * (x**2) * l ** (q - 1)
    )

  pg = jaxopt.ProjectedGradient(
      fun=fun, projection=proj, maxiter=maxiter, tol=tol
  )
  init = jnp.zeros_like(y)
  sol = pg.run(init, y=y).params
  return jnp.cumsum(sol) * sign


class IsotonicMaskPavTest(parameterized.TestCase):
  """

  """

  @parameterized.parameters(itertools.product(LS, PS))
  def test_output(self, l, p):
    """Checks output shape."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(100))
    w = jnp.array(seed.rand(100))
    z = isotonic_pav.isotonic_mask_pav(y, w, l=l, p=p)
    self.assertAlmostEqual(z.shape, y.shape)

  @parameterized.parameters(itertools.product(LS, PS))
  def test_compare_with_pg(self, l, p, thresh=1e-5):
    """Comparing the output with projected gradient descent."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.rand(10))
    w = jnp.array(seed.rand(10))
    q = p / (p - 1)
    bisect_max_iter = 100
    out_pg = isotonic_pg(y, w, l=l, p=p)
    out_pav = isotonic_pav.isotonic_mask_pav(
        y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )
    min_pg = jnp.sum(
        ((jnp.absolute(y - out_pg)) ** q) / q + w * out_pg * l ** (q - 1)
    )
    min_pav = jnp.sum(
        ((jnp.absolute(y - out_pav)) ** q) / q + w * out_pav * l ** (q - 1)
    )
    assert min_pav <= min_pg or ((out_pg - out_pav) ** 2).mean() < thresh

  @parameterized.parameters(itertools.product(LS, PS))
  def test_jvp(self, l, p, eps=1e-3):
    """Compare the jvp with finite differences."""
    seed = np.random.RandomState(0)
    n = 20
    y = jnp.array(seed.randn(n))
    w = jnp.array(seed.rand(n))
    v = jnp.array(seed.randn(n))
    bisect_max_iter = 100
    jvp = (
        jax.jacobian(isotonic_pav.isotonic_mask_pav)(
            y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
        )
        @ v
    )
    approx = (
        isotonic_pav.isotonic_mask_pav(
            y + eps * v, w, l=l, p=p, bisect_max_iter=bisect_max_iter
        )
        - isotonic_pav.isotonic_mask_pav(
            y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
        )
    ) / eps
    self.assertSequenceAlmostEqual(jvp, approx, places=2)

  @parameterized.parameters(itertools.product(LS, PS))
  def test_multi_dimentional_input(self, l, p, n_features=5, n_batches=3):
    """Checks the vmaping."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n_batches, n_features))
    w = jnp.array(seed.rand(n_features))
    z = isotonic_pav.isotonic_mask_pav(y, w, l=l, p=p)
    for i in range(n_batches):
      self.assertSequenceAlmostEqual(
          z[i], isotonic_pav.isotonic_mask_pav(y[i], w, l=l, p=p), places=6
      )

  @parameterized.parameters(itertools.product(LS, PS))
  def test_multi_dimentional_grad(self, l, p, n_features=5, n_batches=3):
    """Checks the vmaping."""
    seed = np.random.RandomState(0)

    def predict(s):
      return (
          isotonic_pav.isotonic_mask_pav(s, w, l=l, p=p, bisect_max_iter=50)
          + isotonic_pav.isotonic_mask_pav(
              s**2, w, l=l, p=p, bisect_max_iter=50
          )
      ).sum()

    s = jnp.array(seed.randn(n_batches, n_features))
    w = jnp.array(seed.rand(n_features))
    z = jax.grad(predict)(s)
    for i in range(n_batches):
      self.assertSequenceAlmostEqual(z[i], jax.grad(predict)(s[i]), places=6)


class IsotonicMagPavTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(LS, PS))
  def test_output(self, l, p):
    """Checks output shape."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(100))
    w = jnp.array(seed.rand(100))
    z = isotonic_pav.isotonic_mag_pav(y, w, l=l, p=p)
    self.assertAlmostEqual(z.shape, y.shape)

  @parameterized.parameters(itertools.product(LS, PS))
  def test_compare_with_pg(self, l, p, thresh=1e-5):
    """Comparing the output with projected gradient descent."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.rand(10))
    w = jnp.array(seed.rand(10))
    q = p / (p - 1)
    bisect_max_iter = 100
    out_pg = isotonic_mag_pg(y, w, l=l, p=p)
    out_pav = isotonic_pav.isotonic_mag_pav(
        y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
    )
    min_pg = jnp.sum(
        ((jnp.absolute(y - out_pg)) ** q) / q
        + 0.5 * w * out_pg**2 * l ** (q - 1)
    )
    min_pav = jnp.sum(
        ((jnp.absolute(y - out_pav)) ** q) / q
        + 0.5 * w * out_pav**2 * l ** (q - 1)
    )
    assert min_pav <= min_pg or ((out_pg - out_pav) ** 2).mean() < thresh

  @parameterized.parameters(itertools.product(LS, PS))
  def test_jvp(self, l, p, eps=1e-3):
    """Compare the jvp with finite differences."""
    seed = np.random.RandomState(0)
    n = 20
    y = jnp.array(seed.randn(n))
    w = jnp.array(seed.rand(n))
    v = jnp.array(seed.randn(n))
    bisect_max_iter = 100
    jvp = (
        jax.jacobian(isotonic_pav.isotonic_mag_pav)(
            y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
        )
        @ v
    )
    approx = (
        isotonic_pav.isotonic_mag_pav(
            y + eps * v, w, l=l, p=p, bisect_max_iter=bisect_max_iter
        )
        - isotonic_pav.isotonic_mag_pav(
            y, w, l=l, p=p, bisect_max_iter=bisect_max_iter
        )
    ) / eps
    self.assertSequenceAlmostEqual(jvp, approx, places=2)

  @parameterized.parameters(itertools.product(LS, PS))
  def test_multi_dimentional_input(self, l, p, n_features=5, n_batches=3):
    """Checks the vmaping."""
    seed = np.random.RandomState(0)
    y = jnp.array(seed.randn(n_batches, n_features))
    w = jnp.array(seed.rand(n_features))
    z = isotonic_pav.isotonic_mag_pav(y, w, l=l, p=p)
    for i in range(n_batches):
      self.assertSequenceAlmostEqual(
          z[i], isotonic_pav.isotonic_mag_pav(y[i], w, l=l, p=p), places=6
      )


class UtilsTest(parameterized.TestCase):

  def test_solver(self):
    """Checks the cubic solver."""
    # Test case h > 0
    a, b, c, d = 4.0, -3.0, 1.0, 2.0
    sol_np = np.real(np.roots([a, b, c, d])[-1])
    sol_ours = isotonic_pav._solve_real_root(a, b, c, d)
    self.assertAlmostEqual(sol_np, sol_ours, places=4)
    # Test case h = 0
    a, b, c, d = 1.0, 0.0, 0.0, 0.0
    sol_np = np.real(np.roots([a, b, c, d])[-1])
    sol_ours = isotonic_pav._solve_real_root(a, b, c, d)
    self.assertAlmostEqual(sol_np, sol_ours, places=4)
    # Test case h > 0
    a, b, c, d = 40.0, -30.0, -1.0, 2.0
    sol_np = np.real(np.roots([a, b, c, d])[0])
    sol_ours = isotonic_pav._solve_real_root(a, b, c, d)
    self.assertAlmostEqual(sol_np, sol_ours, places=4)


if __name__ == "__main__":
  absltest.main()

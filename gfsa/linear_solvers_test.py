# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for gfsa.linear_solvers."""

import functools
from absl.testing import absltest
import jax
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np
from gfsa import linear_solvers


class LinearSolversTest(absltest.TestCase):

  def test_richardson_solve(self):
    """Check that richardson_solve produces correct outputs and derivatives."""

    # Ensure we converge to the fixed point
    matrix = jax.random.normal(jax.random.PRNGKey(0), (50, 50))
    matrix = jnp.eye(50) - 0.9 * matrix / jnp.sum(
        jnp.abs(matrix), axis=0, keepdims=True)
    b = jax.random.normal(jax.random.PRNGKey(1), (50,))

    def iter_solve(matrix, b):
      return linear_solvers.richardson_solve(
          lambda x: matrix @ x, b, iterations=100)

    # Correct output
    jtu.check_close(
        iter_solve(matrix, b), jax.scipy.linalg.solve(matrix, b), rtol=1e-4)

    # Correct jvp
    dmatrix = jax.random.normal(jax.random.PRNGKey(2), (50, 50))
    db = jax.random.normal(jax.random.PRNGKey(3), (50,))
    jtu.check_close(
        jax.jvp(iter_solve, (matrix, b), (dmatrix, db)),
        jax.jvp(jax.scipy.linalg.solve, (matrix, b), (dmatrix, db)),
        rtol=1e-4)

    # Correct vjp
    co_x = jax.random.normal(jax.random.PRNGKey(3), (50,))
    jtu.check_close(
        jax.vjp(iter_solve, matrix, b)[1](co_x),
        jax.vjp(jax.scipy.linalg.solve, matrix, b)[1](co_x),
        rtol=1e-4)

  def test_richardson_solve_structured(self):
    """Check that richardson_solve works on pytrees."""

    def structured_matvec(m1, m2, m3, xs):
      x1, x2 = xs
      b1 = m1 @ x1
      b2 = {
          "foo": m2 @ x2["foo"],
          "bar": m3 @ x2["bar"],
      }
      return b1, b2

    def structured_direct_solve(m1, m2, m3, b1, b2):
      x1 = jax.scipy.linalg.solve(m1, b1)
      x2 = {
          "foo": jax.scipy.linalg.solve(m2, b2["foo"]),
          "bar": jax.scipy.linalg.solve(m3, b2["bar"]),
      }
      return x1, x2

    def structured_iter_solve(m1, m2, m3, b1, b2):
      return linear_solvers.richardson_solve(
          functools.partial(structured_matvec, m1, m2, m3), (b1, b2),
          iterations=200)

    def mk_mat(key):
      matrix = jax.random.normal(jax.random.PRNGKey(key), (50, 50))
      return jnp.eye(50) - 0.9 * matrix / jnp.sum(
          jnp.abs(matrix), axis=0, keepdims=True)

    m1 = mk_mat(0)
    m2 = mk_mat(1)
    m3 = mk_mat(2)
    b1 = jax.random.normal(jax.random.PRNGKey(3), (50,))
    b2 = {
        "foo": jax.random.normal(jax.random.PRNGKey(4), (50,)),
        "bar": jax.random.normal(jax.random.PRNGKey(5), (50,)),
    }
    jtu.check_close(
        structured_iter_solve(m1, m2, m3, b1, b2),
        structured_direct_solve(m1, m2, m3, b1, b2))

  def test_richardson_solve_precision(self):
    """Check that precision is maintained even across many orders of magnitude."""
    matrix = (jnp.eye(50) - jnp.diag(jnp.full([49], 0.1), -1)).astype("float32")
    b = jnp.array([1.] + [0.] * 49, dtype="float32")
    x_est = linear_solvers.richardson_solve(
        lambda x: matrix @ x, b, iterations=50)
    x_expected = jnp.power(10., -jnp.arange(50)).astype("float32")

    # Both the closed-form and estimated values should round off to zero at the
    # same time; the only limit should be the resolution of float32 and not
    # any inaccuracies during the solve.
    np.testing.assert_allclose(x_est, x_expected, atol=0, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()

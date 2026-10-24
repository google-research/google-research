# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for matrix_inverse_pth_root_eigh in distributed_shampoo.

`matrix_inverse_pth_root_eigh` used to build the result twice: once as
`u @ diag(inv_e) @ u.T` and then again as `root @ root.T`, with the first value
immediately overwritten. The first form materialises an n-by-n diagonal and
performs a full O(n^3) matmul whose result is discarded, on every call.
"""

import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from scalable_shampoo.optax import distributed_shampoo


def _spd_matrix(n, seed=0, scale=1.0):
  rng = np.random.default_rng(seed)
  a = rng.normal(size=(n, n)).astype(np.float32)
  return jnp.asarray(scale * (a @ a.T + n * np.eye(n, dtype=np.float32)))


class MatrixInversePthRootTest(absltest.TestCase):

  def test_matches_reference_from_eigendecomposition(self):
    """The result equals u @ diag(inv_e) @ u.T, the form that was discarded."""
    matrix = _spd_matrix(6)
    p = 2.0
    ridge_epsilon = 1e-4

    actual, _ = distributed_shampoo.matrix_inverse_pth_root_eigh(
        matrix, p, ridge_epsilon=ridge_epsilon
    )

    regularized = matrix + ridge_epsilon * jnp.eye(matrix.shape[0])
    e, u = jnp.linalg.eigh(regularized)
    inv_e = jnp.power(jnp.maximum(e, ridge_epsilon), -1.0 / p)
    expected = u @ jnp.diag(inv_e) @ u.T

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  def test_matches_brute_force_inverse(self):
    """For p=1 the result is the regularized inverse itself."""
    matrix = _spd_matrix(5, seed=1)
    actual, _ = distributed_shampoo.matrix_inverse_pth_root_eigh(
        matrix, 1.0, ridge_epsilon=1e-3
    )
    regularized = matrix + 1e-3 * jnp.eye(matrix.shape[0])
    expected = jnp.linalg.inv(regularized)
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  def test_the_two_forms_are_equal(self):
    """The discarded expression and the kept one agree, per column scaling.

    This documents why removing the first `val` line cannot change results:
    u @ diag(inv_e) @ u.T == (u * sqrt(inv_e)) @ (u * sqrt(inv_e)).T
    """
    matrix = _spd_matrix(4, seed=2)
    e, u = jnp.linalg.eigh(matrix)
    inv_e = jnp.power(jnp.maximum(e, 1e-6), -0.5)

    mm = functools.partial(jnp.matmul)
    long_form = mm(mm(u, jnp.diag(inv_e)), u.T)
    root = u * jnp.sqrt(inv_e)
    short_form = mm(root, root.T)

    np.testing.assert_allclose(long_form, short_form, rtol=1e-5, atol=1e-5)

  def test_output_is_symmetric_and_finite(self):
    matrix = _spd_matrix(8, seed=3)
    actual, metrics = distributed_shampoo.matrix_inverse_pth_root_eigh(
        matrix, 4.0, ridge_epsilon=1e-6
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))))
    np.testing.assert_allclose(actual, actual.T, rtol=1e-4, atol=1e-4)
    self.assertTrue(bool(jnp.isfinite(
        metrics.inverse_pth_root_errors)))


class NoRedundantProductTest(absltest.TestCase):
  """Guards against reintroducing the discarded matmul."""

  def test_result_is_not_computed_twice(self):
    """`inv_e` must be applied once, via the column-scaled `root` form."""
    import inspect  # pylint: disable=g-import-not-at-top

    source = inspect.getsource(
        distributed_shampoo.matrix_inverse_pth_root_eigh
    )
    # The discarded form materialised a full n-by-n diagonal.
    self.assertNotIn(
        'jnp.diag(inv_e)',
        source,
        'matrix_inverse_pth_root_eigh builds the result from a diagonal '
        'again; `root = u * jnp.sqrt(inv_e); root @ root.T` is the same '
        'value with one fewer O(n^3) matmul.',
    )


if __name__ == '__main__':
  absltest.main()

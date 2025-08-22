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

"""Tests for distributed_shampoo."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from scalable_shampoo.optax import distributed_shampoo


class PaddingTest(parameterized.TestCase):

  def assertAllClose(self, x, y, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoPadding',
          'max_size': 3,
          'result': [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
      },
      {
          'testcase_name':
              'Padding',
          'max_size':
              5,
          'result': [[1., 1., 1., 0., 0.], [1., 1., 1., 0., 0.],
                     [1., 1., 1., 0., 0.], [0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 1.]],
      },
  )
  def test_pad_square_matrix(self, max_size, result):
    self.assertAllClose(
        distributed_shampoo.pad_square_matrix(
            mat=jnp.ones(shape=(3, 3), dtype=jnp.float32), max_size=max_size),
        jnp.asarray(result, dtype=jnp.float32))

  @parameterized.named_parameters(
      {
          'testcase_name': 'TooLarge',
          'shape': (3, 3),
          'max_size': 2
      },
      {
          'testcase_name': 'NotSquare',
          'shape': (3, 4),
          'max_size': 5
      },
  )
  def test_pad_square_matrix_error(self, shape, max_size):
    with self.assertRaises(ValueError):
      distributed_shampoo.pad_square_matrix(
          mat=jnp.ones(shape=shape), max_size=max_size)


def _pth_root_difference_cases():
  """Returns cases for _pth_root_difference() test."""
  cases = []
  # The test checks accuracy of
  # (w + a)^(-1/p) - (w + b)^(-1/p)
  # so generate corresponding parameters.
  p_vals = [2, 4, 6, 8]
  a_vals = b_vals = [1e-6, 1e-5, 0.0, 1.0]
  w_vals = [1e-6, 1e-5, 1.0, 1e3]
  for p, a, b, w in itertools.product(p_vals, a_vals, b_vals, w_vals):
    cases.append({'p': p, 'a': a, 'b': b, 'w': w})
  return cases


class DistributedShampooTest(chex.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([[1., 3.],
                                   [2., 4.]]), jnp.array([[3., 4.], [3., 4.]]))
    self.per_step_updates = (jnp.array([[500., 5.], [500., 5.]]),
                             jnp.array([[300., 3.], [300., 3.]]))
    self.per_step_updates_custom_preconditioner = (self.per_step_updates,
                                                   (jnp.array([[200., 4.],
                                                               [200., 4.]]),
                                                    jnp.array([[600., 2.],
                                                               [600., 2.]])))
    self.rng = np.random.default_rng(1234)
    shape = ([2, 5], [6, 3])
    dt = self.init_params[0].dtype

    def make_shape(bigger_first_entry):
      x = tuple(self.rng.standard_normal(size=s) for s in shape)
      if bigger_first_entry:
        for xx in x:
          xx[Ellipsis, 0] *= 100
      return tuple(jnp.array(xx).astype(dt) for xx in x)

    self.init_params_larger = make_shape(False)
    self.per_step_updates_larger = make_shape(True)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      {
          'testcase_name': 'default',
          'best_effort_memory_usage_reduction': True,
          'expected_value': -0.57,
      },
      {
          'testcase_name': 'default_nomerge',
          'best_effort_memory_usage_reduction': True,
          'merge_small_dims_block_size': 1,
          'expected_value': -0.57,
      },
      {
          'testcase_name': 'default_larger',
          'best_effort_memory_usage_reduction': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'default_larger_nomerge',
          'best_effort_memory_usage_reduction': True,
          'slightly_larger': True,
          'merge_small_dims_block_size': 1,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'materialize_statistics',
          'best_effort_memory_usage_reduction': True,
      },
      {
          'testcase_name': 'blocked_statistics',
          'best_effort_memory_usage_reduction': True,
      },
      {
          'testcase_name': 'default_quantized',
      },
      {
          'testcase_name': 'materialize_statistics_quantized',
      },
      {
          'testcase_name': 'blocked_statistics_quantized',
      },
      {
          'testcase_name': 'no_training_metrics',
          'generate_training_metrics': False,
      },
      {
          'testcase_name': 'larger_reuse',
          'best_effort_memory_usage_reduction': True,
          'reuse_preconditioner': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'larger_reuse_highmem',
          'best_effort_memory_usage_reduction': False,
          'reuse_preconditioner': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'larger_reuse_highmem_nomerge',
          'best_effort_memory_usage_reduction': False,
          'merge_small_dims_block_size': 1,
          'reuse_preconditioner': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
  )
  def test_distributed_shampoo(
      self,
      best_effort_memory_usage_reduction=False,
      merge_small_dims_block_size=4096,
      generate_training_metrics=True,
      slightly_larger=False,
      expected_value=None,
      reuse_preconditioner=False,
  ):
    params = self.init_params_larger if slightly_larger else self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name='batch',
        preconditioning_compute_steps=2,
        best_effort_memory_usage_reduction=best_effort_memory_usage_reduction,
        merge_small_dims_block_size=merge_small_dims_block_size,
        generate_training_metrics=generate_training_metrics,
        reuse_preconditioner=reuse_preconditioner,
    )
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    if slightly_larger:
      updates = self.per_step_updates_larger
    else:
      updates = self.per_step_updates

    def _update(unused_batch):
      return transform_fn(updates, state, params)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))
    if expected_value is not None:
      last_entry = updates[1][-1, -1, -1]
      self.assertLess(
          abs(last_entry - expected_value),
          1e-4,
          msg=f'{last_entry=}, {expected_value=}')
    for _ in range(5):
      updates, state = pmap_fn(jnp.array([1.0]))
      chex.assert_tree_all_finite((params, updates, state))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters([
      {
          'testcase_name': 'default',
      },
      {
          'testcase_name': 'no_training_metrics',
          'generate_training_metrics': False,
      },
  ])
  def test_distributed_shampoo_no_pmap(self, generate_training_metrics=True):
    params = self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name=None,
        preconditioning_compute_steps=2,
        generate_training_metrics=generate_training_metrics)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)
    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    updates, state = transform_fn(self.per_step_updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))

  def _gen_symmetrix_matrix(self, dim, condition_number):
    u = scipy.stats.ortho_group.rvs(
        dim=dim, random_state=self.rng).astype(np.float64)
    v = u.T
    diag = np.diag([condition_number**(-i / (dim - 1)) for i in range(dim)])
    return u @ diag @ v

  def test_matrix_inverse_root(self):
    """Test for matrix inverse pth root."""

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10**e
      ms = self._gen_symmetrix_matrix(16, condition_number)
      self.assertLess(
          np.abs(np.linalg.cond(ms) - condition_number),
          condition_number * 0.01)
      metrics = distributed_shampoo.matrix_inverse_pth_root(
          ms.astype(np.float32), 4, ridge_epsilon=1e-12)[1]
      error = metrics.inverse_pth_root_errors
      if e < 7:
        self.assertLess(error, 0.1)
      else:
        # No guarantee of success after e >= 7
        pass

  @parameterized.parameters([{'sz': sz} for sz in [4, 32]])
  def test_matrix_inverse_root_padding(self, sz):
    """Test padding does not affect result much."""

    # Note sz == 1 case will not pass tests here b/c the method
    # is exact for scalars (but padding triggers coupled iteration).

    condition_number = 1e3
    ms = self._gen_symmetrix_matrix(sz, condition_number).astype(np.float32)

    # Shift matrix norm down by some large factor, so that improper padding
    # handling results in an error by increasing the condition number.
    ms = jnp.array(ms) * 1e-3

    rt, metrics = distributed_shampoo.matrix_inverse_pth_root(
        ms, 4, ridge_epsilon=1e-3)
    err = metrics.inverse_pth_root_errors
    pad_ms = distributed_shampoo.pad_square_matrix(ms, sz * 2)
    pad_rt, metrics = distributed_shampoo.matrix_inverse_pth_root(
        pad_ms, 4, ridge_epsilon=1e-3, padding_start=sz)
    pad_err = metrics.inverse_pth_root_errors
    pad_rt_principal = pad_rt[:sz, :sz]
    np.testing.assert_allclose(
        rt,
        pad_rt_principal,
        # The fact that this is so large keeps vladf up at night,
        # but without padding_start argument it's even worse (>1).
        rtol=1e-2,
        err_msg=np.array2string(rt - pad_rt_principal))
    self.assertLessEqual(pad_err, 4 * err)
    self.assertEqual(np.abs(pad_rt[sz:]).sum(), 0)
    self.assertEqual(np.abs(pad_rt[:, sz:]).sum(), 0)

  def test_all_padding(self):
    """Test full padding matrix."""
    empty = jnp.zeros([0, 0])
    padded = distributed_shampoo.pad_square_matrix(empty, 10)
    rt, metrics = distributed_shampoo.matrix_inverse_pth_root(
        padded, 4, ridge_epsilon=1e-3, padding_start=0)
    err = metrics.inverse_pth_root_errors
    self.assertEqual(np.abs(rt).sum(), 0.0)
    self.assertEqual(np.abs(err).sum(), 0.0)

  def _make_pth_diff_message(self, w, alpha, beta, p):
    left = f'({w} + {alpha})^(-1.0 / {p}) - '
    right = f'({w} + {beta})^(-1.0 / {p})'
    return left + right

  @parameterized.parameters(_pth_root_difference_cases())
  def test_pth_root_difference(self, p, a, b, w):
    """Test stable difference computation."""
    pth_rt_diff = jax.jit(
        functools.partial(distributed_shampoo._pth_root_difference, p=p))
    actual = pth_rt_diff(w, a, b)
    # in float64
    exp = (-1.0 / p)
    expected = (w + a)**exp - (w + b)**exp

    self.assertAlmostEqual(
        actual,
        expected,
        msg=self._make_pth_diff_message(w, a, b, p),
        delta=1e-2)

  @parameterized.parameters([{'p': p} for p in [2, 4, 8]])
  def test_lobpcg_preconditioning(self, p):
    """Checks that root calculation is valid with top-k preconditioning."""
    rng = np.random.RandomState(seed=42)
    n = 11
    epsilon = jnp.float32(1e-4)
    a_asymm = jnp.array(rng.random((n, n)), jnp.float32)
    a = jnp.matmul(a_asymm.T, a_asymm, precision=jax.lax.Precision.HIGHEST)
    log2 = (p - 1).bit_length()
    assert 2**log2 == p, (p, log2)

    root = functools.partial(
        distributed_shampoo.matrix_inverse_pth_root, ridge_epsilon=epsilon, p=p)
    root_lobpcg = functools.partial(
        root, lobpcg_topk_precondition=2, lobpcg_max_iter=10)

    methods = {'default': root, 'precond': root_lobpcg}
    spectrum_err, entry_err = {}, {}
    for k, method in methods.items():
      rt = jax.jit(method)(a)[0]

      # Recover the inverse by repeated squaring of inverse p-th root.
      inv = np.asarray(rt).astype(np.float64)
      for _ in range(log2):
        inv = inv.dot(inv)

      approx_id = inv.dot(a)
      spectrum = np.linalg.eigvalsh(approx_id)
      spectrum_err[k] = np.abs(1 - spectrum)
      entry_err[k] = np.mean(np.abs(approx_id - np.eye(n)))

    with np.printoptions(precision=2):

      def print_dict(d):
        return '\n'.join(f'{k} {v}' for k, v in d.items())

      err_msg = (f'p={p} log2(p)={log2}\n'
                 f'spectrum error\n{print_dict(spectrum_err)}\n'
                 f'entry_err\n{print_dict(entry_err)}')

      self.assertLessEqual(
          np.median(spectrum_err['precond']),
          2 * np.median(spectrum_err['default']),
          msg=err_msg)

      self.assertLessEqual(
          entry_err['precond'], entry_err['default'] * 2, msg=err_msg)


if __name__ == '__main__':
  absltest.main()

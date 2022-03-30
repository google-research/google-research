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

"""Tests for distributed_shampoo."""

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

  @parameterized.named_parameters(
      {
          'testcase_name':
              'LastTwoBlockRows',
          'starting_block':
              2,
          'result': [[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]
      },
      {
          'testcase_name':
              'LastBlockRow',
          'starting_block':
              3,
          'result': [[0., 0., 0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 1.]]
      },
      {
          'testcase_name': 'Empty',
          'starting_block': 4,
          'result': [[], []],
      },
  )
  def test_make_sliced_padding(self, starting_block, result):
    self.assertAllClose(
        distributed_shampoo.make_sliced_padding(
            symmetric_block_size=2,
            num_blocks=4,
            starting_block=starting_block,
            dtype=jnp.float32), jnp.asarray(result, dtype=jnp.float32))

  @parameterized.parameters(
      {
          'shape': (0, 0),
          'result': [[1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                     [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.]]
      },
      {
          'shape': (2, 2),
          'result': [[1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                     [1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.]]
      },
      {
          'shape': (2, 2 * 3),
          'result': [[1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0.],
                     [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1.]]
      },
      {
          'shape': (2, 2 * 6),
          'result': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
      },
  )
  def test_pad_block_symmetric_matrix(self, shape, result):
    self.assertAllClose(
        distributed_shampoo.pad_block_symmetric_matrix(
            mat=jnp.ones(shape=shape, dtype=jnp.float32),
            symmetric_block_size=2,
            max_num_blocks=6), jnp.asarray(result, dtype=jnp.float32))


class DistributedShampooTest(chex.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([[1., 3.],
                                   [2., 4.]]), jnp.array([[3., 4.], [3., 4.]]))
    self.per_step_updates = (jnp.array([[500., 5.], [500., 5.]]),
                             jnp.array([[300., 3.], [300., 3.]]))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      {
          'testcase_name': 'default',
          'best_effort_memory_usage_reduction': True,
          'symmetric_block_size': None,
          'block_statistics': False
      },
      {
          'testcase_name': 'materialize_statistics',
          'best_effort_memory_usage_reduction': True,
          'symmetric_block_size': 2,
          'block_statistics': False
      },
      {
          'testcase_name': 'blocked_statistics',
          'best_effort_memory_usage_reduction': True,
          'symmetric_block_size': 2,
          'block_statistics': True
      },
      {
          'testcase_name': 'default_quantized',
          'best_effort_memory_usage_reduction': False,
          'symmetric_block_size': None,
          'block_statistics': False
      },
      {
          'testcase_name': 'materialize_statistics_quantized',
          'best_effort_memory_usage_reduction': False,
          'symmetric_block_size': 2,
          'block_statistics': False
      },
      {
          'testcase_name': 'blocked_statistics_quantized',
          'best_effort_memory_usage_reduction': False,
          'symmetric_block_size': 2,
          'block_statistics': True
      },
  )
  def test_distributed_shampoo(self, best_effort_memory_usage_reduction,
                               symmetric_block_size, block_statistics):
    params = self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name='batch',
        preconditioning_compute_steps=2,
        best_effort_memory_usage_reduction=best_effort_memory_usage_reduction)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    def _update(unused_batch):
      return transform_fn(self.per_step_updates, state, params)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))

  def test_matrix_inverse_root(self):
    """Test for matrix inverse pth root."""

    def _gen_symmetrix_matrix(dim, condition_number):
      u = scipy.stats.ortho_group.rvs(dim=dim).astype(np.float64)
      v = u.T
      diag = np.diag([condition_number**(-i / (dim - 1)) for i in range(dim)])
      return u @ diag @ v

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10**e
      ms = _gen_symmetrix_matrix(16, condition_number)
      self.assertLess(
          np.abs(np.linalg.cond(ms) - condition_number),
          condition_number * 0.01)
      error = distributed_shampoo.matrix_inverse_pth_root(
          ms.astype(np.float32), 4, ridge_epsilon=1e-12)[1]
      if e < 7:
        self.assertLess(error, 0.1)
      else:
        # No guarantee of success after e >= 7
        pass


if __name__ == '__main__':
  absltest.main()

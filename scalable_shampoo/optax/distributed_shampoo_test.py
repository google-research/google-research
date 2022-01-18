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

import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from scalable_shampoo.optax import distributed_shampoo


class DistributedShampooTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (
        jnp.array([[1., 3.], [2., 4.]]), jnp.array([[3., 4.], [3., 4.]]))
    self.per_step_updates = (jnp.array([[500., 5.], [500., 5.]]),
                             jnp.array([[300., 3.], [300., 3.]]))

  @chex.all_variants(with_pmap=False)
  def test_distributed_shampoo(self):
    params = self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name='batch',
        preconditioning_compute_steps=2,
        best_effort_memory_usage_reduction=True)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    def _update(unused_batch):
      return transform_fn(self.per_step_updates, state, params)
    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))

  @chex.all_variants(with_pmap=False)
  def test_distributed_shampoo_quantization(self):
    params = self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name='batch',
        preconditioning_compute_steps=2,
        best_effort_memory_usage_reduction=True)
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
      diag = np.diag([condition_number ** (-i/(dim-1)) for i in range(dim)])
      return u @ diag @ v

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10 ** e
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

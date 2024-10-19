# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

from scalable_shampoo.optax import sm3


class SM3Test(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (
        jnp.array([[0.5, 0.5], [0.5, 0.5]]))
    self.per_step_updates = (jnp.array([[0.1, -0.1], [0.01, 0.01]]))

  @chex.all_variants(with_pmap=False)
  def test_sm3_basic(self):
    params = self.init_params

    optim = sm3.sm3(0.1, 0.9, 0.999)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    def _update(unused_batch):
      return transform_fn(self.per_step_updates, state, params)
    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))


if __name__ == '__main__':
  absltest.main()

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

"""Tests for optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import imp.max.modeling as mnn
from imp.max.optimization import config as opt_config
from imp.max.optimization import optimizers


class OptimizersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'sgd',
          'config': opt_config.SgdOptimizer(),
      }, {
          'testcase_name': 'adam',
          'config': opt_config.AdamOptimizer(),
      }, {
          'testcase_name': 'adamw',
          'config': opt_config.AdamWOptimizer(),
      }, {
          'testcase_name': 'adafactor',
          'config': opt_config.AdafactorOptimizer(),
      }, {
          'testcase_name': 'galore_adamw',
          'config': opt_config.GaLoreAdamWOptimizer(),
      },
  )
  def test_can_create_optimizer(self, config):
    optimizer = optimizers.get_optimizer(config)
    self.assertIsNotNone(optimizer)

  def test_galore_adamw(self):
    rank = 2
    svd_frequency = 2
    optimizer = optimizers.galore_adamw(
        learning_rate=0.1,
        rank=rank,
        svd_frequency=svd_frequency,
    )

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = mnn.Dense(5, kernel_shardings=('x', 'y'), name='dense_0')(x)
        x = mnn.Dense(2, kernel_shardings=('y', 'x'), name='dense_1')(x)
        return x.mean()

    model = Model()
    x = jax.random.normal(jax.random.key(0), (1, 2, 3))
    variables = model.init(jax.random.key(1), x)
    opt_state = optimizer.init(variables)

    @jax.jit
    def forward(variables):
      return model.apply(variables, x)

    # pytype: disable=attribute-error
    prev_projectors = opt_state[0].low_rank_projectors
    for _ in range(3 * svd_frequency):
      if opt_state[0].count % svd_frequency != 0:
        chex.assert_trees_all_equal(
            prev_projectors, opt_state[0].low_rank_projectors)
      else:
        chex.assert_trees_all_equal_shapes_and_dtypes(
            prev_projectors, opt_state[0].low_rank_projectors)
      grads = jax.grad(forward)(variables)
      updates, opt_state = optimizer.update(grads, opt_state, variables)
      prev_projectors = opt_state[0].low_rank_projectors
      variables = optax.apply_updates(variables, updates)
      # pytype: enable=attribute-error

    # The expected (unboxed) variables after updates
    expected_unboxed_updated_variables = {'params': {
        'dense_0': {
            'bias': jnp.asarray(
                [0.01761331, 0.11243041, 0.4335969, -0.59332484, -0.5919182],
                dtype=jnp.float32,
            ),
            'kernel': jnp.asarray(
                [[1.0843619, -0.200192, -0.06736058, -1.086317, -1.0990942],
                 [0.7323343, 0.35144898, 0.11247608, 0.9063656, 0.9431407],
                 [-0.56888074, 0.4732227, -0.4803256, -0.84843534, 0.2110598]],
                dtype=jnp.float32,
            ),
        },
        'dense_1': {
            'bias': jnp.asarray([-0.5999806, -0.5999806], dtype=jnp.float32),
            'kernel': jnp.asarray(
                [[-0.11302392, -0.22834896],
                 [0.44374648, -0.15414245],
                 [0.6436958, -0.27725446],
                 [0.77066934, 0.78202415],
                 [0.59039664, 0.9307547]],
                dtype=jnp.float32)}}}

    chex.assert_trees_all_equal(
        nn.unbox(variables),
        expected_unboxed_updated_variables,
    )


if __name__ == '__main__':
  absltest.main()

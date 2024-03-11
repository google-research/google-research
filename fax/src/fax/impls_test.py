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

import chex
import jax
from jax import numpy as jnp
import tensorflow as tf

from . import impls


class ImplsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._placements = {'clients': 100}
    self._sequence_length = 10
    self._comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
    )

  def test_map_to_server_placement_applies_fn(self):
    def add_constant(x):
      return x + jnp.array([1.0, 2.0])

    server_val = jnp.array([0.0, 1.0])
    actual_output = self._comp_factory.map_to_placement(
        add_constant, server_val, 'server'
    )
    expected_output = add_constant(server_val)
    chex.assert_trees_all_equal(actual_output, expected_output)

  def test_broadcast_on_float(self):
    actual_output = self._comp_factory.broadcast_to_placement(0.0, 'clients')
    expected_output = jnp.zeros(shape=[100])
    chex.assert_trees_all_equal(actual_output, expected_output)

  def test_runs_temp_sens_example(self):
    def _one_if_over(x, y):
      return jax.lax.cond(x > y, lambda: 1.0, lambda: 0.0)

    def temp_sens_example(m, t):
      t_at_c = self._comp_factory.broadcast_to_placement(t, 'clients')
      total_over = self._comp_factory.map_to_placement(
          _one_if_over, (m, t_at_c), 'clients'
      )
      return self._comp_factory.mean_from_placement(total_over)

    key = jax.random.PRNGKey(2)
    random_measurements = jax.random.uniform(
        key, shape=[self._placements['clients']]
    )

    self.assertEqual(
        temp_sens_example(random_measurements, jnp.array(0.5)), 0.53
    )

  def test_runs_fake_training(self):
    def _reduce_sequence(sequence, model):

      for _ in range(sequence.shape[0]):
        model += 1
      return model

    def fake_training(model, data):
      model_at_clients = self._comp_factory.broadcast_to_placement(
          model, 'clients'
      )
      trained_models = self._comp_factory.map_to_placement(
          _reduce_sequence, (data, model_at_clients), 'clients'
      )
      return self._comp_factory.mean_from_placement(trained_models)

    clients_data = jnp.ones(
        shape=[self._placements['clients'], self._sequence_length]
    )

    model = jnp.array(0.0)

    self.assertEqual(fake_training(model, clients_data), 10.0)


if __name__ == '__main__':
  tf.test.main()

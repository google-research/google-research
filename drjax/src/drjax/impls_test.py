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

from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import tensorflow as tf

from . import impls


@parameterized.named_parameters(
    ('spmd_axis_name_on', True), ('spmd_axis_name_off', False)
)
class ImplsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._placements = {'clients': 100}
    self._sequence_length = 10

  def test_broadcast_on_float(self, use_spmd_axis_name):
    comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
        use_spmd_axis_name=use_spmd_axis_name,
    )
    actual_output = comp_factory.broadcast_to_placement(0.0, 'clients')
    expected_output = jnp.zeros(shape=[100])
    chex.assert_trees_all_equal(actual_output, expected_output)

  def test_runs_temp_sens_example(self, use_spmd_axis_name):
    comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
        use_spmd_axis_name=use_spmd_axis_name,
    )
    def _one_if_over(x, y):
      return jax.lax.cond(x > y, lambda: 1.0, lambda: 0.0)

    def temp_sens_example(m, t):
      t_at_c = comp_factory.broadcast_to_placement(t, 'clients')
      total_over = comp_factory.map_to_placement(
          _one_if_over, (m, t_at_c), 'clients'
      )
      return comp_factory.mean_from_placement(total_over)

    key = jax.random.PRNGKey(2)
    random_measurements = jax.random.uniform(
        key, shape=[self._placements['clients']]
    )

    self.assertEqual(
        temp_sens_example(random_measurements, jnp.array(0.5)), 0.53
    )

  def test_runs_fake_training(self, use_spmd_axis_name):
    comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
        use_spmd_axis_name=use_spmd_axis_name,
    )
    def _reduce_sequence(sequence, model):

      for _ in range(sequence.shape[0]):
        model += 1
      return model

    def fake_training(model, data):
      model_at_clients = comp_factory.broadcast_to_placement(model, 'clients')
      trained_models = comp_factory.map_to_placement(
          _reduce_sequence, (data, model_at_clients), 'clients'
      )
      return comp_factory.mean_from_placement(trained_models)

    clients_data = jnp.ones(
        shape=[self._placements['clients'], self._sequence_length]
    )

    model = jnp.array(0.0)

    self.assertEqual(fake_training(model, clients_data), 10.0)


if __name__ == '__main__':
  tf.test.main()

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

"""Tests for ES-ENAS policies."""
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from es_enas import policies
from es_optimization import blackbox_functions


class PoliciesTest(parameterized.TestCase):

  def setUp(self):
    self.env = blackbox_functions.get_environment('Pendulum')
    self.state_dimensionality = self.env.state_dimensionality()
    self.action_dimensionality = self.env.action_dimensionality()
    self.example_state = np.zeros(self.state_dimensionality)
    super().setUp()

  @parameterized.named_parameters(('Linear', 0), ('One Hidden Layer', 1))
  def test_edge_sparsity_hidden(self, num_hidden_layers):

    hidden_layers = [32] * num_hidden_layers
    hidden_layer_edge_num = [2] * (num_hidden_layers + 1)
    policy = policies.NumpyEdgeSparsityPolicy(
        self.state_dimensionality,
        self.action_dimensionality,
        hidden_layers=hidden_layers,
        hidden_layer_edge_num=hidden_layer_edge_num,
        edge_policy_sample_mode='aggregate_edges')

    action = policy.get_action(self.example_state)
    logging.info('edges: %s', policy.edge_dict)
    self.assertLen(action, self.action_dimensionality)

  @parameterized.named_parameters(('aggregate_edges', 'aggregate_edges'),
                                  ('independent_edges', 'independent_edges'),
                                  ('residual_edges', 'residual_edges'))
  def test_edge_sparsity_sample_modes(self, edge_policy_sample_mode):

    hidden_layers = [32]
    hidden_layer_edge_num = [5, 17]
    policy = policies.NumpyEdgeSparsityPolicy(
        self.state_dimensionality,
        self.action_dimensionality,
        hidden_layers=hidden_layers,
        hidden_layer_edge_num=hidden_layer_edge_num,
        edge_policy_sample_mode=edge_policy_sample_mode)

    action = policy.get_action(self.example_state)
    logging.info('edges: %s', policy.edge_dict)
    self.assertLen(action, self.action_dimensionality)

  @parameterized.named_parameters(('Linear', 0), ('One Hidden Layer', 1))
  def test_weight_sharing_hidden(self, num_hidden_layers):
    hidden_layers = [32] * num_hidden_layers
    policy = policies.NumpyWeightSharingPolicy(
        self.state_dimensionality,
        self.action_dimensionality,
        hidden_layers=hidden_layers,
        num_partitions=17)
    action = policy.get_action(self.example_state)
    logging.info('edges: %s', policy.edge_dict)
    self.assertLen(action, self.action_dimensionality)

  @parameterized.named_parameters(('Linear', 0), ('One Hidden Layer', 1))
  def test_op_edge(self, num_hidden_layers):
    hidden_layers = [32] * num_hidden_layers
    policy = policies.OpEdgePolicy(
        self.state_dimensionality,
        self.action_dimensionality,
        hidden_layers=hidden_layers,
        num_edges=2)
    action = policy.get_action(self.example_state)
    logging.info('edges: %s', policy.edge_dict)
    logging.info('nodes: %s', policy.node_op_dict)
    self.assertLen(action, self.action_dimensionality)

  @parameterized.named_parameters(('Linear', 0), ('One Hidden Layer', 1),
                                  ('Two Hidden Layers', 2))
  def test_efficient_net(self, num_hidden_layers):
    hidden_layers = [32] * num_hidden_layers
    hidden_layer_edge_num = [5, 17]
    edge_policy_sample_mode = 'aggregate_edges'
    policy = policies.EfficientNetPolicy(
        self.state_dimensionality,
        self.action_dimensionality,
        hidden_layers=hidden_layers,
        hidden_layer_edge_num=hidden_layer_edge_num,
        edge_policy_sample_mode=edge_policy_sample_mode)
    action = policy.get_action(self.example_state)
    logging.info('edges: %s', policy.edge_dict)
    self.assertLen(action, self.action_dimensionality)
    num_multiplications = policy.compute_flops_multiplication()
    avg_inference_time = policy.compute_inference_time(samples=10)

    logging.info(
        'Number of multiplications: %d, Average Inference time: %f seconds',
        num_multiplications, avg_inference_time)


if __name__ == '__main__':
  absltest.main()

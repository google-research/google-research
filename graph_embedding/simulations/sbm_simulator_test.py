# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for simulation functions library."""

import collections

from absl.testing import absltest
import numpy as np

from graph_embedding.simulations import sbm_simulator


class SbmSimulatorTestSbm(absltest.TestCase):

  def setUp(self):
    super(SbmSimulatorTestSbm, self).setUp()
    self.simulation_with_graph = sbm_simulator.StochasticBlockModel()
    sbm_simulator.SimulateSbm(
        self.simulation_with_graph,
        num_vertices=50,
        num_edges=500,
        pi=[0.5, 0.5],
        prop_mat=np.ones(shape=(2, 2)))

  def test_simulate_sbm(self):
    self.assertEqual(self.simulation_with_graph.graph.num_vertices(), 50)
    self.assertEqual(np.sum(
        [k == 0 for k in self.simulation_with_graph.graph_memberships]),
                     25)
    self.assertEqual(self.simulation_with_graph.type1_clusters, [0, 1])

  def test_simulate_sbm_community_sizes(self):
    simulation = sbm_simulator.StochasticBlockModel()
    unbalanced_pi = np.array([x + 1 for x in range(10)]) / 55
    sbm_simulator.SimulateSbm(
        simulation,
        num_vertices=50,
        num_edges=100,
        pi=unbalanced_pi,
        prop_mat=np.ones(shape=(10, 10)))
    expected_sizes = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    group_counts = collections.defaultdict(int)
    for cluster_id in simulation.graph_memberships:
      group_counts[cluster_id] += 1
    actual_sizes = [count for cluster_id, count in sorted(group_counts.items())]
    self.assertSameStructure(expected_sizes, actual_sizes)
    self.assertEqual(simulation.type1_clusters, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  # This test fails if the precision at sbm_simulator.SimulateSbm is >= 16. It
  # is currently hard-coded at 12 to cover all conceivable realistic ways to
  # programmatically construct a simplex vector (e.g. `pi` below).
  def test_simulate_sbm_community_sizes_seven_groups(self):
    simulation = sbm_simulator.StochasticBlockModel()
    num_communities = 7
    community_size_slope = 0.5
    pi = (
        np.array(range(num_communities)) * community_size_slope +
        np.ones(num_communities))
    pi /= np.sum(pi)
    sbm_simulator.SimulateSbm(
        simulation,
        num_vertices=500,
        num_edges=10000,
        pi=pi,
        prop_mat=np.ones(shape=(num_communities, num_communities)))
    expected_sizes = [29, 43, 58, 71, 85, 100, 114]
    group_counts = collections.Counter(simulation.graph_memberships)
    actual_sizes = [count for cluster_id, count in sorted(group_counts.items())]
    self.assertEqual(expected_sizes, actual_sizes)
    self.assertEqual(simulation.type1_clusters, [0, 1, 2, 3, 4, 5, 6])


class SbmSimulatorTestFeatures(SbmSimulatorTestSbm):

  def test_simulate_features_dimensions(self):
    sbm_simulator.SimulateFeatures(
        self.simulation_with_graph,
        center_var=1.0,
        feature_dim=16,
        num_groups=2)
    self.assertIsInstance(self.simulation_with_graph.node_features, np.ndarray)
    self.assertEqual(self.simulation_with_graph.node_features.shape[0], 50)
    self.assertEqual(self.simulation_with_graph.node_features.shape[1], 16)

  def test_simulate_features_equal_memberships(self):
    sbm_simulator.SimulateFeatures(
        self.simulation_with_graph,
        center_var=1.0,
        feature_dim=16,
        num_groups=2,
        match_type=sbm_simulator.MatchType.NESTED)
    self.assertSameStructure(
        list(self.simulation_with_graph.graph_memberships),
        list(self.simulation_with_graph.feature_memberships))

  def test_simulate_features_nested_memberships(self):
    sbm_simulator.SimulateFeatures(
        self.simulation_with_graph,
        center_var=1.0,
        feature_dim=16,
        num_groups=3,
        match_type=sbm_simulator.MatchType.NESTED)
    expected_memberships = [0] * 13 + [1] * 12 + [2] * 25
    self.assertSameStructure(
        list([int(d) for d in self.simulation_with_graph.feature_memberships]),
        expected_memberships)

  def test_simulate_features_grouped_memberships(self):
    simulation = sbm_simulator.StochasticBlockModel()
    sbm_simulator.SimulateSbm(
        simulation, 30, 100, pi=np.ones(3) / 3, prop_mat=np.ones((3, 3)))
    sbm_simulator.SimulateFeatures(
        simulation,
        center_var=1.0,
        feature_dim=4,
        num_groups=2,
        match_type=sbm_simulator.MatchType.GROUPED)
    expected_memberships = [0] * 20 + [1] * 10
    self.assertSameStructure(
        list([int(d) for d in simulation.feature_memberships]),
        expected_memberships)

  def test_simulate_edge_features(self):
    sbm_simulator.SimulateEdgeFeatures(
        self.simulation_with_graph, feature_dim=4)
    self.assertLen(self.simulation_with_graph.edge_features,
                   self.simulation_with_graph.graph.num_edges())
    self.assertTrue(
        all([
            v.shape[0] == 4
            for v in self.simulation_with_graph.edge_features.values()
        ]))


if __name__ == '__main__':
  absltest.main()

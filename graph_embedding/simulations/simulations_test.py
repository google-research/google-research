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

"""Tests for simulation functions."""
import collections

from absl.testing import absltest
import numpy as np

from graph_embedding.simulations import sbm_simulator
from graph_embedding.simulations import simulations


class SimulationsTest(absltest.TestCase):

  def test_simulate_sbm_with_features(self):
    result = simulations.GenerateStochasticBlockModelWithFeatures(
        num_vertices=50, num_edges=500, pi=[0.5, 0.5],
        prop_mat=np.ones(shape=(2, 2)),
        feature_center_distance=1.0, feature_dim=16,
        feature_group_match_type=simulations.MatchType.NESTED,
        edge_feature_dim=4)

    self.assertEqual(result.graph.num_vertices(), 50)
    self.assertEqual(np.sum([k == 0 for k in result.graph_memberships]), 25)
    self.assertEqual(result.node_features1.shape[0], 50)
    self.assertEqual(result.node_features1.shape[1], 16)
    self.assertSameStructure(list(result.graph_memberships),
                             list(result.feature_memberships))
    self.assertLen(result.edge_features, result.graph.num_edges())
    self.assertTrue(all([v.shape[0] == 4 for
                         v in result.edge_features.values()]))

  def test_heterogeneous_sbm(self):
    result = simulations.GenerateStochasticBlockModelWithFeatures(
        num_vertices=200,
        num_edges=16000,
        pi=np.array([0.5, 0.5]),
        feature_center_distance=1.0,
        feature_dim=32,
        num_vertices2=200,
        pi2=np.array([0.5, 0.5]),
        feature_center_distance2=1.0,
        feature_dim2=48,
        feature_type_correlation=0.5,
        feature_type_center_distance=1.0,
        edge_probability_profile=sbm_simulator.EdgeProbabilityProfile(
            p_to_q_ratio1=5.0,
            p_to_q_ratio2=5.0,
            p_to_q_ratio_cross=5.0)
    )
    self.assertEqual(result.graph.num_vertices(), 400)
    membership_counts = collections.Counter(result.graph_memberships)
    self.assertEqual(membership_counts[0], 100)
    self.assertEqual(membership_counts[1], 100)
    self.assertEqual(membership_counts[2], 100)
    self.assertEqual(membership_counts[3], 100)
    self.assertEqual(result.type1_clusters, [0, 1])
    self.assertEqual(result.type2_clusters, [2, 3])
    self.assertEqual(result.cross_links, [(0, 2), (1, 3)])
    self.assertIsInstance(result.node_features1, np.ndarray)
    self.assertEqual(result.node_features1.shape[0], 200)
    self.assertEqual(result.node_features1.shape[1], 32)
    self.assertIsInstance(result.node_features2, np.ndarray)
    self.assertEqual(result.node_features2.shape[0], 200)
    self.assertEqual(result.node_features2.shape[1], 48)


if __name__ == '__main__':
  absltest.main()

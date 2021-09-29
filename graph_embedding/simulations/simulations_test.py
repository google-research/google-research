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

from absl.testing import absltest
import numpy as np

from graph_embedding.simulations import simulations


class SimulationsTest(absltest.TestCase):

  def test_simulate_sbm_with_features(self):
    result = simulations.GenerateStochasticBlockModelWithFeatures(
        num_vertices=50, num_edges=500, pi=[0.5, 0.5],
        prop_mat=np.ones(shape=(2, 2)),
        feature_center_distance=1.0, feature_dim=16, num_feature_groups=2,
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


if __name__ == '__main__':
  absltest.main()

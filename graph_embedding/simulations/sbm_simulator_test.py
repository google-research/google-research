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

"""Tests for SbmSimulator class."""

from absl.testing import absltest
import numpy as np

from graph_embedding.simulations import sbm_simulator


# TODO(palowitch): add tests for feature generators.
class SbmSimulatorTest(absltest.TestCase):

  def test_simulate_sbm(self):
    simulator = sbm_simulator.SbmSimulator()
    simulator.SimulateSbm(num_vertices=10, num_edges=20, pi=[0.5, 0.5],
                          prop_mat=np.ones(shape=(2, 2)))
    self.assertEqual(simulator.graph.num_vertices(), 10)


if __name__ == '__main__':
  absltest.main()

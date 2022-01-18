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

"""Tests for control_flow_graph_feature."""

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest

from ipagnn.datasets.control_flow_programs import control_flow_graph_feature
from ipagnn.datasets.control_flow_programs import python_programs
from ipagnn.datasets.control_flow_programs.encoders import encoders
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats_config
from ipagnn.datasets.control_flow_programs.program_generators import program_generators

ArithmeticRepeatsConfig = arithmetic_repeats_config.ArithmeticRepeatsConfig


class ControlFlowGraphFeatureTest(absltest.TestCase):

  def test_encode_example(self):
    config = ArithmeticRepeatsConfig(base=10, length=5)
    python_source = program_generators.generate_python_source(
        config.length, config)
    cfg = python_programs.to_cfg(python_source)
    program_encoder = encoders.get_program_encoder(config)
    feature = control_flow_graph_feature.ControlFlowGraphFeature(
        include_back_edges=False, encoder=program_encoder)
    feature.encode_example((cfg, python_source))

  def test_get_adjacency_matrix(self):
    cfg = python_programs.to_cfg("""
v1 = 0
while v1 > 1:
  v1 -= 2
  v0 += 3
""")
    adj = control_flow_graph_feature.get_adjacency_matrix(cfg.nodes, 4)
    logging.info(adj)
    self.assertListEqual(
        adj.T.tolist(),
        [[0., 1., 0., 0., 0.],  # 0 -> 1
         [0., 0., 1., 0., 1.],  # 1 -> {2, 4}
         [0., 0., 0., 1., 0.],  # 2 -> 3
         [0., 1., 0., 0., 0.],  # 3 -> 1
         [0., 0., 0., 0., 1.]]  # 4 -> 4 (exit node)
    )

  def test_get_adjacency_list(self):
    cfg = python_programs.to_cfg("""
v1 = 2
while v1 > 0:
  v1 -= 1
  v0 += 2
  v0 -= 1
  v0 *= 4
""")
    adj = control_flow_graph_feature.get_adjacency_list(cfg.nodes, 6)
    self.assertEqual(
        adj,
        [[1, 0],
         [2, 1],
         [6, 1],  # while to exit
         [3, 2],
         [4, 3],
         [5, 4],
         [1, 5],
         [6, 6]]  # exit to exit
    )

  def test_get_branch_list_from_nodes(self):
    cfg = python_programs.to_cfg("""
v1 = 2
while v1 > 0:
  v1 -= 1
  v0 += 2
  v0 -= 1
  v0 *= 4
""")
    branch_list = control_flow_graph_feature.get_branch_list(cfg.nodes, 6)
    self.assertEqual(
        branch_list,
        [[1, 1],  # v1 = 2
         [2, 6],  # while v1 > 0:
         [3, 3],  # v1 -= 1
         [4, 4],  # v0 += 2
         [5, 5],  # v0 -= 1
         [1, 1],  # v0 *= 4
         [6, 6]]  # exit-node
    )


if __name__ == '__main__':
  absltest.main()

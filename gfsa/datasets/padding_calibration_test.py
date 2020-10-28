# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Tests for gfsa.datasets.google.random_python.padding_calibration."""

from absl.testing import absltest
import gast
from gfsa import automaton_builder
from gfsa import py_ast_graphs
from gfsa.datasets import graph_bundle
from gfsa.datasets import padding_calibration


class PaddingCalibrationTest(absltest.TestCase):

  def test_calibrate_padding(self):
    # Make sure padding calibration doesn't error out, so that it works when
    # run interactively.
    def build_example(size):
      tree = gast.Module(
          body=[gast.Constant(value=i, kind=None) for i in range(size)],
          type_ignores=[])
      py_graph, ast_to_node_id = (py_ast_graphs.py_ast_to_graph(tree))
      edges = []
      for i in range(1, size, 2):
        edges.append((ast_to_node_id[id(tree.body[i])],
                      ast_to_node_id[id(tree.body[i - 1])], 1))
      return graph_bundle.convert_graph_with_edges(py_graph, edges,
                                                   py_ast_graphs.BUILDER)

    padding_calibration.calibrate_padding(
        example_builder=build_example,
        desired_sizes=graph_bundle.PaddingConfig(
            static_max_metadata=automaton_builder.EncodedGraphMetadata(
                num_nodes=64, num_input_tagged_nodes=64),
            max_initial_transitions=128,
            max_in_tagged_transitions=256,
            max_edges=64,
        ),
        samples=50,
        optimization_max_steps=500,
        round_to_powers_of_two=True)


if __name__ == '__main__':
  absltest.main()

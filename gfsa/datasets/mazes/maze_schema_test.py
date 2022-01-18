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

# Lint as: python3
"""Tests for gfsa.datasets.mazes.maze_schema."""

from absl.testing import absltest
import numpy as np
from gfsa import graph_types
from gfsa import schema_util
from gfsa.datasets.mazes import maze_schema


class MazeSchemaTest(absltest.TestCase):

  def test_encode_maze(self):
    """Tests that an encoded maze is correct and matches the schema."""

    maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]).astype(bool)

    encoded_graph, coordinates = maze_schema.encode_maze(maze)

    # Check coordinates.
    expected_coords = []
    for r in range(3):
      for c in range(5):
        if (r, c) != (1, 1):
          expected_coords.append((r, c))

    self.assertEqual(coordinates, expected_coords)

    # Check a few nodes.
    self.assertEqual(
        encoded_graph[graph_types.NodeId("cell_0_0")],
        graph_types.GraphNode(
            graph_types.NodeType("cell_xRxD"), {
                graph_types.OutEdgeType("R_out"): [
                    graph_types.InputTaggedNode(
                        graph_types.NodeId("cell_0_1"),
                        graph_types.InEdgeType("L_in"))
                ],
                graph_types.OutEdgeType("D_out"): [
                    graph_types.InputTaggedNode(
                        graph_types.NodeId("cell_1_0"),
                        graph_types.InEdgeType("U_in"))
                ],
            }))

    self.assertEqual(
        encoded_graph[graph_types.NodeId("cell_1_4")],
        graph_types.GraphNode(
            graph_types.NodeType("cell_LxUD"), {
                graph_types.OutEdgeType("L_out"): [
                    graph_types.InputTaggedNode(
                        graph_types.NodeId("cell_1_3"),
                        graph_types.InEdgeType("R_in"))
                ],
                graph_types.OutEdgeType("U_out"): [
                    graph_types.InputTaggedNode(
                        graph_types.NodeId("cell_0_4"),
                        graph_types.InEdgeType("D_in"))
                ],
                graph_types.OutEdgeType("D_out"): [
                    graph_types.InputTaggedNode(
                        graph_types.NodeId("cell_2_4"),
                        graph_types.InEdgeType("U_in"))
                ],
            }))

    # Check schema validity.
    schema_util.assert_conforms_to_schema(encoded_graph,
                                          maze_schema.build_maze_schema(2))


if __name__ == "__main__":
  absltest.main()

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

"""Constants and functions for maze task."""

from typing import List, Tuple

from gfsa import automaton_builder
from gfsa import graph_types
from gfsa.datasets import graph_bundle
from gfsa.datasets.mazes import maze_schema


DIRECTION_ORDERING = "LRUD"


def maze_primitive_edges(
    maze_graph
):
  """Build a graph bundle for a given maze.

  Args:
    maze_graph: Encoded graph representing the maze.

  Returns:
    List of edges corresponding to primitive actions in the maze.
  """
  primitives = []
  for node_id, node_info in maze_graph.items():
    for i, direction in enumerate(DIRECTION_ORDERING):
      out_key = graph_types.OutEdgeType(f"{direction}_out")
      if out_key in node_info.out_edges:
        dest, = node_info.out_edges[out_key]
        primitives.append((node_id, dest.node_id, i))
      else:
        primitives.append((node_id, node_id, i))

  return primitives


SCHEMA = maze_schema.build_maze_schema(2)

# Backtracking doesn't make sense for maze environment.
BUILDER = automaton_builder.AutomatonBuilder(SCHEMA, with_backtrack=False)

PADDING_CONFIG = graph_bundle.PaddingConfig(
    static_max_metadata=automaton_builder.EncodedGraphMetadata(
        num_nodes=256, num_input_tagged_nodes=512),
    max_initial_transitions=512,
    max_in_tagged_transitions=2048,
    max_edges=1024)

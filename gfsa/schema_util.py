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

"""Helpers for working with node and graph schemas.

See graph_types.py for descriptions of node and graph schemas, which represent
families of graph POMDPs with the same action and observation spaces.
"""

from typing import List

from gfsa import graph_types


def assert_conforms_to_schema(graph,
                              graph_schema):
  """Checks that a graph conforms to a schema.

  In order to conform to a schema:
  - Every node in the graph must have a type that is in the schema.
  - Every edge must connect to another node in the graph.
  - The input and output edge types for a node must belong to the node type's
      possible input and output edge types.
  - Every node of a given node type must have at least one outgoing edge of
      each possible outgoing edge type.

  Args:
    graph: Graph to check.
    graph_schema: Schema to use.

  Raises:
    ValueError: If graph fails to validate against the schema.
  """
  for node_id, node in graph.items():
    if node.node_type not in graph_schema:
      raise ValueError(f"Node {node_id}'s type {node.node_type} not in schema")

    node_schema = graph_schema[node.node_type]
    for out_edge_type in node_schema.out_edges:
      # Check if the list of edges with this type exists and is non-empty
      if not node.out_edges.get(out_edge_type):
        raise ValueError(
            f"Node {node_id} missing out edge of type {out_edge_type}")

    for out_edge_type, destinations in node.out_edges.items():
      if out_edge_type not in node_schema.out_edges:
        raise ValueError(
            f"Node {node_id} has out-edges of invalid type {out_edge_type}")

      for dest in destinations:
        if dest.node_id not in graph:
          raise ValueError(
              f"Node {node_id} has connection to missing node {dest.node_id}")
        dest_node_type = graph[dest.node_id].node_type
        if dest_node_type not in graph_schema:
          raise ValueError(
              f"Node {dest.node_id}'s type {dest_node_type} not in schema")
        if dest.in_edge not in graph_schema[dest_node_type].in_edges:
          raise ValueError(
              f"Node {dest.node_id} has in-edges of invalid type "
              f"{dest.in_edge} (from node {node_id}, outgoing type "
              f"{out_edge_type})")


def all_input_tagged_nodes(
    graph):
  """Gets all input-tagged nodes that are reachable in a graph.

  An input-tagged node is considered reachable if there is a way to get to that
  node by an edge of that input type.

  Output will be ordered lexicographically, first by the node ordering in the
  input graph's keys, then by edge type.

  Args:
    graph: Graph to process.

  Returns:
    List of all input-tagged nodes.
  """
  node_id_to_index = {node_id: i for i, node_id in enumerate(graph)}
  reachable_itns = set()
  for node in graph.values():
    for itns in node.out_edges.values():
      reachable_itns.update(itns)

  return sorted(
      reachable_itns,
      key=lambda itn: (node_id_to_index[itn.node_id], itn.in_edge))

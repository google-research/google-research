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

# Lint as: python3
"""Shared utilities for building edges for graphs.

Helper functions here operate either on generic automaton graphs or specifically
on python ASTs.
"""

import collections
from typing import Dict, List, Optional, Set, Tuple

import gast

from gfsa import graph_types

JUMPS_OUT_OF_EDGE_TYPE = "EXTRA_JUMPS_OUT_OF"

PROGRAM_GRAPH_EDGE_TYPES = {
    "PG_CFG_NEXT",
    "PG_LAST_READ",
    "PG_LAST_WRITE",
    "PG_COMPUTED_FROM",
    "PG_RETURNS_TO",
    "PG_FORMAL_ARG_NAME",
    "PG_NEXT_SYNTAX",
    "PG_LAST_LEXICAL_USE",
    "PG_CALLS",
}

SAME_IDENTIFIER_EDGE_TYPE = "EXTRA_SAME_IDENTIFIER"


def compute_jumps_out_edges(
    tree,
    ast_to_node_id,
    from_return=True,
    from_retval=True,
    from_break_cont=True
):
  """Compute EXTRA_JUMPS_OUT_OF edges.

  There are two types edges with this type:
  - All break and continue statements should be connected to the enclosing loop.
  - All return statements and return value expressions should be connected to
    the function body.

  This task is quite simple, but requires the automaton to learn non-trivial
  behavior: it must remember whether it came from a break/continue or a return,
  and it must use context to figure out whether a given expression is a return
  value or some other type of expression.

  Args:
    tree: The AST to construct targets for.
    ast_to_node_id: Dictionary that maps AST node ids to their graph node id.
    from_return: Whether to include edges from return statements to their
      containing function.
    from_retval: Whether to include edges from return VALUES to their containing
      function.
    from_break_cont: Whether to include edges from break/continue statements to
      their containing loop.

  Returns:
    List of "EXTRA_JUMPS_OUT_OF" edges.
  """
  result = []

  # pytype: disable=attribute-error
  def _go(subtree, parent_func,
          parent_loop):
    """Recursively process a subtree.

    The high level strategy is to recursively walk down the tree. When we see
    a function or loop node, we update the `parent_func` or `parent_loop`
    argument, and then continue descending. When we reach a `break`, `continue`,
    or `return`, we then connect these nodes to the corresponding innermost
    function or loop.

    Args:
      subtree: Current subtree to process.
      parent_func: The AST node corresponding to the (innermost) FunctionDef
        that contains this subtree.
      parent_loop: The AST node corresponding to the (innermost) For or While
        loop that contains this subtree.
    """
    if isinstance(subtree, gast.Return):
      assert parent_func, "return outside function"
      if from_return:
        result.append((ast_to_node_id[id(subtree)],
                       ast_to_node_id[id(parent_func)], JUMPS_OUT_OF_EDGE_TYPE))
      if from_retval and subtree.value:
        result.append((ast_to_node_id[id(subtree.value)],
                       ast_to_node_id[id(parent_func)], JUMPS_OUT_OF_EDGE_TYPE))
    elif isinstance(subtree, (gast.Break, gast.Continue)):
      assert parent_loop, "break or continue outside loop"
      if from_break_cont:
        result.append((ast_to_node_id[id(subtree)],
                       ast_to_node_id[id(parent_loop)], JUMPS_OUT_OF_EDGE_TYPE))
    elif isinstance(subtree, gast.FunctionDef):
      # Update current function
      for stmt in subtree.body:
        _go(stmt, subtree, None)
    elif isinstance(subtree, (gast.For, gast.While)):
      # Update current loop
      for stmt in subtree.body:
        _go(stmt, parent_func, subtree)
    else:
      for child in gast.iter_child_nodes(subtree):
        _go(child, parent_func, parent_loop)

  # pytype: enable=attribute-error

  _go(tree, None, None)
  return result


def schema_edge_types(schema,
                      with_node_types = False):
  """Returns a list of schema edge types.

  Args:
    schema: Automaton graph schema to use.
    with_node_types: Whether to include the node type of the source node.

  Returns:
    Set of schema edge types.
  """
  result = set()
  for node_type, node_schema in schema.items():
    for out_type in node_schema.out_edges:
      if with_node_types:
        result.add(f"SCHEMA_{out_type}_FROM_{node_type}")
      else:
        result.add(f"SCHEMA_{out_type}")
  return result


def compute_schema_edges(
    graph,
    with_node_types = False
):
  """Compute SCHEMA_* edges from the encoded graph.

  We extract the outgoing edges that the automaton sees, but remove the sentinel
  "missing" edges. Incoming edges are redundant and less informative so they
  are not included.

  Args:
    graph: Automaton graph to use.
    with_node_types: Whether to include the node type of the source node.

  Returns:
    List of schema edges.
  """
  result = []
  for source_node_id, node_info in graph.items():
    for out_type, destinations in node_info.out_edges.items():
      for in_tagged_node in destinations:
        if not in_tagged_node.in_edge.endswith("_missing"):
          if with_node_types:
            edge_type = f"SCHEMA_{out_type}_FROM_{node_info.node_type}"
            result.append((source_node_id, in_tagged_node.node_id, edge_type))
          else:
            edge_type = f"SCHEMA_{out_type}"
            result.append((source_node_id, in_tagged_node.node_id, edge_type))

  return result


def compute_same_identifier_edges(
    tree, ast_to_node_id
):
  """Compute EXTRA_SAME_IDENTIFIER edges from an AST.

  These edges connect any two `Name` nodes with the same identifier, including

  Args:
    tree: The AST to construct an example for.
    ast_to_node_id: Dictionary that maps AST node ids to their graph node id.

  Returns:
    List of same-identifier edges.
  """
  result = []
  nodes_by_identifier = collections.defaultdict(list)
  for ast_node in gast.walk(tree):
    if isinstance(ast_node, gast.Name):
      graph_node_id = ast_to_node_id[id(ast_node)]
      identifier = ast_node.id  # pytype: disable=attribute-error
      for matching in nodes_by_identifier[identifier]:
        result.append((graph_node_id, matching, SAME_IDENTIFIER_EDGE_TYPE))
        result.append((matching, graph_node_id, SAME_IDENTIFIER_EDGE_TYPE))
      nodes_by_identifier[identifier].append(graph_node_id)
      result.append((graph_node_id, graph_node_id, SAME_IDENTIFIER_EDGE_TYPE))

  return result


def nth_child_edge_types(max_child_count):
  """Constructs the edge types for nth-child edges.

  Args:
    max_child_count: Maximum number of children that get explicit nth-child
      edges.

  Returns:
    Set of edge type names.
  """
  return {f"CHILD_INDEX_{i}" for i in range(max_child_count)}


def compute_nth_child_edges(
    graph, max_child_count
):
  """Computes CHILD_INDEX_* edges from a graph.

  We assume that the graph was generated by `generic_ast_graphs.ast_to_graph`
  and thus that sequence edges are represented with "{field}_out_first/last" and
  sequence helper items. Note that the produced edges connect the parent node
  to each of the child helper nodes, since those exist between the parent node
  and the child AST nodes. This is done so that the edges for the field name and
  the index align with each other (i.e. connect the same pair of nodes), to make
  it easier to learn useful embeddings.

  Args:
    graph: Automaton graph to use.
    max_child_count: Maximum number of children that get explicit nth-child
      edges.

  Returns:
    List of CHILD_INDEX_* edges.
  """
  result = []
  for source_node_id, node_info in graph.items():
    for out_type, destinations in node_info.out_edges.items():
      if out_type.endswith("_out_first"):
        # This is a sequence node, so we should add nth-child edges.
        in_tagged_node, = destinations
        i = 0
        while (i < max_child_count and
               not in_tagged_node.in_edge.endswith("_missing")):
          result.append(
              (source_node_id, in_tagged_node.node_id, f"CHILD_INDEX_{i}"))
          # Follow the chain of next pointers.
          in_tagged_node, = graph[in_tagged_node.node_id].out_edges["next_out"]
          i += 1

  return result


def encode_edges(
    edges,
    edge_types,
    skip_unknown = False
):
  """Converts each string edge type into an index in `edge_types`.

  Args:
    edges: Edges to encode.
    edge_types: Ordered list of types, used to determine the integer indices.
    skip_unknown: Whether to ignore edge types that aren't in the ordered list.

  Returns:
    Encoded edges.

  Raises:
    KeyError: if skip_unknown=False and an edge type isn't recognized.
  """
  type_to_idx_map = {type_name: i for i, type_name in enumerate(edge_types)}
  if len(type_to_idx_map) != len(edge_types):
    raise ValueError(f"Duplicate values in edge type list {edge_types}")
  result = []
  for s, d, t in edges:
    if t in type_to_idx_map:
      result.append((s, d, type_to_idx_map[t]))
    elif not skip_unknown:
      raise KeyError(f"Unrecognized edge type {t}")
  return result

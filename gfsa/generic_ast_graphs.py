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

"""Language-agnostic AST representation and schema utilities.

This module provides a dataclass for defining an AST structure, and then allows
conversion from ASTs to schemas and automaton graphs.
"""

import enum
from typing import Any, Dict, List, Tuple, Union

import dataclasses

from gfsa import graph_types
from gfsa import jax_util

# Type aliases for readability.
NodeType = FieldName = SequenceItemType = str


class FieldType(enum.Enum):
  """Specifies the expected contents of a given field."""
  NO_CHILDREN = "NO_CHILDREN"  # Metadata field or unused syntax.
  ONE_CHILD = "ONE_CHILD"
  OPTIONAL_CHILD = "OPTIONAL_CHILD"
  NONEMPTY_SEQUENCE = "NONEMPTY_SEQUENCE"
  SEQUENCE = "SEQUENCE"
  IGNORE = "IGNORE"  # Might have children, but don't convert them.

  def __repr__(self):
    """Represents values with valid Python syntax."""
    return str(self)


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class ASTNodeSpec:
  """Represents the expected fields of an AST node and their contents.

  This class specifies what types of fields each AST node is expected to have.
  An ASTNodeSpec can be used to construct a NodeSchema, and can also be used to
  convert an AST node into a GraphNode.

  Attributes:
    fields: Collection of expected fields, annotated with the type of children
      we expect.
    sequence_item_types: Mapping from sequence field names to the type of item
      we expect them to contain; this determines how sequence helper node
      parameters are shared (each sequence item type maps to a distinct helper
      node type).
    has_parent: Whether this node type has a parent. We assume each node type
      either always appears as the root or never appears as the root; if this is
      not the case, inputs should be preprocessed to ensure this (i.e. by
      wrapping everything in a singleton root node).
  """
  fields: Dict[FieldName, FieldType] = dataclasses.field(default_factory=dict)
  sequence_item_types: Dict[FieldName, SequenceItemType] = (
      dataclasses.field(default_factory=dict))
  has_parent: bool = True


ASTSpec = Dict[NodeType, ASTNodeSpec]


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class GenericASTNode:
  """Generic representation of an AST node.

  Attributes:
    node_id: ID of this node. Should be unique in the tree.
    node_type: Type of this node.
    fields: Dictionary of children for each field.
  """
  node_id: Any
  node_type: str
  fields: Dict[str, List["GenericASTNode"]]


def build_ast_graph_schema(ast_spec):
  """Builds a graph schema for an AST.

  This logic is described in Appendix B.1 of the paper.

  Each AST node becomes a new node type, whose edges are determined by its
  fields (along with whether it has a parent). Additionally, we generate helper
  node types for each grammar category that appears as a sequence (i.e. one
  for sequences of statments, another for sequences of expressions).

  Nodes with a parent get two edge types for that parent:
  - (in) "parent_in": the edge from the parent to this node
  - (out) "parent_out": the edge from this node to its parent

  ONE_CHILD fields become two edge types:
  - (in) "{field_name}_in": the edge from the child to this node
  - (out) "{field_name}_out": the edge from this node to the child

  OPTIONAL_CHILD fields become three edge types (to account for the
  invariant where every outgoing edge type must have at least one edge):
  - (in) "{field_name}_in": the edge from the child to this node (if it exists)
  - (in) "{field_name}_missing": a loopback edge from this node to itself, used
      as a sentinel value for when the child is missing
  - (out) "{field_name}_out": if there is a child of this type, this edge
      connects to that child; otherwise, it is a loopback edge to this node
      with incoming type "{field_name}_missing".

  NONEMPTY_SEQUENCE or SEQUENCE fields become 4 or 5 edge types:
  - (in) "{field_name}_in": used for EVERY edge from a child-item-helper to this
      node
  - (in) "{field_name}_missing": loopback edges for missing outgoing edges
      (only for SEQUENCE)
  - (out) "{field_name}_all": edges to every child-item-helper, or a loop back
      to "{field_name}_missing" if the sequence is empty
  - (out) "{field_name}_first": edges to the first child-item-helper, or a loop
      back to "{field_name}_missing" if the sequence is empty
  - (out) "{field_name}_last": edges to the last child-item-helper, or a loop
      back to "{field_name}_missing" if the sequence is empty

  NO_CHILDREN fields must be empty ([] or None) and will throw an error
  otherwise. IGNORE fields will be simply ignored.

  Child item helpers are added between any AST node with a sequence of children
  and the children in that sequence. These helpers have the following edge
  types:
  - "item_{in/out}": between the helper and the child
  - "parent_{in/out}": between the helper and the parent
  - "next_{in/out/missing}": between this helper and the next one in the
      sequence, or a loop from "out" to "missing" if this is the last element
  - "prev_{in/out/missing}": between this helper and the previous one in the
      sequence, or a loop from "out" to "missing" if this is the first element
  There is one helper type for each unique value in `sequence_item_types` across
  all node types.

  Args:
    ast_spec: AST spec to generate a schema for.

  Returns:
    GraphSchema with nodes as described above.
  """
  result = {}
  seen_sequence_categories = set()
  # Build node schemas for each AST node
  for node_type, node_spec in ast_spec.items():
    node_schema = graph_types.NodeSchema([], [])

    # Add possible edge types
    if node_spec.has_parent:
      node_schema.in_edges.append(graph_types.InEdgeType("parent_in"))
      node_schema.out_edges.append(graph_types.OutEdgeType("parent_out"))

    for field, field_type in node_spec.fields.items():
      if field_type in {FieldType.ONE_CHILD, FieldType.OPTIONAL_CHILD}:
        node_schema.in_edges.append(graph_types.InEdgeType(f"{field}_in"))
        node_schema.out_edges.append(graph_types.OutEdgeType(f"{field}_out"))
        if field_type == FieldType.OPTIONAL_CHILD:
          node_schema.in_edges.append(
              graph_types.InEdgeType(f"{field}_missing"))
      elif field_type in {FieldType.SEQUENCE, FieldType.NONEMPTY_SEQUENCE}:
        seen_sequence_categories.add(node_spec.sequence_item_types[field])
        node_schema.in_edges.append(graph_types.InEdgeType(f"{field}_in"))
        node_schema.out_edges.append(
            graph_types.OutEdgeType(f"{field}_out_all"))
        node_schema.out_edges.append(
            graph_types.OutEdgeType(f"{field}_out_first"))
        node_schema.out_edges.append(
            graph_types.OutEdgeType(f"{field}_out_last"))
        if field_type == FieldType.SEQUENCE:
          node_schema.in_edges.append(
              graph_types.InEdgeType(f"{field}_missing"))
      elif field_type in {FieldType.NO_CHILDREN, FieldType.IGNORE}:
        # No edges for these fields.
        pass
      else:
        raise ValueError(f"Unexpected field type {field_type}")

    result[graph_types.NodeType(node_type)] = node_schema

  # Build node schemas for each category helper
  for category in sorted(seen_sequence_categories):
    helper_type = graph_types.NodeType(f"{category}-seq-helper")
    assert helper_type not in result
    node_schema = graph_types.NodeSchema(
        in_edges=[
            graph_types.InEdgeType("parent_in"),
            graph_types.InEdgeType("item_in"),
            graph_types.InEdgeType("next_in"),
            graph_types.InEdgeType("next_missing"),
            graph_types.InEdgeType("prev_in"),
            graph_types.InEdgeType("prev_missing")
        ],
        out_edges=[
            graph_types.OutEdgeType("parent_out"),
            graph_types.OutEdgeType("item_out"),
            graph_types.OutEdgeType("next_out"),
            graph_types.OutEdgeType("prev_out")
        ])
    result[helper_type] = node_schema

  return result


def ast_to_graph(
    root,
    ast_spec,
    ignore_unknown_fields = False,
):
  """Converts a generic AST into a graph POMDP.

  The graph returned by this function will conform to the schema produced by
  `build_ast_graph_schema` for the same `ast_spec`.

  For each AST node that is processed, we ensure that:
  - the node's type is registered in the schema
  - the node has the expected number of children for each field listed
  We then add all of the edges described in the docstring for
  `build_python_graph_schema`. Finally, we recursively process all children of
  the node for fields that are listed in the schema.

  Depending on `ignore_unknown_fields`, fields that are NOT listed in the schema
  are either ignored or produce an error. Fields marked IGNORE (and, when
  `ignore_unknown_fields=True`, fields not in the schema) will not be traversed,
  and thus any subtrees along those fields will not be present in the output.

  Nodes are given string IDs based on their type and the path from the root.

  TODO(ddjohnson) Handle unexpected node types and spec mismatches using unknown
  helper nodes and edges.

  Args:
    root: AST root node.
    ast_spec: Spec describing the AST structure.
    ignore_unknown_fields: Whether to ignore fields that are not in the spec.

  Returns:
    Graph that conforms to the schema returned by `build_ast_graph_schema`,
    and a mapping from AST `node_id` values to the corresponding node ids in
    the automaton graph.

  Raises:
    ValueError if the AST node does not match the expected format.
  """
  result = {}
  forward_mapping = {}

  def get_graph_node_id(
      type_name,
      path_from_root,
  ):
    """Builds an ID for a graph node, and creates the node if necessary."""
    id_parts = ["root"]
    for part in path_from_root:
      id_parts.extend(("_", str(part)))
    id_parts.extend(("__", type_name))
    node_id = graph_types.NodeId("".join(id_parts))
    if node_id not in result:
      result[node_id] = graph_types.GraphNode(
          node_type=graph_types.NodeType(type_name), out_edges={})
    return node_id

  def connect(from_id, out_type, in_type,
              to_id):
    """Adds directed edges to the graph."""
    out_type = graph_types.OutEdgeType(out_type)
    in_type = graph_types.InEdgeType(in_type)
    if out_type not in result[from_id].out_edges:
      result[from_id].out_edges[out_type] = []
    result[from_id].out_edges[out_type].append(
        graph_types.InputTaggedNode(node_id=to_id, in_edge=in_type))

  def process(ast_node, path_from_root):
    """Processes a node and its children in DFS order."""
    if ast_node.node_type not in ast_spec:
      raise ValueError(f"Unknown AST node type '{ast_node.node_type}'")
    node_id = get_graph_node_id(ast_node.node_type, path_from_root)
    node_spec = ast_spec[ast_node.node_type]

    forward_mapping[ast_node.node_id] = node_id

    for field, field_type in node_spec.fields.items():
      children = ast_node.fields.get(field, ())
      if field_type in {FieldType.ONE_CHILD, FieldType.OPTIONAL_CHILD}:
        if children:
          if len(children) > 1:
            if field_type == FieldType.ONE_CHILD:
              raise ValueError(f"Expected 1 child for field '{field}' "
                               f"of node {ast_node}; got {len(children)}")
            else:
              raise ValueError(f"Expected at most 1 child for field '{field}' "
                               f"of node {ast_node}; got {len(children)}")
          child, = children
          child_path = path_from_root + [field]
          child_id = get_graph_node_id(child.node_type, child_path)
          connect(node_id, f"{field}_out", "parent_in", child_id)
          connect(child_id, "parent_out", f"{field}_in", node_id)
          process(child, child_path)
        elif field_type == FieldType.OPTIONAL_CHILD:
          connect(node_id, f"{field}_out", f"{field}_missing", node_id)
        else:
          raise ValueError(f"Expected 1 child for field '{field}' of node "
                           f"{ast_node}; got 0")

      elif field_type in {FieldType.SEQUENCE, FieldType.NONEMPTY_SEQUENCE}:
        # Add all helpers and non-looping next/prev edges.
        helper_ids = []
        category = node_spec.sequence_item_types[field]
        for i, child in enumerate(children):
          helper_path = path_from_root + [field, i]
          helper_id = get_graph_node_id(f"{category}-seq-helper", helper_path)
          helper_ids.append(helper_id)

          child_path = path_from_root + [field, i, "item"]
          child_id = get_graph_node_id(child.node_type, child_path)

          connect(node_id, f"{field}_out_all", "parent_in", helper_id)
          connect(helper_id, "parent_out", f"{field}_in", node_id)
          connect(helper_id, "item_out", "parent_in", child_id)
          connect(child_id, "parent_out", "item_in", helper_id)
          if i:
            connect(helper_id, "prev_out", "next_in", helper_ids[i - 1])
            connect(helper_ids[i - 1], "next_out", "prev_in", helper_id)

          process(child, child_path)

        # Add boundary edges.
        if children:
          connect(helper_ids[0], "prev_out", "prev_missing", helper_ids[0])
          connect(helper_ids[-1], "next_out", "next_missing", helper_ids[-1])
          connect(node_id, f"{field}_out_first", "parent_in", helper_ids[0])
          connect(node_id, f"{field}_out_last", "parent_in", helper_ids[-1])
        elif field_type == FieldType.SEQUENCE:
          connect(node_id, f"{field}_out_all", f"{field}_missing", node_id)
          connect(node_id, f"{field}_out_first", f"{field}_missing", node_id)
          connect(node_id, f"{field}_out_last", f"{field}_missing", node_id)
        else:
          raise ValueError(f"Expected a nonempty sequence for field '{field}' "
                           f"of node {ast_node} but found empty sequence.")

      elif field_type == FieldType.NO_CHILDREN:
        if children:
          raise ValueError(f"Expected field '{field}' of node {ast_node} to "
                           f"be empty, but found {len(children)} children.")
      elif field_type == FieldType.IGNORE:
        # Don't follow this field.
        pass
      else:
        raise ValueError(f"Unexpected field type {field_type}")

    if not ignore_unknown_fields:
      # Check for unexpected fields.
      for field, children in ast_node.fields.items():
        if field not in node_spec.fields and children:
          raise ValueError(f"Found unknown field {field} on node "
                           f"{ast_node.node_id} of type {ast_node.node_type}")

  process(root, [])
  return result, forward_mapping

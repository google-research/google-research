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
"""Tests for gfsa.datasets.graph_edge_util."""

import textwrap

from absl.testing import absltest
import gast

from gfsa import generic_ast_graphs
from gfsa import graph_types
from gfsa.datasets import graph_edge_util


class GraphEdgeUtilTest(absltest.TestCase):

  def test_compute_jumps_out_edges(self):
    tree = gast.parse(
        textwrap.dedent("""\
          def foo():        # tree.body[0]
            return          # tree.body[0].body[0]
            while True:     # tree.body[0].body[1]
              break         # tree.body[0].body[1].body[0]
              continue      # tree.body[0].body[1].body[1]
              return        # tree.body[0].body[1].body[2]
              while True:   # tree.body[0].body[1].body[3]
                break       # tree.body[0].body[1].body[3].body[0]
                return 4    # tree.body[0].body[1].body[3].body[1]
          """))

    expected_type = graph_edge_util.JUMPS_OUT_OF_EDGE_TYPE
    expected_targets = [
        (tree.body[0].body[0], tree.body[0], expected_type),
        (tree.body[0].body[1].body[0], tree.body[0].body[1], expected_type),
        (tree.body[0].body[1].body[1], tree.body[0].body[1], expected_type),
        (tree.body[0].body[1].body[2], tree.body[0], expected_type),
        (tree.body[0].body[1].body[3].body[0], tree.body[0].body[1].body[3],
         expected_type),
        (tree.body[0].body[1].body[3].body[1], tree.body[0], expected_type),
        (tree.body[0].body[1].body[3].body[1].value, tree.body[0],
         expected_type),
    ]

    # For this test, we pretend that the AST nodes are the node ids.
    targets = graph_edge_util.compute_jumps_out_edges(
        tree, {id(x): x for x in gast.walk(tree)})
    self.assertCountEqual(targets, expected_targets)

  def test_schema_edges(self):
    mini_schema = {
        "foo":
            graph_types.NodeSchema(in_edges=["ignored"], out_edges=["a", "b"]),
        "bar":
            graph_types.NodeSchema(in_edges=["ignored"], out_edges=["b", "c"]),
    }

    mini_graph = {
        "foo_node":
            graph_types.GraphNode(
                "foo", {
                    "a": [graph_types.InputTaggedNode("bar_node", "ignored")],
                    "b": [graph_types.InputTaggedNode("foo_node", "ignored")]
                }),
        "bar_node":
            graph_types.GraphNode(
                "bar", {
                    "b": [graph_types.InputTaggedNode("bar_node", "ignored")],
                    "c": [graph_types.InputTaggedNode("foo_node", "ignored")]
                }),
    }

    schema_edge_types = graph_edge_util.schema_edge_types(
        mini_schema, with_node_types=False)
    self.assertEqual(schema_edge_types, {"SCHEMA_a", "SCHEMA_b", "SCHEMA_c"})

    schema_edges = graph_edge_util.compute_schema_edges(
        mini_graph, with_node_types=False)
    self.assertEqual(schema_edges, [
        ("foo_node", "bar_node", "SCHEMA_a"),
        ("foo_node", "foo_node", "SCHEMA_b"),
        ("bar_node", "bar_node", "SCHEMA_b"),
        ("bar_node", "foo_node", "SCHEMA_c"),
    ])

    schema_edge_types = graph_edge_util.schema_edge_types(
        mini_schema, with_node_types=True)
    self.assertEqual(
        schema_edge_types, {
            "SCHEMA_a_FROM_foo", "SCHEMA_b_FROM_foo", "SCHEMA_b_FROM_bar",
            "SCHEMA_c_FROM_bar"
        })

    schema_edges = graph_edge_util.compute_schema_edges(
        mini_graph, with_node_types=True)
    self.assertEqual(schema_edges, [
        ("foo_node", "bar_node", "SCHEMA_a_FROM_foo"),
        ("foo_node", "foo_node", "SCHEMA_b_FROM_foo"),
        ("bar_node", "bar_node", "SCHEMA_b_FROM_bar"),
        ("bar_node", "foo_node", "SCHEMA_c_FROM_bar"),
    ])

  def test_compute_same_identifier_edges(self):
    list_node = gast.parse("[x, x, x, y, y]").body[0].value
    ast_to_node_id = {
        id(list_node.elts[0]): "x0",
        id(list_node.elts[1]): "x1",
        id(list_node.elts[2]): "x2",
        id(list_node.elts[3]): "y0",
        id(list_node.elts[4]): "y1",
    }

    same_identifier_edges = graph_edge_util.compute_same_identifier_edges(
        list_node, ast_to_node_id)
    self.assertCountEqual(same_identifier_edges, [
        ("x0", "x0", "EXTRA_SAME_IDENTIFIER"),
        ("x0", "x1", "EXTRA_SAME_IDENTIFIER"),
        ("x0", "x2", "EXTRA_SAME_IDENTIFIER"),
        ("x1", "x0", "EXTRA_SAME_IDENTIFIER"),
        ("x1", "x1", "EXTRA_SAME_IDENTIFIER"),
        ("x1", "x2", "EXTRA_SAME_IDENTIFIER"),
        ("x2", "x0", "EXTRA_SAME_IDENTIFIER"),
        ("x2", "x1", "EXTRA_SAME_IDENTIFIER"),
        ("x2", "x2", "EXTRA_SAME_IDENTIFIER"),
        ("y0", "y0", "EXTRA_SAME_IDENTIFIER"),
        ("y0", "y1", "EXTRA_SAME_IDENTIFIER"),
        ("y1", "y0", "EXTRA_SAME_IDENTIFIER"),
        ("y1", "y1", "EXTRA_SAME_IDENTIFIER"),
    ])

  def test_compute_nth_child_edges(self):
    mini_ast_spec = {
        "root":
            generic_ast_graphs.ASTNodeSpec(
                fields={"children": generic_ast_graphs.FieldType.SEQUENCE},
                sequence_item_types={"children": "child"},
                has_parent=False),
        "leaf":
            generic_ast_graphs.ASTNodeSpec()
    }
    mini_ast_node = generic_ast_graphs.GenericASTNode(
        "root", "root", {
            "children": [
                generic_ast_graphs.GenericASTNode("leaf0", "leaf", {}),
                generic_ast_graphs.GenericASTNode("leaf1", "leaf", {}),
                generic_ast_graphs.GenericASTNode("leaf2", "leaf", {}),
                generic_ast_graphs.GenericASTNode("leaf3", "leaf", {}),
                generic_ast_graphs.GenericASTNode("leaf4", "leaf", {}),
            ]
        })
    mini_ast_graph, _ = generic_ast_graphs.ast_to_graph(mini_ast_node,
                                                        mini_ast_spec)

    # Allowing 10 children.
    nth_child_edges = graph_edge_util.compute_nth_child_edges(
        mini_ast_graph, 10)
    self.assertEqual(nth_child_edges,
                     [("root__root", f"root_children_{i}__child-seq-helper",
                       f"CHILD_INDEX_{i}") for i in range(5)])

    # Allowing 2 children.
    nth_child_edges = graph_edge_util.compute_nth_child_edges(mini_ast_graph, 2)
    self.assertEqual(nth_child_edges,
                     [("root__root", f"root_children_{i}__child-seq-helper",
                       f"CHILD_INDEX_{i}") for i in range(2)])

  def test_encode_edges(self):
    edges = [
        ("a", "b", "foo"),
        ("b", "c", "bar"),
        ("c", "a", "foo"),
    ]
    edge_types = ["quz", "foo", "bar"]
    encoded = graph_edge_util.encode_edges(edges, edge_types)
    self.assertEqual(encoded, [
        ("a", "b", 1),
        ("b", "c", 2),
        ("c", "a", 1),
    ])


if __name__ == "__main__":
  absltest.main()

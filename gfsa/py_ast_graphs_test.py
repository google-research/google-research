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
"""Tests for gfsa.py_ast_graphs."""

import textwrap
from absl.testing import absltest
from absl.testing import parameterized
import gast
from gfsa import graph_types
from gfsa import py_ast_graphs
from gfsa import schema_util


class PyASTGraphsTest(parameterized.TestCase):

  def test_schema(self):
    """Check that the generated schema is reasonable."""
    # Building should succeed
    schema = py_ast_graphs.SCHEMA

    # Should have one graph node for each AST node, plus sorted helpers.
    seq_helpers = [
        "BoolOp_values-seq-helper",
        "Call_args-seq-helper",
        "For_body-seq-helper",
        "FunctionDef_body-seq-helper",
        "If_body-seq-helper",
        "If_orelse-seq-helper",
        "Module_body-seq-helper",
        "While_body-seq-helper",
        "arguments_args-seq-helper",
    ]
    expected_keys = [
        *py_ast_graphs.PY_AST_SPECS.keys(),
        *seq_helpers,
    ]
    self.assertEqual(list(schema.keys()), expected_keys)

    # Check a few elements
    expected_fundef_in_edges = {
        "parent_in",
        "returns_in",
        "returns_missing",
        "body_in",
        "args_in",
    }
    expected_fundef_out_edges = {
        "parent_out",
        "returns_out",
        "body_out_all",
        "body_out_first",
        "body_out_last",
        "args_out",
    }
    self.assertEqual(
        set(schema["FunctionDef"].in_edges), expected_fundef_in_edges)
    self.assertEqual(
        set(schema["FunctionDef"].out_edges), expected_fundef_out_edges)

    expected_expr_in_edges = {"parent_in", "value_in"}
    expected_expr_out_edges = {"parent_out", "value_out"}
    self.assertEqual(set(schema["Expr"].in_edges), expected_expr_in_edges)
    self.assertEqual(set(schema["Expr"].out_edges), expected_expr_out_edges)

    expected_seq_helper_in_edges = {
        "parent_in",
        "item_in",
        "next_in",
        "next_missing",
        "prev_in",
        "prev_missing",
    }
    expected_seq_helper_out_edges = {
        "parent_out",
        "item_out",
        "next_out",
        "prev_out",
    }
    for seq_helper in seq_helpers:
      self.assertEqual(
          set(schema[seq_helper].in_edges), expected_seq_helper_in_edges)
      self.assertEqual(
          set(schema[seq_helper].out_edges), expected_seq_helper_out_edges)

  def test_ast_graph_conforms_to_schema(self):
    # Some example code using a few different syntactic constructs, to cover
    # a large set of nodes in the schema
    root = gast.parse(
        textwrap.dedent("""\
        def foo(n):
          if n <= 1:
            return 1
          else:
            return foo(n-1) + foo(n-2)

        def bar(m, n) -> int:
          x = n
          for i in range(m):
            if False:
              continue
            x = x + i
          while True:
            break
          return x

        x0 = 1 + 2 - 3 * 4 / 5
        x1 = (1 == 2) and (3 < 4) and (5 > 6)
        x2 = (7 <= 8) and (9 >= 10) or (11 != 12)
        x2 = bar(13, 14 + 15)
        """))

    graph, _ = py_ast_graphs.py_ast_to_graph(root)

    # Graph should match the schema
    schema_util.assert_conforms_to_schema(graph, py_ast_graphs.SCHEMA)

  def test_ast_graph_nodes(self):
    """Check node IDs, node types, and forward mapping."""
    root = gast.parse(
        textwrap.dedent("""\
        pass
        def foo(n):
            if n <= 1:
              return 1
        """))

    graph, forward_map = py_ast_graphs.py_ast_to_graph(root)

    # pytype: disable=attribute-error
    self.assertIn("root__Module", graph)
    self.assertEqual(graph["root__Module"].node_type, "Module")
    self.assertEqual(forward_map[id(root)], "root__Module")

    self.assertIn("root_body_1__Module_body-seq-helper", graph)
    self.assertEqual(graph["root_body_1__Module_body-seq-helper"].node_type,
                     "Module_body-seq-helper")

    self.assertIn("root_body_1_item_body_0_item__If", graph)
    self.assertEqual(graph["root_body_1_item_body_0_item__If"].node_type, "If")
    self.assertEqual(forward_map[id(root.body[1].body[0])],
                     "root_body_1_item_body_0_item__If")

    self.assertIn("root_body_1_item_body_0_item_test_left__Name", graph)
    self.assertEqual(
        graph["root_body_1_item_body_0_item_test_left__Name"].node_type, "Name")
    self.assertEqual(forward_map[id(root.body[1].body[0].test.left)],
                     "root_body_1_item_body_0_item_test_left__Name")
    # pytype: enable=attribute-error

  def test_ast_graph_unique_field_edges(self):
    """Test that edges for unique fields are correct."""
    root = gast.parse("print(1)")
    graph, _ = py_ast_graphs.py_ast_to_graph(root)

    self.assertEqual(graph["root_body_0_item__Expr"].out_edges["value_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_0_item_value__Call"),
            in_edge=graph_types.InEdgeType("parent_in"))
    ])

    self.assertEqual(
        graph["root_body_0_item_value__Call"].out_edges["parent_out"], [
            graph_types.InputTaggedNode(
                node_id=graph_types.NodeId("root_body_0_item__Expr"),
                in_edge=graph_types.InEdgeType("value_in"))
        ])

  def test_ast_graph_optional_field_edges(self):
    """Test that edges for optional fields are correct."""
    root = gast.parse("return 1\nreturn")
    graph, _ = py_ast_graphs.py_ast_to_graph(root)

    self.assertEqual(graph["root_body_0_item__Return"].out_edges["value_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_0_item_value__Constant"),
            in_edge=graph_types.InEdgeType("parent_in"))
    ])

    self.assertEqual(
        graph["root_body_0_item_value__Constant"].out_edges["parent_out"], [
            graph_types.InputTaggedNode(
                node_id=graph_types.NodeId("root_body_0_item__Return"),
                in_edge=graph_types.InEdgeType("value_in"))
        ])

    self.assertEqual(graph["root_body_1_item__Return"].out_edges["value_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_1_item__Return"),
            in_edge=graph_types.InEdgeType("value_missing"))
    ])

  def test_ast_graph_sequence_field_edges(self):
    """Test that edges for sequence fields are correct.

    Note that sequence fields produce connections between three nodes: the
    parent, the helper node, and the child.
    """
    root = gast.parse(
        textwrap.dedent("""\
        print(1)
        print(2)
        print(3)
        print(4)
        print(5)
        print(6)
        """))

    graph, _ = py_ast_graphs.py_ast_to_graph(root)

    # Child edges from the parent node
    node = graph["root__Module"]
    self.assertLen(node.out_edges["body_out_all"], 6)
    self.assertEqual(node.out_edges["body_out_first"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_0__Module_body-seq-helper"),
            in_edge=graph_types.InEdgeType("parent_in"))
    ])
    self.assertEqual(node.out_edges["body_out_last"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_5__Module_body-seq-helper"),
            in_edge=graph_types.InEdgeType("parent_in"))
    ])

    # Edges from the sequence helper
    node = graph["root_body_0__Module_body-seq-helper"]
    self.assertEqual(node.out_edges["parent_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root__Module"),
            in_edge=graph_types.InEdgeType("body_in"))
    ])
    self.assertEqual(node.out_edges["item_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_0_item__Expr"),
            in_edge=graph_types.InEdgeType("parent_in"))
    ])
    self.assertEqual(node.out_edges["prev_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_0__Module_body-seq-helper"),
            in_edge=graph_types.InEdgeType("prev_missing"))
    ])
    self.assertEqual(node.out_edges["next_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_1__Module_body-seq-helper"),
            in_edge=graph_types.InEdgeType("prev_in"))
    ])

    # Parent edge of the item
    node = graph["root_body_0_item__Expr"]
    self.assertEqual(node.out_edges["parent_out"], [
        graph_types.InputTaggedNode(
            node_id=graph_types.NodeId("root_body_0__Module_body-seq-helper"),
            in_edge=graph_types.InEdgeType("item_in"))
    ])

  @parameterized.named_parameters(
      {
          "testcase_name": "unexpected_type",
          "ast": gast.Subscript(value=None, slice=None, ctx=None),
          "expected_error": "Unknown AST node type 'Subscript'",
      }, {
          "testcase_name":
              "too_many_unique",
          "ast":
              gast.Assign(
                  targets=[
                      gast.Name("foo", gast.Store(), None, None),
                      gast.Name("bar", gast.Store(), None, None)
                  ],
                  value=gast.Constant(True, None)),
          "expected_error":
              "Expected 1 child for field 'targets' of node .*; got 2",
      }, {
          "testcase_name":
              "missing_unique",
          "ast":
              gast.Assign(targets=[], value=gast.Constant(True, None)),
          "expected_error":
              "Expected 1 child for field 'targets' of node .*; got 0",
      }, {
          "testcase_name":
              "too_many_optional",
          "ast":
              gast.Return(value=[
                  gast.Name("foo", gast.Load(), None, None),
                  gast.Name("bar", gast.Load(), None, None)
              ]),
          "expected_error":
              "Expected at most 1 child for field 'value' of node .*; got 2",
      })
  def test_invalid_graphs(self, ast, expected_error):
    with self.assertRaisesRegex(ValueError, expected_error):
      py_ast_graphs.py_ast_to_graph(ast)


if __name__ == "__main__":
  absltest.main()

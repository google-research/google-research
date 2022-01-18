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
"""Tests for gfsa.datasets.graph_bundle."""

import textwrap
from absl.testing import absltest
import gast
import jax
import numpy as np
from gfsa import automaton_builder
from gfsa import py_ast_graphs
from gfsa.datasets import graph_bundle


class GraphBundleTest(absltest.TestCase):

  def test_convert_example(self):
    tree = gast.parse(
        textwrap.dedent("""\
          def foo():
            x = 5
            return x
          """))

    py_graph, ast_to_node_id = (py_ast_graphs.py_ast_to_graph(tree))
    ast_edges = [
        (tree.body[0].body[1], tree.body[0], 1),
        (tree.body[0].body[1].value, tree.body[0].body[0].targets[0], 2),
    ]
    converted_edges = [(ast_to_node_id[id(source)], ast_to_node_id[id(dest)],
                        edge_type) for (source, dest, edge_type) in ast_edges]
    example = graph_bundle.convert_graph_with_edges(
        py_graph, converted_edges, builder=py_ast_graphs.BUILDER)

    self.assertEqual(
        list(py_graph), [
            "root__Module",
            "root_body_0__Module_body-seq-helper",
            "root_body_0_item__FunctionDef",
            "root_body_0_item_args__arguments",
            "root_body_0_item_body_0__FunctionDef_body-seq-helper",
            "root_body_0_item_body_0_item__Assign",
            "root_body_0_item_body_0_item_targets__Name",
            "root_body_0_item_body_0_item_value__Constant",
            "root_body_0_item_body_1__FunctionDef_body-seq-helper",
            "root_body_0_item_body_1_item__Return",
            "root_body_0_item_body_1_item_value__Name",
        ])

    self.assertEqual(
        example.graph_metadata,
        automaton_builder.EncodedGraphMetadata(
            num_nodes=11, num_input_tagged_nodes=27))

    self.assertEqual(example.node_types.shape, (11,))

    np.testing.assert_array_equal(example.edges.input_indices, [[1], [2]])
    np.testing.assert_array_equal(example.edges.output_indices,
                                  [[9, 2], [10, 6]])
    np.testing.assert_array_equal(example.edges.values, [1, 1])

    self.assertEqual(example.automaton_graph.initial_to_in_tagged.values.shape,
                     (34,))
    self.assertEqual(example.automaton_graph.initial_to_special.shape, (11,))
    self.assertEqual(
        example.automaton_graph.in_tagged_to_in_tagged.values.shape, (103,))
    self.assertEqual(example.automaton_graph.in_tagged_to_special.shape, (27,))

    # Verify that the transition matrix can be built with the right size.
    routing_params = py_ast_graphs.BUILDER.initialize_routing_params(
        None, 1, 1, noise_factor=0)
    transition_matrix = py_ast_graphs.BUILDER.build_transition_matrix(
        routing_params, example.automaton_graph, example.graph_metadata)

    self.assertEqual(transition_matrix.initial_to_in_tagged.shape,
                     (1, 11, 1, 27, 1))
    self.assertEqual(transition_matrix.initial_to_special.shape, (1, 11, 1, 3))
    self.assertEqual(transition_matrix.in_tagged_to_in_tagged.shape,
                     (1, 27, 1, 27, 1))
    self.assertEqual(transition_matrix.in_tagged_to_special.shape,
                     (1, 27, 1, 3))
    self.assertEqual(transition_matrix.in_tagged_node_indices.shape, (27,))

  def test_convert_no_targets(self):
    tree = gast.parse(
        textwrap.dedent("""\
          def foo():
            x = 5
            return x
          """))

    py_graph, _ = py_ast_graphs.py_ast_to_graph(tree)
    example = graph_bundle.convert_graph_with_edges(
        py_graph, [], builder=py_ast_graphs.BUILDER)

    # Target indices should still be a valid operator, but with no nonzero
    # entries.
    self.assertEqual(example.edges.input_indices.shape, (0, 1))
    self.assertEqual(example.edges.output_indices.shape, (0, 2))
    self.assertEqual(example.edges.values.shape, (0,))

  def test_pad_example(self):
    tree = gast.parse(
        textwrap.dedent("""\
          def foo():
            x = 5
            return x
          """))

    py_graph, ast_to_node_id = (py_ast_graphs.py_ast_to_graph(tree))
    ast_edges = [
        (tree.body[0].body[1], tree.body[0], 1),
        (tree.body[0].body[1].value, tree.body[0].body[0].targets[0], 2),
    ]
    converted_edges = [(ast_to_node_id[id(source)], ast_to_node_id[id(dest)],
                        edge_type) for (source, dest, edge_type) in ast_edges]
    example = graph_bundle.convert_graph_with_edges(
        py_graph, converted_edges, builder=py_ast_graphs.BUILDER)

    padding_config = graph_bundle.PaddingConfig(
        static_max_metadata=automaton_builder.EncodedGraphMetadata(
            num_nodes=16, num_input_tagged_nodes=34),
        max_initial_transitions=64,
        max_in_tagged_transitions=128,
        max_edges=4)

    padded_example = graph_bundle.pad_example(example, padding_config)

    # Metadata is not affected by padding.
    self.assertEqual(
        padded_example.graph_metadata,
        automaton_builder.EncodedGraphMetadata(
            num_nodes=11, num_input_tagged_nodes=27))

    # Everything else is padded.
    self.assertEqual(padded_example.node_types.shape, (16,))

    np.testing.assert_array_equal(padded_example.edges.input_indices,
                                  [[1], [2], [0], [0]])
    np.testing.assert_array_equal(padded_example.edges.output_indices,
                                  [[9, 2], [10, 6], [0, 0], [0, 0]])
    np.testing.assert_array_equal(padded_example.edges.values, [1, 1, 0, 0])

    self.assertEqual(
        padded_example.automaton_graph.initial_to_in_tagged.values.shape, (64,))
    self.assertEqual(padded_example.automaton_graph.initial_to_special.shape,
                     (16,))
    self.assertEqual(
        padded_example.automaton_graph.in_tagged_to_in_tagged.values.shape,
        (128,))
    self.assertEqual(padded_example.automaton_graph.in_tagged_to_special.shape,
                     (34,))

    # Transition matrix also becomes padded once it is built.
    # (Note that we pass the padded static metadata to the transition matrix
    # builder, since the encoded graph has been padded.)
    routing_params = py_ast_graphs.BUILDER.initialize_routing_params(
        None, 1, 1, noise_factor=0)
    transition_matrix = py_ast_graphs.BUILDER.build_transition_matrix(
        routing_params, padded_example.automaton_graph,
        padding_config.static_max_metadata)

    self.assertEqual(transition_matrix.initial_to_in_tagged.shape,
                     (1, 16, 1, 34, 1))
    self.assertEqual(transition_matrix.initial_to_special.shape, (1, 16, 1, 3))
    self.assertEqual(transition_matrix.in_tagged_to_in_tagged.shape,
                     (1, 34, 1, 34, 1))
    self.assertEqual(transition_matrix.in_tagged_to_special.shape,
                     (1, 34, 1, 3))
    self.assertEqual(transition_matrix.in_tagged_node_indices.shape, (34,))

  def test_zeros_like_padded_example(self):
    tree = gast.parse("pass")
    py_graph, _ = py_ast_graphs.py_ast_to_graph(tree)
    example = graph_bundle.convert_graph_with_edges(
        py_graph, [], builder=py_ast_graphs.BUILDER)

    padding_config = graph_bundle.PaddingConfig(
        static_max_metadata=automaton_builder.EncodedGraphMetadata(
            num_nodes=16, num_input_tagged_nodes=34),
        max_initial_transitions=64,
        max_in_tagged_transitions=128,
        max_edges=4)

    padded_example = graph_bundle.pad_example(example, padding_config)
    generated = graph_bundle.zeros_like_padded_example(padding_config)

    def _check(x, y):
      x = np.asarray(x)
      y = np.asarray(y)
      self.assertEqual(x.shape, y.shape)
      self.assertEqual(x.dtype, y.dtype)

    jax.tree_multimap(_check, generated, padded_example)


if __name__ == "__main__":
  absltest.main()

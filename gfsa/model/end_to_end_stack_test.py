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
"""Tests for gfsa.model.end_to_end_stack."""

from absl.testing import absltest
from absl.testing import parameterized
import gin
import jax
import jax.numpy as jnp
from gfsa import automaton_builder
from gfsa import graph_types
from gfsa.datasets import graph_bundle
from gfsa.model import end_to_end_stack

NODE_DIM = 10
EDGE_DIM = 20

CONFIG = """\
automaton_layer.FiniteStateGraphAutomaton.num_out_edges = 3
automaton_layer.FiniteStateGraphAutomaton.num_intermediate_states = 2
automaton_layer.FiniteStateGraphAutomaton.steps = 4

end_to_end_stack.edge_variant_automaton.variant_edge_types = ["foo"]

end_to_end_stack.embedding_variant_automaton.num_variants = 3

edge_supervision_models.transformer_steps.layers = 2
graph_layers.NodeSelfAttention.heads = 2
graph_layers.NodeSelfAttention.query_key_dim = 16
graph_layers.NodeSelfAttention.value_dim = 16

edge_supervision_models.ggnn_steps.iterations = 2
graph_layers.LinearMessagePassing.message_dim = 32

edge_supervision_models.nri_steps.layers = 2
edge_supervision_models.nri_steps.mlp_etov_dims = [256, 256]
edge_supervision_models.nri_steps.with_residual_layer_norm = True
graph_layers.NRIEdgeLayer.allow_non_adjacent = True
graph_layers.NRIEdgeLayer.mlp_vtoe_dims = [32, 32]
end_to_end_stack.nri_encoder_readout.num_edge_types = 3
"""


class EndToEndStackTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "variantless_automaton_stacked",
          "component": "variantless_automaton",
          "embed_edges": False,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM + 3 * 2,  # Bidirectional edges
          },
      },
      {
          "testcase_name": "variantless_automaton_embedded",
          "component": "variantless_automaton",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      },
      {
          "testcase_name": "edge_variant_automaton_stacked",
          "component": "edge_variant_automaton",
          "embed_edges": False,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM + 3 * 2,  # Bidirectional edges
          },
      },
      {
          "testcase_name": "edge_variant_automaton_embedded",
          "component": "edge_variant_automaton",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      },
      {
          "testcase_name": "embedding_variant_automaton_stacked",
          "component": "embedding_variant_automaton",
          "embed_edges": False,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM + 3 * 2,  # Bidirectional edges
          },
      },
      {
          "testcase_name": "embedding_variant_automaton_embedded",
          "component": "embedding_variant_automaton",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      },
      {
          "testcase_name": "nri_encoder_readout_stacked",
          "component": "nri_encoder_readout",
          "embed_edges": False,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM + 3 * 2,  # Bidirectional edges
          },
      },
      {
          "testcase_name": "nri_encoder_readout_embedded",
          "component": "nri_encoder_readout",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      },
      {
          "testcase_name": "ggnn_adapter",
          "component": "ggnn_adapter",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      },
      {
          "testcase_name": "transformer_adapter",
          "component": "transformer_adapter",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      },
      {
          "testcase_name": "nri_adapter",
          "component": "nri_adapter",
          "embed_edges": True,
          "expected_dims": {
              "node": NODE_DIM,
              "edge": EDGE_DIM,
          },
      })
  def test_component_shapes(self,
                            component,
                            embed_edges,
                            expected_dims,
                            extra_config=None):
    gin.clear_config()
    gin.parse_config(CONFIG)
    if extra_config:
      gin.parse_config(extra_config)

    # Run the computation with placeholder inputs.
    (node_out, edge_out), _ = end_to_end_stack.ALL_COMPONENTS[component].init(
        jax.random.PRNGKey(0),
        graph_context=end_to_end_stack.SharedGraphContext(
            bundle=graph_bundle.zeros_like_padded_example(
                graph_bundle.PaddingConfig(
                    static_max_metadata=automaton_builder.EncodedGraphMetadata(
                        num_nodes=16, num_input_tagged_nodes=32),
                    max_initial_transitions=11,
                    max_in_tagged_transitions=12,
                    max_edges=13)),
            static_metadata=automaton_builder.EncodedGraphMetadata(
                num_nodes=16, num_input_tagged_nodes=32),
            edge_types_to_indices={"foo": 0},
            builder=automaton_builder.AutomatonBuilder({
                graph_types.NodeType("node"):
                    graph_types.NodeSchema(
                        in_edges=[graph_types.InEdgeType("in")],
                        out_edges=[graph_types.InEdgeType("out")])
            }),
            edges_are_embedded=embed_edges),
        node_embeddings=jnp.zeros((16, NODE_DIM)),
        edge_embeddings=jnp.zeros((16, 16, EDGE_DIM)))

    self.assertEqual(node_out.shape, (16, expected_dims["node"]))
    self.assertEqual(edge_out.shape, (16, 16, expected_dims["edge"]))


if __name__ == "__main__":
  absltest.main()

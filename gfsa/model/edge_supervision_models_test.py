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

"""Tests for gfsa.model.edge_supervision_models."""

import functools
import textwrap
from absl.testing import absltest
from absl.testing import parameterized
import dataclasses
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
from gfsa import automaton_builder
from gfsa import sparse_operator
from gfsa.datasets import graph_bundle
from gfsa.model import edge_supervision_models


class EdgeSupervisionModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()

  def test_variants_from_edges(self):
    example = graph_bundle.zeros_like_padded_example(
        graph_bundle.PaddingConfig(
            static_max_metadata=automaton_builder.EncodedGraphMetadata(
                num_nodes=5, num_input_tagged_nodes=0),
            max_initial_transitions=0,
            max_in_tagged_transitions=0,
            max_edges=8))
    example = dataclasses.replace(
        example,
        graph_metadata=automaton_builder.EncodedGraphMetadata(
            num_nodes=4, num_input_tagged_nodes=0),
        edges=sparse_operator.SparseCoordOperator(
            input_indices=jnp.array([[0], [0], [0], [1], [1], [2], [0], [0]]),
            output_indices=jnp.array([[1, 2], [2, 3], [3, 0], [2, 0], [0, 2],
                                      [0, 3], [0, 0], [0, 0]]),
            values=jnp.array([1, 1, 1, 1, 1, 1, 0, 0])))

    weights = edge_supervision_models.variants_from_edges(
        example,
        automaton_builder.EncodedGraphMetadata(
            num_nodes=5, num_input_tagged_nodes=0),
        variant_edge_type_indices=[2, 0],
        num_edge_types=3)
    expected = np.array([
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    ], np.float32)
    # Only assert on the non-padded part.
    np.testing.assert_allclose(weights[:4, :4], expected)

  def test_ggtnn_steps(self):
    gin.parse_config(
        textwrap.dedent("""\
            edge_supervision_models.ggnn_steps.iterations = 10
            graph_layers.LinearMessagePassing.message_dim = 5
            """))

    _, params = edge_supervision_models.ggnn_steps.init(
        jax.random.PRNGKey(0),
        node_embeddings=jnp.zeros((5, 3), jnp.float32),
        edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32))

    # This component should only contain one step block, with two sublayers.
    self.assertEqual(set(params.keys()), {"step"})
    self.assertLen(params["step"], 2)

    # Gradients should work.
    outs, vjpfun = jax.vjp(
        functools.partial(
            edge_supervision_models.ggnn_steps.call,
            node_embeddings=jnp.zeros((5, 3), jnp.float32),
            edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32)),
        params,
    )
    vjpfun(outs)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "shared",
          "expected_block_count":
              1,
          "config":
              textwrap.dedent("""\
                  transformer_steps.layers = 3
                  transformer_steps.share_weights = True
                  transformer_steps.mask_to_neighbors = False
                  NodeSelfAttention.heads = 2
                  NodeSelfAttention.query_key_dim = 3
                  NodeSelfAttention.value_dim = 4
                  """),
      }, {
          "testcase_name":
              "unshared",
          "expected_block_count":
              3,
          "config":
              textwrap.dedent("""\
                  transformer_steps.layers = 3
                  transformer_steps.share_weights = False
                  transformer_steps.mask_to_neighbors = False
                  NodeSelfAttention.heads = 2
                  NodeSelfAttention.query_key_dim = 3
                  NodeSelfAttention.value_dim = 4
                  """),
      }, {
          "testcase_name":
              "shared_masked",
          "expected_block_count":
              1,
          "config":
              textwrap.dedent("""\
                  transformer_steps.layers = 3
                  transformer_steps.share_weights = True
                  transformer_steps.mask_to_neighbors = True
                  NodeSelfAttention.heads = 2
                  NodeSelfAttention.query_key_dim = 3
                  NodeSelfAttention.value_dim = 4
                  """),
      })
  def test_transformer_steps(self, config, expected_block_count):
    gin.parse_config(config)

    _, params = edge_supervision_models.transformer_steps.init(
        jax.random.PRNGKey(0),
        node_embeddings=jnp.zeros((5, 3), jnp.float32),
        edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32),
        neighbor_mask=jnp.zeros((5, 5), jnp.float32),
        num_real_nodes_per_graph=4)

    # This component should contain the right number of blocks.
    self.assertLen(params, expected_block_count)
    for block in params.values():
      # Each block contains 4 sublayers.
      self.assertLen(block, 4)

    # Gradients should work.
    outs, vjpfun = jax.vjp(
        functools.partial(
            edge_supervision_models.transformer_steps.call,
            node_embeddings=jnp.zeros((5, 3), jnp.float32),
            edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32),
            neighbor_mask=jnp.zeros((5, 5), jnp.float32),
            num_real_nodes_per_graph=4),
        params,
    )
    vjpfun(outs)

  def test_transformer_steps_masking(self):
    """Transformer should mask out padding even if not masked to neigbors."""
    gin.parse_config(
        textwrap.dedent("""\
            transformer_steps.layers = 1
            transformer_steps.share_weights = False
            transformer_steps.mask_to_neighbors = False
            NodeSelfAttention.heads = 2
            NodeSelfAttention.query_key_dim = 3
            NodeSelfAttention.value_dim = 4
            """))

    with flax.deprecated.nn.capture_module_outputs() as outputs:
      edge_supervision_models.transformer_steps.init(
          jax.random.PRNGKey(0),
          node_embeddings=jnp.zeros((5, 3), jnp.float32),
          edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32),
          neighbor_mask=jnp.zeros((5, 5), jnp.float32),
          num_real_nodes_per_graph=4)

    attention_weights, = (v[0]
                          for k, v in outputs.as_dict().items()
                          if k.endswith("attend/attention_weights"))
    expected = np.array([[[0.25, 0.25, 0.25, 0.25, 0.0]] * 5] * 2)
    np.testing.assert_allclose(attention_weights, expected)

  def test_nri_steps(self):
    gin.parse_config(
        textwrap.dedent("""\
            graph_layers.NRIEdgeLayer.allow_non_adjacent = True
            graph_layers.NRIEdgeLayer.mlp_vtoe_dims = [4, 4]
            nri_steps.mlp_etov_dims = [8, 8]
            nri_steps.with_residual_layer_norm = True
            nri_steps.layers = 3
            """))

    _, params = edge_supervision_models.nri_steps.init(
        jax.random.PRNGKey(0),
        node_embeddings=jnp.zeros((5, 3), jnp.float32),
        edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32),
        num_real_nodes_per_graph=4)

    # This component should contain the right number of blocks.
    self.assertLen(params, 3)
    for block in params.values():
      # Each block contains 5 sublayers:
      # - NRI message pass
      # - Three dense layers (from mlp_etov_dims, then back to embedding space)
      # - Layer norm
      self.assertLen(block, 5)

    # Gradients should work.
    outs, vjpfun = jax.vjp(
        functools.partial(
            edge_supervision_models.nri_steps.call,
            node_embeddings=jnp.zeros((5, 3), jnp.float32),
            edge_embeddings=jnp.zeros((5, 5, 4), jnp.float32),
            num_real_nodes_per_graph=4),
        params,
    )
    vjpfun(outs)


if __name__ == "__main__":
  absltest.main()

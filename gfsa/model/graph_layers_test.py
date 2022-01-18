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
"""Tests for gfsa.model.graph_layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from gfsa import sparse_operator
from gfsa.model import graph_layers

# Some arbitrary shapes that don't match up with each other, to expose shape
# mismatches in the code.
EDGE_EMBEDDING_DIM = 3
MESSAGE_DIM = 4
NODE_EMBEDDING_DIM = 6  # must be even
NUM_EDGE_TYPES = 5
NUM_EDGES = 7
NUM_HEADS = 8
NUM_NODE_TYPES = 9
NUM_NODES = 10
QUERY_KEY_DIM = 11
VALUE_DIM = 12
NUM_TOKENS = 13
VOCAB_SIZE = 14
NRI_HIDDEN_DIMS = (11, 12)


class GraphLayersTest(parameterized.TestCase):

  def _check_shape_and_dtype(self, actual, desired):
    self.assertEqual(actual.shape, desired.shape)
    self.assertEqual(actual.dtype, desired.dtype)

  def test_NodeTypeNodeEmbedding_shapes(self):
    outs, _ = graph_layers.NodeTypeNodeEmbedding.init(
        jax.random.PRNGKey(0),
        node_types=jnp.zeros((NUM_NODES,), jnp.int32),
        num_node_types=NUM_NODE_TYPES,
        embedding_dim=NODE_EMBEDDING_DIM)

    expected = jax.ShapeDtypeStruct((NUM_NODES, NODE_EMBEDDING_DIM),
                                    jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_PositionalAndTypeNodeEmbedding_shapes(self):
    outs, _ = graph_layers.PositionalAndTypeNodeEmbedding.init(
        jax.random.PRNGKey(0),
        node_types=jnp.zeros((NUM_NODES,), jnp.int32),
        num_node_types=NUM_NODE_TYPES,
        embedding_dim=NODE_EMBEDDING_DIM,
        period_scale=512)

    expected = jax.ShapeDtypeStruct((NUM_NODES, NODE_EMBEDDING_DIM),
                                    jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  @parameterized.parameters(None, 3)
  def test_TokenOperatorNodeEmbedding_shapes(self, bottleneck_dim):
    outs, _ = graph_layers.TokenOperatorNodeEmbedding.init(
        jax.random.PRNGKey(0),
        operator=sparse_operator.SparseCoordOperator(
            input_indices=jnp.zeros((NUM_TOKENS, 1), jnp.int32),
            output_indices=jnp.zeros((NUM_TOKENS, 1), jnp.int32),
            values=jnp.zeros((NUM_TOKENS,), jnp.int32)),
        vocab_size=VOCAB_SIZE,
        num_nodes=NUM_NODES,
        embedding_dim=NODE_EMBEDDING_DIM,
        bottleneck_dim=bottleneck_dim)

    expected = jax.ShapeDtypeStruct((NUM_NODES, NODE_EMBEDDING_DIM),
                                    jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_LearnableEdgeEmbeddings_shapes(self):
    outs, _ = graph_layers.LearnableEdgeEmbeddings.init(
        jax.random.PRNGKey(0),
        edges=sparse_operator.SparseCoordOperator(
            input_indices=jnp.zeros((NUM_EDGES, 1), jnp.int32),
            output_indices=jnp.zeros((NUM_EDGES, 2), jnp.int32),
            values=jnp.zeros((NUM_EDGES,), jnp.int32)),
        num_nodes=NUM_NODES,
        num_edge_types=NUM_EDGE_TYPES,
        forward_edge_type_indices=[0, 2],
        reverse_edge_type_indices=[3, 1],
        embedding_dim=EDGE_EMBEDDING_DIM)

    expected = jax.ShapeDtypeStruct((NUM_NODES, NUM_NODES, EDGE_EMBEDDING_DIM),
                                    jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_LinearMessagePassing_shapes(self):
    outs, _ = graph_layers.LinearMessagePassing.init(
        jax.random.PRNGKey(0),
        edge_embeddings=jnp.zeros((NUM_NODES, NUM_NODES, EDGE_EMBEDDING_DIM),
                                  jnp.float32),
        node_embeddings=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32),
        message_dim=MESSAGE_DIM,
        with_bias=True)

    expected = jax.ShapeDtypeStruct((NUM_NODES, MESSAGE_DIM), jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_LinearMessagePassing_efficient_conv(self):
    edge_embeddings = jax.random.normal(
        jax.random.PRNGKey(0), (NUM_NODES, NUM_NODES, EDGE_EMBEDDING_DIM))
    node_embeddings = jax.random.normal(
        jax.random.PRNGKey(1), (NUM_NODES, NODE_EMBEDDING_DIM))

    simple_outs, params = graph_layers.LinearMessagePassing.init(
        jax.random.PRNGKey(2),
        edge_embeddings=edge_embeddings,
        node_embeddings=node_embeddings,
        message_dim=MESSAGE_DIM,
        use_efficient_conv=False)

    efficient_outs = graph_layers.LinearMessagePassing.call(
        params,
        edge_embeddings=edge_embeddings,
        node_embeddings=node_embeddings,
        message_dim=MESSAGE_DIM,
        use_efficient_conv=True)

    np.testing.assert_allclose(efficient_outs, simple_outs, rtol=1e-5)

  @parameterized.parameters("like_great", "full_relative")
  def test_NodeSelfAttention_shapes(self, which):
    like_great = {"like_great": True, "full_relative": False}[which]
    outs, _ = graph_layers.NodeSelfAttention.init(
        jax.random.PRNGKey(0),
        edge_embeddings=jnp.zeros((NUM_NODES, NUM_NODES, EDGE_EMBEDDING_DIM),
                                  jnp.float32),
        node_embeddings=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32),
        heads=NUM_HEADS,
        query_key_dim=QUERY_KEY_DIM,
        value_dim=VALUE_DIM,
        out_dim=MESSAGE_DIM,
        mask=jnp.zeros((NUM_NODES, NUM_NODES), jnp.float32),
        like_great=like_great)

    expected = jax.ShapeDtypeStruct((NUM_NODES, MESSAGE_DIM), jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  @parameterized.parameters((which, message_passing)  # pylint:disable=g-complex-comprehension
                            for which in ("edges_only", "pairwise_only", "both")
                            for message_passing in (True, False))
  def test_NRIEdgeLayer_message_passing_shapes(self, which, message_passing):
    outs, _ = graph_layers.NRIEdgeLayer.init(
        jax.random.PRNGKey(0),
        edge_embeddings=None if which == "pairwise_only" else jnp.zeros(
            (NUM_NODES, NUM_NODES, EDGE_EMBEDDING_DIM), jnp.float32),
        node_embeddings=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32),
        mlp_vtoe_dims=NRI_HIDDEN_DIMS + (MESSAGE_DIM,),
        mask=jnp.zeros((NUM_NODES, NUM_NODES), jnp.float32),
        allow_non_adjacent=(which != "edges_only"),
        message_passing=message_passing)

    if message_passing:
      expected = jax.ShapeDtypeStruct((NUM_NODES, MESSAGE_DIM), jnp.float32)
    else:
      expected = jax.ShapeDtypeStruct((NUM_NODES, NUM_NODES, MESSAGE_DIM),
                                      jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_residual_layer_norm_update_shapes(self):
    outs, _ = graph_layers.residual_layer_norm_update.init(
        jax.random.PRNGKey(0),
        node_states=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32),
        messages=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32))

    expected = jax.ShapeDtypeStruct((NUM_NODES, NODE_EMBEDDING_DIM),
                                    jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_gated_recurrent_update_shapes(self):
    outs, _ = graph_layers.gated_recurrent_update.init(
        jax.random.PRNGKey(0),
        node_states=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32),
        messages=jnp.zeros((NUM_NODES, MESSAGE_DIM), jnp.float32))

    expected = jax.ShapeDtypeStruct((NUM_NODES, NODE_EMBEDDING_DIM),
                                    jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def test_BilinearPairwiseReadout_shapes(self):
    outs, _ = graph_layers.BilinearPairwiseReadout.init(
        jax.random.PRNGKey(0),
        node_embeddings=jnp.zeros((NUM_NODES, NODE_EMBEDDING_DIM), jnp.float32))

    expected = jax.ShapeDtypeStruct((NUM_NODES, NUM_NODES), jnp.float32)
    self._check_shape_and_dtype(outs, expected)

  def _setup_edges(self):
    """Set up an edge operator with type indices."""
    edges = sparse_operator.SparseCoordOperator(
        input_indices=jnp.array([[0], [0], [1], [2], [0], [1], [0]]),
        output_indices=jnp.array([[0, 1], [1, 2], [2, 3], [2, 0], [0, 2],
                                  [1, 3], [0, 0]]),
        values=jnp.array([1, 1, 1, 1, 1, 1, 0]))
    forward_edge_type_indices = [2, 0]
    reverse_edge_type_indices = [0]
    return edges, forward_edge_type_indices, reverse_edge_type_indices

  def test_binary_index_edge_embeddings(self):
    (edges, forward_edge_type_indices,
     reverse_edge_type_indices) = self._setup_edges()
    result = graph_layers.binary_index_edge_embeddings(
        edges,
        num_nodes=4,
        num_edge_types=3,
        forward_edge_type_indices=forward_edge_type_indices,
        reverse_edge_type_indices=reverse_edge_type_indices)

    expected = np.zeros([4, 4, 3], np.float32)
    expected[2, 0, 0] = 1
    expected[0, 1, 1] = 1
    expected[1, 2, 1] = 1
    expected[0, 2, 1] = 1
    expected[1, 0, 2] = 1
    expected[2, 1, 2] = 1
    expected[2, 0, 2] = 1
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters("edge_mask", "LearnableEdgeEmbeddings")
  def test_edge_nonzeros(self, which):
    (edges, forward_edge_type_indices,
     reverse_edge_type_indices) = self._setup_edges()
    if which == "edge_mask":
      nonzeros = graph_layers.edge_mask(
          edges,
          num_nodes=4,
          num_edge_types=3,
          forward_edge_type_indices=forward_edge_type_indices,
          reverse_edge_type_indices=reverse_edge_type_indices)
    elif which == "LearnableEdgeEmbeddings":
      embeddings, _ = graph_layers.LearnableEdgeEmbeddings.init(
          jax.random.PRNGKey(0),
          edges=edges,
          num_nodes=4,
          num_edge_types=3,
          forward_edge_type_indices=forward_edge_type_indices,
          reverse_edge_type_indices=reverse_edge_type_indices,
          embedding_dim=EDGE_EMBEDDING_DIM)
      nonzeros = jnp.any(embeddings != 0, axis=-1).astype(jnp.float32)

    expected = np.zeros([4, 4], np.float32)
    expected[2, 0] = 1
    expected[0, 1] = 1
    expected[1, 2] = 1
    expected[0, 2] = 1
    expected[1, 0] = 1
    expected[2, 1] = 1
    np.testing.assert_allclose(nonzeros, expected)


if __name__ == "__main__":
  absltest.main()

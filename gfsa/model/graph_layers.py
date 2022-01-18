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
"""Flax module for various layers on graphs.

To allow shared implementations and compositionality, the separate components
of the models are split into pieces, each of which has a defined signature.
"""

from typing import Optional, Sequence, Tuple

import flax
import gin
import jax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

from gfsa import jax_util
from gfsa import sparse_operator
from gfsa.model import model_util

# pylint: disable=unexpected-keyword-arg

###################
# Node embeddings #
###################


class NodeTypeNodeEmbedding(flax.nn.Module):
  """Initial embedding of nodes using their types."""

  @gin.configurable("NodeTypeNodeEmbedding")
  def apply(self, node_types, num_node_types,
            embedding_dim):
    """Compute initial node embeddings.

    Args:
      node_types: <int32[num_nodes]> giving node type indices of each node.
      num_node_types: Total number of node types.
      embedding_dim: Dimensionality of the embedding space.

    Returns:
      <float32[num_nodes, embedding_dim]> embedding array.
    """
    node_type_embeddings = self.param(
        "node_type_embeddings",
        shape=(num_node_types, embedding_dim),
        initializer=initializers.variance_scaling(1.0, "fan_out",
                                                  "truncated_normal"))
    return node_type_embeddings[node_types]


def positional_node_embedding(
    num_nodes,
    embedding_dim,
    period_scale,
    tie_in_with = None):
  """Positional embedding of nodes.

  Note that for program graphs this will be a pre-order traversal of the AST,
  although we do not check that here.

  Args:
    num_nodes: How many nodes to compute embeddings for.
    embedding_dim: Size of the embedding layer. Must be even.
    period_scale: Scale of the largest period, measured in nodes.
    tie_in_with: Arbitrary optional NDArray; if this is a traced value, then we
      delay computation of the embeddings until running the XLA computation.
      This can reduce memory usage slightly.

  Returns:
      <float32[num_nodes, embedding_dim]> embedding array.
  """
  if embedding_dim % 2 != 0:
    raise ValueError("Positional embedding requires embedding_dim to be even.")
  arange_nodes = jnp.arange(num_nodes)
  arange_embedding = jnp.arange(embedding_dim // 2)
  if tie_in_with is not None:
    arange_nodes = jax.lax.tie_in(tie_in_with, arange_nodes)
    arange_embedding = jax.lax.tie_in(tie_in_with, arange_embedding)

  pos, offset = jnp.meshgrid(arange_nodes, arange_embedding, indexing="ij")
  pos = pos.astype(jnp.float32)
  offset = offset.astype(jnp.float32)
  angles = pos / jnp.power(period_scale, 2 * offset / embedding_dim)
  sin_embeddings = jnp.sin(angles)
  cos_embeddings = jnp.cos(angles)
  # Concatenate
  embeddings = jnp.reshape(
      jnp.stack([sin_embeddings, cos_embeddings], -1),
      [num_nodes, embedding_dim])
  return embeddings


class PositionalAndTypeNodeEmbedding(flax.nn.Module):
  """Initial embedding of nodes using their type and position."""

  @gin.configurable("PositionalAndTypeNodeEmbedding")
  def apply(self,
            node_types,
            num_node_types,
            embedding_dim,
            period_scale = gin.REQUIRED):
    """Compute initial node embeddings.

    Args:
      node_types: <int32[num_nodes]> giving node type indices of each node.
      num_node_types: Total number of node types.
      embedding_dim: Dimensionality of the embedding space.
      period_scale: Scale of the largest period, measured in nodes.

    Returns:
      <float32[num_nodes, embedding_dim]> embedding array.
    """
    node_type = NodeTypeNodeEmbedding(node_types, num_node_types, embedding_dim)
    position = positional_node_embedding(
        node_types.shape[0], embedding_dim, period_scale, tie_in_with=node_type)
    return node_type + position


class TokenOperatorNodeEmbedding(flax.nn.Module):
  """Embeds nodes using a token operator."""

  @gin.configurable("TokenOperatorNodeEmbedding")
  def apply(self,
            operator,
            vocab_size,
            num_nodes,
            embedding_dim,
            bottleneck_dim = None):
    """Compute token node embeddings.

    Args:
      operator: Operator from tokens to nodes.
      vocab_size: How many tokens there are in the vocabulary.
      num_nodes: How many nodes there are in the graph.
      embedding_dim: Dimensionality of the embedding space.
      bottleneck_dim: Optional initial dimension of the embedding space, which
        will be projected out.

    Returns:
      <float32[num_nodes, embedding_dim]> embedding array.
    """
    param_dim = (
        bottleneck_dim if bottleneck_dim is not None else embedding_dim)
    token_embeddings = self.param(
        "token_embeddings",
        shape=(vocab_size, param_dim),
        initializer=initializers.variance_scaling(1.0, "fan_out",
                                                  "truncated_normal"))

    node_token_embeddings = operator.apply_add(
        token_embeddings,
        jnp.zeros((num_nodes, param_dim)),
        in_dims=(0,),
        out_dims=(0,))

    if bottleneck_dim is not None:
      node_token_embeddings = flax.nn.Dense(
          node_token_embeddings, features=embedding_dim, bias=False)

    return node_token_embeddings


###################
# Edge embeddings #
###################


def _forward_and_reverse_subsets(
    num_edge_types, forward_edge_type_indices,
    reverse_edge_type_indices
):
  """Helper function to extract a subset of edges.

  Args:
    num_edge_types: How many total edge types there are.
    forward_edge_type_indices: Indices of the edge types to embed in the forward
      direction.
    reverse_edge_type_indices: Indices of the edge types to embed in the reverse
      direction.

  Returns:
    Tuple (forward_index_map, forward_values, reverse_index_map, reverse_values)
    where the index_map arrays are <int32[num_edge_types]> and specify the
    directed-subset index for each original forward edge type index (i.e. if
    forward_edge_type_indices = [3, 7] and reverse_edge_type_indices =
    [7, 31, 40]  then forward_index_map[7] = 1 and reverse_index_map[7] = 2),
    and the value arrays are 1 for edges that belong to the subset and 0 for the
    rest.
  """
  forward_index_map = np.zeros([num_edge_types], dtype=np.int32)
  forward_values = np.zeros([num_edge_types], dtype=np.float32)
  reverse_index_map = np.zeros([num_edge_types], dtype=np.int32)
  reverse_values = np.zeros([num_edge_types], dtype=np.float32)
  i = 0
  for edge_type_index in forward_edge_type_indices:
    forward_index_map[edge_type_index] = i
    forward_values[edge_type_index] = 1.0
    i += 1
  for edge_type_index in reverse_edge_type_indices:
    reverse_index_map[edge_type_index] = i
    reverse_values[edge_type_index] = 1.0
    i += 1

  return jax.tree_map(
      jnp.array,
      (forward_index_map, forward_values, reverse_index_map, reverse_values))


def binary_index_edge_embeddings(
    edges, num_nodes,
    num_edge_types, forward_edge_type_indices,
    reverse_edge_type_indices):
  """Computes multi-hot binary edge embeddings.

  Args:
    edges: Edges, represented as a sparse operator from a vector indexed by edge
      type to an adjacency matrix.
    num_nodes: Number of nodes in the graph.
    num_edge_types: How many total edge types there are.
    forward_edge_type_indices: Indices of the edge types to embed in the forward
      direction.
    reverse_edge_type_indices: Indices of the edge types to embed in the reverse
      direction.

  Returns:
    <float32[num_nodes, num_nodes, embedding_dim]> embedding array, where
    embedding_dim = len(forward_edge_type_indices)
                  + len(reverse_edge_type_indices)
  """
  embedding_dim = len(forward_edge_type_indices) + len(
      reverse_edge_type_indices)

  # Map from old edge types to new ones.
  (forward_index_map, forward_values, reverse_index_map, reverse_values) = (
      _forward_and_reverse_subsets(num_edge_types, forward_edge_type_indices,
                                   reverse_edge_type_indices))

  e_in_flat = edges.input_indices.squeeze(1)

  result = jnp.zeros([num_nodes, num_nodes, embedding_dim])
  result = result.at[edges.output_indices[:, 0], edges.output_indices[:, 1],
                     forward_index_map[e_in_flat]].add(
                         edges.values * forward_values[e_in_flat])
  result = result.at[edges.output_indices[:, 1], edges.output_indices[:, 0],
                     reverse_index_map[e_in_flat]].add(
                         edges.values * reverse_values[e_in_flat])
  return result


class LearnableEdgeEmbeddings(flax.nn.Module):
  """An edge embedding with learnable continuous outputs."""

  @gin.configurable("LearnableEdgeEmbeddings")
  def apply(self,
            edges,
            num_nodes,
            num_edge_types,
            forward_edge_type_indices,
            reverse_edge_type_indices,
            embedding_dim = gin.REQUIRED):
    """Compute multi-hot binary edge embeddings.

    Args:
      edges: Edges, represented as a sparse operator from a vector indexed by
        edge type to an adjacency matrix.
      num_nodes: Number of nodes in the graph.
      num_edge_types: How many total edge types there are.
      forward_edge_type_indices: Indices of the edge types to embed in the
        forward direction.
      reverse_edge_type_indices: Indices of the edge types to embed in the
        reverse direction.
      embedding_dim: Dimension of the learned embedding.

    Returns:
      <float32[num_nodes, num_nodes, embedding_dim]> embedding array
    """
    total_edge_count = (
        len(forward_edge_type_indices) + len(reverse_edge_type_indices))
    edge_type_embeddings = self.param(
        "edge_type_embeddings",
        shape=(total_edge_count, embedding_dim),
        initializer=initializers.variance_scaling(1.0, "fan_out",
                                                  "truncated_normal"))

    # Build new operators that include only our desired edge types by mapping
    # the `num_edge_types` to `total_edge_count`.
    (forward_index_map, forward_values, reverse_index_map, reverse_values) = (
        _forward_and_reverse_subsets(num_edge_types, forward_edge_type_indices,
                                     reverse_edge_type_indices))

    e_in_flat = edges.input_indices.squeeze(1)

    forward_operator = sparse_operator.SparseCoordOperator(
        input_indices=forward_index_map[edges.input_indices],
        output_indices=edges.output_indices,
        values=edges.values * forward_values[e_in_flat])

    reverse_operator = sparse_operator.SparseCoordOperator(
        input_indices=reverse_index_map[edges.input_indices],
        output_indices=edges.output_indices,
        values=edges.values * reverse_values[e_in_flat])

    # Apply our adjusted operators, gathering from our extended embeddings
    # array.
    result = jnp.zeros([embedding_dim, num_nodes, num_nodes])
    result = forward_operator.apply_add(
        in_array=edge_type_embeddings,
        out_array=result,
        in_dims=[0],
        out_dims=[1, 2])
    result = reverse_operator.apply_add(
        in_array=edge_type_embeddings,
        out_array=result,
        in_dims=[0],
        out_dims=[2, 1])

    # Force it to actually be materialized as
    # [(batch,) embedding_dim, num_nodes, num_nodes] to reduce downstream
    # effects of the bad padding required by the above.
    result = jax_util.force_physical_layout(result)

    return result.transpose((1, 2, 0))


def edge_mask(edges, num_nodes,
              num_edge_types, forward_edge_type_indices,
              reverse_edge_type_indices):
  """Compute a masked adjacency matrix that has 1s wherever there are edges.

  Args:
    edges: Edges, represented as a sparse operator from a vector indexed by edge
      type to an adjacency matrix.
    num_nodes: Number of nodes in the graph.
    num_edge_types: How many total edge types there are.
    forward_edge_type_indices: Indices of the edge types to embed in the forward
      direction.
    reverse_edge_type_indices: Indices of the edge types to embed in the reverse
      direction.

  Returns:
    <float32[num_nodes, num_nodes]> mask array.
  """
  (_, forward_values, _, reverse_values) = (
      _forward_and_reverse_subsets(num_edge_types, forward_edge_type_indices,
                                   reverse_edge_type_indices))

  e_in_flat = edges.input_indices.squeeze(1)

  result = jnp.zeros([num_nodes, num_nodes])
  result = result.at[edges.output_indices[:, 0],
                     edges.output_indices[:, 1]].add(edges.values *
                                                     forward_values[e_in_flat])
  result = result.at[edges.output_indices[:, 1],
                     edges.output_indices[:, 0]].add(edges.values *
                                                     reverse_values[e_in_flat])
  return jnp.minimum(result, 1.0)


########################
# Information exchange #
########################


class LinearMessagePassing(flax.nn.Module):
  """Learnable message-passing layer using linear message aggregation.

  In particular, computes the message to send from a given node across a given
  edge as (in expanded einsum notation)

    message[m] = edge_embed[e] * src_node_state[i] * params[e, i, m].

  When expressed for all nodes at once:

    message[dst, m] = edge_embed[src, dst, e] * node_state[src, i]
                                              * params[e, i, m]

  When edge_embed[:, :, e] is the adjacency matrix for type e, this becomes
  a standard GGNN message-passing step. When it is a continuous embedding, this
  becomes a MPNN with a linear edge network.
  """

  @gin.configurable("LinearMessagePassing")
  def apply(self,
            edge_embeddings,
            node_embeddings,
            message_dim = gin.REQUIRED,
            scale_by_num_nodes = True,
            scale_by_edge_embedding = False,
            use_efficient_conv = True,
            just_use_xavier = False,
            with_bias = False):
    """Apply the linear message passing layer.

    Args:
      edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]> dense
        edge embedding matrix, where zeros indicate no edge.
      node_embeddings: <float32[num_nodes, node_embedding_dim]> node embedding
        matrix.
      message_dim: Dimension of the desired messages.
      scale_by_num_nodes: Whether to scale down the message tensor
        initialization by sqrt(num_nodes), to correct for some nodes getting
        many messages.
      scale_by_edge_embedding: Whether to scale down the message tensor
        initialization by sqrt(edge_embedding_dim), to correct for edge
        embeddings having high magnitude (i.e. with learned edge embeddings).
      use_efficient_conv: Whether to directly lower the einsum into an XLA
        convolution, to ensure it is memory efficient.
      just_use_xavier: Whether to do standard Xavier initialization instead of
        scaling down the parameter based on above scaling factors.
      with_bias: Whether to add a bias term (which depends on edge type but not
        source content).

    Returns:
      <float32[num_nodes, message_dim]> containing the sum of received messages.
    """
    edge_embedding_dim = edge_embeddings.shape[-1]
    node_embedding_dim = node_embeddings.shape[-1]
    num_nodes = node_embeddings.shape[0]
    if just_use_xavier:
      message_passing_tensor = self.param(
          "message_passing_tensor",
          shape=(edge_embedding_dim, node_embedding_dim, message_dim),
          initializer=initializers.xavier_normal())
      if with_bias:
        edge_bias_tensor = self.param(
            "edge_bias_tensor",
            shape=(edge_embedding_dim, message_dim),
            initializer=initializers.xavier_normal())
    else:
      variance_correction = node_embedding_dim
      if scale_by_num_nodes:
        variance_correction *= num_nodes
      if scale_by_edge_embedding:
        variance_correction *= edge_embedding_dim
      message_passing_tensor = self.param(
          "message_passing_tensor",
          shape=(edge_embedding_dim, node_embedding_dim, message_dim),
          initializer=initializers.normal()) / np.sqrt(variance_correction)
      if with_bias:
        edge_bias_tensor = self.param(
            "edge_bias_tensor",
            shape=(edge_embedding_dim, message_dim),
            initializer=initializers.normal()) / np.sqrt(
                variance_correction / node_embedding_dim)

    if use_efficient_conv:
      # Carefully chose conv axes so that the node axis is the feature axis.
      # First, sources compute the messages to send.
      # eim, si->sem
      # 0CN,0OI->C0N
      messages = jax.lax.conv_general_dilated(
          message_passing_tensor,
          node_embeddings[None],
          window_strides=(1,),
          padding="VALID",
          dimension_numbers=("0CN", "0OI", "C0N"))
      # Next, messages are sent across edges.
      # sem,sde-> dm
      # C0N,IO0->0CN
      received = jax.lax.conv_general_dilated(
          messages,
          edge_embeddings,
          window_strides=(edge_embedding_dim,),
          padding="VALID",
          dimension_numbers=("C0N", "IO0", "0CN")).squeeze(0)
    else:
      # Let JAX handle the einsum implementation.
      received = jnp.einsum("sde,si,eim->dm", edge_embeddings, node_embeddings,
                            message_passing_tensor)

    if with_bias:
      received = (
          received +
          jnp.einsum("sde,em->dm", edge_embeddings, edge_bias_tensor))

    return received


class NodeSelfAttention(flax.nn.Module):
  """Node self-attention layer.

  Implements (optionally masked) multi-head self attention between nodes.
  Queries depend on the "receiving" node, and keys and values depend on both
  the "sending" node and the edge embedding.

  Specific implementation notes:
    - Depending on `like_great`, we either follow  Wang et al. (2020) or
      Hellendoorn et al. (2020) in terms of how attention is computed. With
      like_great=False, the edges are incorporated as additive learned
      vector-valued components for the key and value. With like_great=True,
      the edges add a bias proportional to the sum of the keys (equivalent
      to adding a vector containing a constant bias to the query).
    - Edge embeddings are shared across heads, as in both of those references.

  Hellendoorn, Maniatis, Singh, Sutton (2020): "Global relational models of
    source code".
  Wang, Shin, Liu, Polozov, Richardson (2020): "RAT-SQL: Relation-aware schema
    encoding and linking for text-to-SQL parsers".
  """

  @gin.configurable("NodeSelfAttention")
  def apply(self,
            edge_embeddings,
            node_embeddings,
            out_dim,
            heads = gin.REQUIRED,
            query_key_dim = None,
            value_dim = None,
            mask = None,
            like_great = False):
    """Apply the attention layer.

    Args:
      edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]> dense
        edge embedding matrix, where zeros indicate no edge.
      node_embeddings: <float32[num_nodes, node_embedding_dim]> node embedding
        matrix.
      out_dim: Output dimension.
      heads: How many attention heads to use.
      query_key_dim: Dimension of the queries and keys. If not provided, assumed
        to be node_embedding_dim / heads.
      value_dim: Dimension of the queries and keys. If not provided, assumed to
        be the same as query_key_dim.
      mask: <float32[num_nodes, num_nodes]> mask determining which other nodes a
        given node is allowed to attend to.
      like_great: Whether to use GREAT-style key-bias attention instead of more
        powerful vector attention (as in RAT).

    Returns:
      <float32[num_nodes, out_dim]> softmax-weighted value sums over nodes in
      the graph.
    """

    def inner_apply(edge_embeddings, node_embeddings):
      # Einsum letters:
      # n: querying node, attends to others.
      # m: queried node, is attended to.
      # h: attention head
      # d: node embedding dimension
      # e: edge embedding dimension
      # q: query/key dimension
      # v: value dimension
      edge_embedding_dim = edge_embeddings.shape[-1]
      node_embedding_dim = node_embeddings.shape[-1]

      nonlocal query_key_dim, value_dim

      if query_key_dim is None:
        if node_embedding_dim % heads != 0:
          raise ValueError(
              "No query_key_dim provided, but node embedding dim "
              f"({node_embedding_dim}) was not divisible by head count "
              f"({heads})")
        query_key_dim = node_embedding_dim // heads

      if value_dim is None:
        value_dim = query_key_dim

      # Compute queries.
      query_tensor = self.param(
          "query_tensor",
          shape=(heads, node_embedding_dim, query_key_dim),
          initializer=initializers.xavier_normal())
      query = jnp.einsum("nd,hdq->hnq", node_embeddings, query_tensor)

      # Dot-product the queries with the node and edge keys.
      node_key_tensor = self.param(
          "node_key_tensor",
          shape=(heads, node_embedding_dim, query_key_dim),
          initializer=initializers.xavier_normal())
      dot_product_logits = jnp.einsum("md,hdq,hnq->hnm", node_embeddings,
                                      node_key_tensor, query)

      if like_great:
        # Edges contribute based on key sums, as in Hellendoorn et al.
        # edge_biases: <float32[num_nodes, num_nodes]>, computed as `w^T e + b`
        edge_biases = flax.nn.Dense(edge_embeddings, features=1).squeeze(-1)
        # Einsum sums keys over `q` dim (equivalent to broadcasting out biases).
        edge_logits = jnp.einsum("md,hdq,nm->hnm", node_embeddings,
                                 node_key_tensor, edge_biases)
      else:
        # Queries attend to edge keys, as in Wang et al.
        edge_key_tensor = self.param(
            "edge_key_tensor",
            shape=(edge_embedding_dim, query_key_dim),
            initializer=initializers.xavier_normal())
        edge_logits = jnp.einsum("nme,eq,hnq->hnm", edge_embeddings,
                                 edge_key_tensor, query)

      # Combine, normalize, and possibly mask.
      attention_logits = ((dot_product_logits + edge_logits) /
                          jnp.sqrt(query_key_dim))
      if mask is not None:
        attention_logits = attention_logits + jnp.log(mask)[None, :, :]

      attention_weights = jax.nn.softmax(attention_logits, axis=2)

      # Wrap attention weights with a Flax module so we can extract intermediate
      # outputs.
      attention_weights = jax_util.flax_tag(
          attention_weights, name="attention_weights")

      # Compute values.
      node_value_tensor = self.param(
          "node_value_tensor",
          shape=(heads, node_embedding_dim, value_dim),
          initializer=initializers.xavier_normal())
      attention_node_value = jnp.einsum(
          "hnm,hmv->hnv", attention_weights,
          jnp.einsum("md,hdv->hmv", node_embeddings, node_value_tensor))

      if like_great:
        # Only nodes contribute to values, as in Hellendoorn et al.
        attention_value = attention_node_value
      else:
        # Edges also contribute to values, as in Wang et al.
        edge_value_tensor = self.param(
            "edge_value_tensor",
            shape=(edge_embedding_dim, value_dim),
            initializer=initializers.xavier_normal())
        attention_edge_value = jnp.einsum("hnm,nme,ev->hnv", attention_weights,
                                          edge_embeddings, edge_value_tensor)

        attention_value = attention_node_value + attention_edge_value

      # Project them back.
      output_tensor = self.param(
          "output_tensor",
          shape=(heads, value_dim, out_dim),
          initializer=initializers.xavier_normal())
      output = jnp.einsum("hnv,hvo->no", attention_value, output_tensor)
      return output

    # Make sure we don't keep any of the intermediate hidden matrices around
    # any longer than we have to.
    if not self.is_initializing():
      inner_apply = jax.checkpoint(inner_apply)

    return inner_apply(edge_embeddings, node_embeddings)


class NRIEdgeLayer(flax.nn.Module):
  """NRI-style computation of vertex->edge hiddens.

  These can be aggregated to do NRI-style message passing, or used as an output
  head or edge embedding modification. Uses a MLP for each pair of nodes, with
  MLP parameters defined separately for each edge type.

  If edge_embeddings is multi-hot, this corresponds directly to the description
  in Kipf et al. 2018. If edge_embeddings is a learned vector, it is more like
  a combination of NRI and R-GCN's basis decomposition (Schlichtkrull et al.):
  the elements of the edge embedding vector become weights for the contributions
  of each basis-vector MLP.

  If allow_non_adjacent=True, we also use an MLP for every pair of nodes, even
  if there is no edge between them. In this case, edge_embeddings may be None
  to only compute outputs based on the two hiddens.
  """

  @gin.configurable("NRIEdgeLayer")
  def apply(
      self,
      edge_embeddings,
      node_embeddings,
      mask,
      mlp_vtoe_dims,
      allow_non_adjacent,
      message_passing,
  ):
    """Apply the NRIEdgeLayer.

    Args:
      edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]> dense
        edge embedding matrix, where zeros indicate no edge.
      node_embeddings: <float32[num_nodes, node_embedding_dim]> node embedding
        matrix.
      mask: <float32[num_nodes, num_nodes]> mask determining which other nodes a
        given node is allowed to send messages to.
      mlp_vtoe_dims: List of hidden and output dimension sizes; determines the
        depth and width of the MLP.
      allow_non_adjacent: Compute messages even for non-adjacent nodes.
      message_passing: If True, accumulate messages to destination nodes.

    Returns:
      If message_passing=True: <float32[num_nodes, out_dim]> messages.
      If message_passing=False: <float32[num_nodes, num_nodes, out_dim]> edge
                                activations.
    """
    if not mlp_vtoe_dims:
      raise ValueError("Must have a nonempty sequence for mlp_vtoe_dims")

    def inner_apply(edge_embeddings, node_embeddings):
      first_layer_dim = mlp_vtoe_dims[0]
      additional_layer_dims = mlp_vtoe_dims[1:]

      if allow_non_adjacent and edge_embeddings is not None:
        num_separate_mlps = 1 + edge_embeddings.shape[-1]
      elif allow_non_adjacent:
        num_separate_mlps = 1
      elif edge_embeddings is not None:
        num_separate_mlps = edge_embeddings.shape[-1]
      else:
        raise ValueError("Either allow_non_adjacent should be True, or "
                         "edge_embeddings should be provided")

      node_embedding_dim = node_embeddings.shape[-1]

      # First layer: process each node embedding.
      weight_from_source = self.param(
          "l0_weight_from_source",
          shape=(num_separate_mlps, node_embedding_dim, first_layer_dim),
          initializer=initializers.xavier_normal())
      weight_from_dest = self.param(
          "l0_weight_from_dest",
          shape=(num_separate_mlps, node_embedding_dim, first_layer_dim),
          initializer=initializers.xavier_normal())
      bias = self.param(
          "l0_bias",
          shape=(num_separate_mlps, first_layer_dim),
          initializer=initializers.zeros)
      from_source = jnp.einsum("sx,kxy->sky", node_embeddings,
                               weight_from_source)
      from_dest = jnp.einsum("dx,kxy->dky", node_embeddings, weight_from_dest)
      activations = jax.nn.relu(from_source[:, None, :, :] +
                                from_dest[None, :, :, :] +
                                bias[None, None, :, :])

      # Additional layers: MLP for each edge type.
      for i, layer_dim in enumerate(additional_layer_dims):
        weight = self.param(
            f"l{i+1}_weight",
            shape=(num_separate_mlps, activations.shape[-1], layer_dim),
            initializer=initializers.xavier_normal())
        bias = self.param(
            f"l{i+1}_bias",
            shape=(num_separate_mlps, layer_dim),
            initializer=initializers.zeros)
        activations = jax.nn.relu(
            jnp.einsum("sdkx,kxy->sdky", activations, weight) +
            bias[None, None, :, :])

      # Sum over edge types and possibly over source nodes.
      if edge_embeddings is None:
        result = activations.squeeze(axis=2)
        if mask is not None:
          result = jnp.where(mask[:, :, None], result, jnp.zeros_like(result))
        if message_passing:
          result = jnp.sum(result, axis=0)
      else:
        if allow_non_adjacent:
          if mask is None:
            pairwise = jnp.ones(edge_embeddings.shape[:2] + (1,))
          else:
            pairwise = mask
          mlp_weights = jnp.concatenate(
              [edge_embeddings,
               pairwise.astype("float")[:, :, None]], -1)
        else:
          mlp_weights = edge_embeddings

        if message_passing:
          result = jnp.einsum("sdky,sdk->dy", activations, mlp_weights)
        else:
          result = jnp.einsum("sdky,sdk->sdy", activations, mlp_weights)

      return result

    # Make sure we don't keep any of the intermediate hidden matrices around
    # any longer than we have to.
    if not self.is_initializing():
      inner_apply = jax.checkpoint(inner_apply)

    return inner_apply(edge_embeddings, node_embeddings)


################
# State update #
################


@flax.nn.module
def residual_layer_norm_update(node_states,
                               messages):
  """Update node states using a residual step and layer norm.

  This is based on the update step in a normal transformer model. We assume the
  node states and messages are the same size.

  Args:
    node_states: <float32[num_nodes, node_embedding_dim]>
    messages: <float32[num_nodes, node_embedding_dim]>

  Returns:
    <float32[num_nodes, node_embedding_dim]> new state.
  """
  combined = node_states + messages
  return model_util.ScaleAndShift(jax.nn.normalize(combined, axis=-1))


@flax.nn.module
def gated_recurrent_update(node_states,
                           messages):
  """Update node states using the GRU equations.

  Args:
    node_states: <float32[num_nodes, state_dim]> previous states.
    messages: <float32[num_nodes, message_dim]> received messages.

  Returns:
    <float32[num_nodes, state_dim]> new states.
  """
  # Simply wrap flax.nn.recurrent.GRUCell to have the desired interface.
  h, h2 = flax.nn.recurrent.GRUCell(node_states, messages)
  assert h is h2
  return h


####################
# Pairwise readout #
####################


class BilinearPairwiseReadout(flax.nn.Module):
  """Extract pairwise information using a shifted bilinear operator.

  In other words, computes

    O[i, j] = N[i, :]^T A N[i, :] + b

  for learned A and b. Note that A is intentionally not restricted to be
  a symmetric matrix, so that we can learn asymmetric outputs (i.e. have
  O[i, j] != O[j, i]).
  """

  def apply(self,
            node_embeddings,
            readout_dim = None):
    """Appy the binary readout layer.

    Args:
      node_embeddings: <float32[num_nodes, node_embedding_dim]> array.
      readout_dim: Size of the readout dimension. If None, no readout dimension
        is added.

    Returns:
      <float32[num_nodes, num_nodes]> array of float scores, which can be
        interpreted as logits.
    """
    if readout_dim is None:
      readout_dim_or_one = 1
    else:
      readout_dim_or_one = readout_dim

    node_embedding_dim = node_embeddings.shape[-1]
    operator = self.param(
        "operator",
        shape=(node_embedding_dim, node_embedding_dim, readout_dim_or_one),
        initializer=initializers.normal())
    # To maintain the variance of the input, divide by sqrt(dim^2) = dim.
    # (Like in a transformer, we divide AFTER applying the tensor product. This
    # is actually very important when using adaptive optimizers. For instance
    # Adam's hyperparameters restrict how much change we can have in parameter
    # space; if the initial parameters are tiny, Adam can quickly destabilize
    # everything, but if the parameters are large and then we apply a
    # correction factor, Adam's gradient updates are ALSO scaled appropriately.)
    result = jnp.einsum("nc,md,cdr->nmr", node_embeddings, node_embeddings,
                        operator) / node_embedding_dim
    bias = self.param(
        "bias", shape=(readout_dim_or_one,), initializer=initializers.zeros)
    output = result + bias
    if readout_dim is None:
      output = output.squeeze(-1)
    return output


@flax.nn.module
def NRIReadout(  # pylint: disable=invalid-name
    node_embeddings,
    readout_dim = None):
  """Extract pairwise information using an NRI-style operation.

  Applies an MLP to pairs of nodes to produce logits.

  Args:
    node_embeddings: <float32[num_nodes, node_embedding_dim]> array.
    readout_dim: Size of the readout dimension. If None, no readout dimension is
      added.

  Returns:
    <float32[num_nodes, num_nodes]> array of float scores, which can be
      interpreted as logits.
  """
  # We assume masking will be done later if needed, after converting logits.
  edge_activations = NRIEdgeLayer(
      edge_embeddings=None,
      node_embeddings=node_embeddings,
      allow_non_adjacent=True,
      message_passing=False,
      mask=None)

  result = flax.nn.Dense(
      edge_activations, features=(1 if readout_dim is None else readout_dim))

  if readout_dim is None:
    result = result.squeeze(-1)

  return result

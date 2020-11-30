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
"""Full models and model components for the fully-supervised edge prediction task."""

from typing import Dict, List, Optional, Sequence

import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np

from gfsa import automaton_builder
from gfsa import jax_util
from gfsa import py_ast_graphs
from gfsa.datasets import graph_bundle
from gfsa.model import automaton_layer
from gfsa.model import graph_layers
from gfsa.model import model_util

# Flax adds name keyword arguments.
# pylint: disable=unexpected-keyword-arg


def variants_from_edges(example,
                        graph_metadata,
                        variant_edge_type_indices,
                        num_edge_types):
  """Convert an edge type to a variant weights matrix.

  Args:
    example: Example to run the automaton on.
    graph_metadata: Statically-known metadata about the graph size. If
      encoded_graph is padded, this should reflect the padded size, not the
      original size.
    variant_edge_type_indices: Edge types to use as variants. Assumes without
      checking that the given variants are mutually exclusive (at most one edge
      of one of these types exists between any pair of nodes).
    num_edge_types: Maximum number of edge types that exist.

  Returns:
    <float32[num_nodes, num_nodes, len(variant_edge_type_indices) + 1]> array
    to be used as variant weights.
  """
  # Assign each variant edge type a nonzero index.
  edge_type_to_variant = np.zeros([num_edge_types], dtype=np.int32)
  for var_idx, edge_idx in enumerate(variant_edge_type_indices):
    edge_type_to_variant[edge_idx] = var_idx + 1
  # Place those indices into a [num_nodes, num_nodes] array.
  num_nodes = graph_metadata.num_nodes
  variants_in_place = example.edges.apply_add(
      in_array=edge_type_to_variant,
      out_array=jnp.zeros([num_nodes, num_nodes], dtype="int32"))
  # Expand it to one-hot.
  variant_weights = jax.nn.one_hot(variants_in_place,
                                   len(variant_edge_type_indices) + 1)
  return variant_weights


def ground_truth_adjacency(
    example,
    graph_metadata,
    target_edge_type, all_edge_types):
  """Helper function to extract a ground-truth adjacency matrix.

  Intended for visualization and debugging.

  Args:
    example: Example to run the automaton on.
    graph_metadata: Statically-known metadata about the graph size. If
      encoded_graph is padded, this should reflect the padded size, not the
      original size.
    target_edge_type: Edge type for the target output.
    all_edge_types: List of possible edge types used in the example.

  Returns:
    <float32[num_nodes, num_nodes]> array, the target adjacency matrix.
  """
  return variants_from_edges(
      example=example,
      graph_metadata=graph_metadata,
      variant_edge_type_indices=[all_edge_types.index(target_edge_type)],
      num_edge_types=len(all_edge_types))[:, :, 1]


@flax.nn.module
@gin.configurable
def automaton_model(example,
                    graph_metadata,
                    edge_types_to_indices,
                    variant_edge_types = (),
                    platt_scale = False,
                    with_backtrack = True):
  """Automaton-based module for edge supervision task.

  Args:
    example: Example to run the automaton on.
    graph_metadata: Statically-known metadata about the graph size. If
      encoded_graph is padded, this should reflect the padded size, not the
      original size.
    edge_types_to_indices: Mapping from edge type names to edge type indices.
    variant_edge_types: Edge types to use as variants. Assumes without checking
      that the given variants are mutually exclusive (at most one edge of one of
      these types exists between any pair of nodes).
    platt_scale: Whether to scale and shift the logits produced by the
      automaton. This can be viewed as a form of Platt scaling applied to the
      automaton logits. If True, this allows the model's output probabilities to
      sum to more than 1, so that it can express one-to-many relations.
    with_backtrack: Whether the automaton can restart the search as an action.

  Returns:
    <float32[num_nodes, num_nodes]> matrix of binary logits for a weighted
    adjacency matrix corresponding to the predicted output edges.
  """
  if variant_edge_types:
    variant_edge_type_indices = [
        edge_types_to_indices[type_str] for type_str in variant_edge_types
    ]
    num_edge_types = len(edge_types_to_indices)
    variant_weights = variants_from_edges(example, graph_metadata,
                                          variant_edge_type_indices,
                                          num_edge_types)
  else:
    variant_weights = None

  absorbing_probs = automaton_layer.FiniteStateGraphAutomaton(
      encoded_graph=example.automaton_graph,
      variant_weights=variant_weights,
      static_metadata=graph_metadata,
      dynamic_metadata=example.graph_metadata,
      builder=automaton_builder.AutomatonBuilder(
          py_ast_graphs.SCHEMA, with_backtrack=with_backtrack),
      num_out_edges=1,
      share_states_across_edges=True).squeeze(axis=0)

  logits = model_util.safe_logit(absorbing_probs)

  if platt_scale:
    logits = model_util.ScaleAndShift(logits)

  return logits


def _layer_helper(step_module,
                  node_embeddings,
                  edge_embeddings,
                  layers,
                  share_weights = False,
                  unroll = False):
  """Helper to define a multilayer model.

  Args:
    step_module: Function that computes a single layer of the model, given node
      and edge embeddings.
    node_embeddings: <float32[num_nodes, node_embedding_dim]>
    edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]>
    layers: How many layers to use.
    share_weights: Whether we share weights between layers; this doesn't seem to
      be common but might allow longer-range connections if `mask_to_neighbors`
      is true.
    unroll: Whether to use Python loops instead of scan, for debugging. Only
      used if share_weights is true; without shared weights we always unroll.

  Returns:
    Final node embeddings <float32[num_nodes, node_embedding_dim]>.
  """
  if share_weights:
    step_module = step_module.shared(name="step")

  # Jax will trace the function once and use the parameters from that one
  # call, so we can't use scan without share_weights.
  if share_weights and not unroll:
    # Call the function once outside the loop so flax can initialize parameters
    # as a side effect. XLA dead code elimination will remove duplicate work.
    _ = step_module(
        node_embeddings=node_embeddings, edge_embeddings=edge_embeddings)

    def body_fn(v, _):
      return (step_module(node_embeddings=v,
                          edge_embeddings=edge_embeddings), None)

    result, _ = jax.lax.scan(
        f=body_fn, init=node_embeddings, xs=None, length=layers)
    return result
  else:
    for _ in range(layers):
      node_embeddings = step_module(
          node_embeddings=node_embeddings, edge_embeddings=edge_embeddings)
    return node_embeddings


@flax.nn.module
@gin.configurable
def ggnn_steps(node_embeddings,
               edge_embeddings,
               iterations = gin.REQUIRED,
               unroll = False):
  """GGNN message-passing and update steps.

  Args:
    node_embeddings: <float32[num_nodes, node_embedding_dim]>
    edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]>
    iterations: How many steps to run the GGNN. Note that parameters are shared
      between iterations.
    unroll: Whether to use Python loops instead of scan, for debugging.

  Returns:
    Final node embeddings <float32[num_nodes, node_embedding_dim]>.
  """

  @flax.nn.module
  def step(node_embeddings, edge_embeddings):
    messages = graph_layers.LinearMessagePassing(
        edge_embeddings=edge_embeddings,
        node_embeddings=node_embeddings,
        name="aggregate")
    return graph_layers.gated_recurrent_update(
        node_embeddings, messages, name="update")

  return _layer_helper(
      step,
      node_embeddings=node_embeddings,
      edge_embeddings=edge_embeddings,
      layers=iterations,
      share_weights=True,
      unroll=unroll)


@flax.nn.module
@gin.configurable
def transformer_steps(node_embeddings,
                      edge_embeddings,
                      neighbor_mask,
                      num_real_nodes_per_graph,
                      layers = gin.REQUIRED,
                      mask_to_neighbors = False,
                      share_weights = False,
                      unroll = False):
  """Transformer message-passing and update steps.

  If mask_to_neighbors = False, this is similar to the RAT model in
  "RAT-SQL: Relation-aware schema encoding and linking for text-to-SQL parsers"
  or the GREAT model in "Global relational models of source code". See
  graph_layers.py for details.

  If mask_to_neighbors = True, this is similar to a Graph Attention Network.

  Args:
    node_embeddings: <float32[num_nodes, node_embedding_dim]>
    edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]>
    neighbor_mask: <float32[num_nodes, num_nodes]> mask to use if
      `mask_to_neighbors` is True.
    num_real_nodes_per_graph: Number of real nodes per graph; used to do valid
      masking.
    layers: How many layers to use.
    mask_to_neighbors: Whether attention is masked to only neighbors.
    share_weights: Whether we share weights between layers; this doesn't seem to
      be common but might allow longer-range connections if `mask_to_neighbors`
      is true.
    unroll: Whether to use Python loops instead of scan, for debugging. Only
      used if share_weights is true; without shared weights we always unroll.

  Returns:
    Final node embeddings <float32[num_nodes, node_embedding_dim]>.
  """
  num_nodes = node_embeddings.shape[0]
  if mask_to_neighbors:
    mask = neighbor_mask
  else:
    mask = jnp.arange(num_nodes) < num_real_nodes_per_graph
    mask = jnp.broadcast_to(mask[None, :],
                            (num_nodes, num_nodes)).astype(jnp.float32)

  @flax.nn.module
  def step(node_embeddings, edge_embeddings):
    attn_out = graph_layers.NodeSelfAttention(
        edge_embeddings=edge_embeddings,
        node_embeddings=node_embeddings,
        mask=mask,
        out_dim=node_embeddings.shape[-1],
        name="attend")
    node_embeddings = graph_layers.residual_layer_norm_update(
        node_embeddings, attn_out, name="attend_ln")

    fc_out = jax.nn.relu(
        flax.nn.Dense(
            node_embeddings,
            features=node_embeddings.shape[-1],
            name="fc_dense"))
    node_embeddings = graph_layers.residual_layer_norm_update(
        node_embeddings, fc_out, name="fc_ln")

    return node_embeddings

  return _layer_helper(
      step,
      node_embeddings=node_embeddings,
      edge_embeddings=edge_embeddings,
      layers=layers,
      share_weights=share_weights,
      unroll=unroll)


@flax.nn.module
@gin.configurable
def nri_steps(node_embeddings,
              edge_embeddings,
              num_real_nodes_per_graph,
              layers = gin.REQUIRED,
              mlp_etov_dims = gin.REQUIRED,
              with_residual_layer_norm = False,
              share_weights = False,
              unroll = False):
  """NRI-style message-passing and update steps.

  This corresponds roughtly to the model described in "Neural Relational
  Inference for Interacting Systems" by Kipf et al, except that it allows
  stacking arbitrary numbers of layers.

  Args:
    node_embeddings: <float32[num_nodes, node_embedding_dim]>
    edge_embeddings: <float32[num_nodes, num_nodes, edge_embedding_dim]>
    num_real_nodes_per_graph: Number of real nodes per graph; used to do valid
      masking.
    layers: How many overall NRI layers/blocks to use.
    mlp_etov_dims: Sizes for each of the hidden layers of the receiving MLP. We
      assume that the final layer is of size node_embedding_dim, so an empty
      sequence will result in a one-layer transformation.
    with_residual_layer_norm: Whether to combine old and new embeddings using
      residual connections; this makes the model a hybrid between NRI and a
      transformer.
    share_weights: Whether we share weights between layers; this doesn't seem to
      be common but might allow longer-range connections if `mask_to_neighbors`
      is true.
    unroll: Whether to use Python loops instead of scan, for debugging. Only
      used if share_weights is true; without shared weights we always unroll.

  Returns:
    Final node embeddings <float32[num_nodes, node_embedding_dim]>.
  """
  num_nodes = node_embeddings.shape[0]
  mask = jnp.arange(num_nodes) < num_real_nodes_per_graph
  mask = mask[None, :] & mask[:, None]

  @flax.nn.module
  def step(node_embeddings, edge_embeddings):
    nri_activations = graph_layers.NRIEdgeLayer(
        edge_embeddings=edge_embeddings,
        node_embeddings=node_embeddings,
        mask=mask,
        message_passing=True,
        name="nri_message_passing")

    for i, dim in enumerate([*mlp_etov_dims, node_embeddings.shape[-1]]):
      nri_activations = flax.nn.Dense(
          nri_activations, features=dim, name=f"fc{i}")
      nri_activations = jax.nn.relu(nri_activations)

    if with_residual_layer_norm:
      node_embeddings = graph_layers.residual_layer_norm_update(
          node_embeddings, nri_activations, name="layer_norm_update")
    else:
      node_embeddings = nri_activations

    return node_embeddings

  return _layer_helper(
      step,
      node_embeddings=node_embeddings,
      edge_embeddings=edge_embeddings,
      layers=layers,
      share_weights=share_weights,
      unroll=unroll)


class BaselineModel(flax.nn.Module):
  """Baseline model for edge-supervision tasks."""

  @gin.configurable("BaselineModel")
  def apply(
      self,
      example,
      graph_metadata,
      edge_types_to_indices,
      forward_edge_types,
      reverse_edge_types,
      use_position_embeddings = gin.REQUIRED,
      learn_edge_embeddings = gin.REQUIRED,
      model_type = gin.REQUIRED,
      nodewise = False,
      nodewise_loop_chunk_size = None,
  ):
    """Single-forward-pass baseline model for edge-supervision task.

    This model propagates information through the graph, then does a pairwise
    bilinear readout to determine which edges to add.

    Args:
      example: Example to run the automaton on.
      graph_metadata: Statically-known metadata about the graph size. If
        encoded_graph is padded, this should reflect the padded size, not the
        original size.
      edge_types_to_indices: Mapping from edge type names to edge type indices.
      forward_edge_types: Edge types to use in the forward direction. As a list
        of lists to allow configuring groups of edges in config files; this will
        be flattened before use.
      reverse_edge_types: Edge types to use in the reverse direction. Note that
        reversed edge types are given a separate embedding from forward edge
        types; undirected edges should be represented by adding two edges in
        opposite directions and then only using `forward_edge_types`. Also a
        list of lists, as above.
      use_position_embeddings: Whether to add position embeddings to node
        embeddings.
      learn_edge_embeddings: Whether to learn an edge embedding for each edge
        type (instead of using a one-hot embedding).
      model_type: One of {"ggnn", "transformer", "nri_encoder"}
      nodewise: Whether to have separate sets of node embeddings for each
        possible start node.
      nodewise_loop_chunk_size: Optional integer, which must divide the number
        of nodes. Splits the nodes into chunks of this size, and runs the model
        on each of those splits in a loop; this recudes the memory usage. Only
        used when nodewise=True.

    Returns:
      <float32[num_nodes, num_nodes]> matrix of binary logits for a weighted
      adjacency matrix corresponding to the predicted output edges.
    """
    # Node types come directly from the schema (for parity with the automaton).
    num_node_types = len(py_ast_graphs.BUILDER.node_types)
    # Edge types are potentially task-specific.
    num_edge_types = len(edge_types_to_indices)
    # pylint: disable=g-complex-comprehension
    forward_edge_type_indices = [
        edge_types_to_indices[type_str]
        for group in forward_edge_types
        for type_str in group
    ]
    reverse_edge_type_indices = [
        edge_types_to_indices[type_str]
        for group in reverse_edge_types
        for type_str in group
    ]
    # pylint: enable=g-complex-comprehension

    # Embed the nodes.
    if use_position_embeddings:
      node_embeddings = graph_layers.PositionalAndTypeNodeEmbedding(
          node_types=example.node_types, num_node_types=num_node_types)
    else:
      node_embeddings = graph_layers.NodeTypeNodeEmbedding(
          node_types=example.node_types, num_node_types=num_node_types)

    # Embed the edges.
    if learn_edge_embeddings:
      edge_embeddings = graph_layers.LearnableEdgeEmbeddings(
          edges=example.edges,
          num_nodes=graph_metadata.num_nodes,
          num_edge_types=num_edge_types,
          forward_edge_type_indices=forward_edge_type_indices,
          reverse_edge_type_indices=reverse_edge_type_indices)
    else:
      edge_embeddings = graph_layers.binary_index_edge_embeddings(
          edges=example.edges,
          num_nodes=graph_metadata.num_nodes,
          num_edge_types=num_edge_types,
          forward_edge_type_indices=forward_edge_type_indices,
          reverse_edge_type_indices=reverse_edge_type_indices)

    def run_steps(node_embeddings):
      """Runs propagation and updates."""
      if model_type == "ggnn":
        final_embeddings = ggnn_steps(node_embeddings, edge_embeddings)
      elif model_type == "transformer":
        neighbor_mask = graph_layers.edge_mask(
            edges=example.edges,
            num_nodes=graph_metadata.num_nodes,
            num_edge_types=num_edge_types,
            forward_edge_type_indices=forward_edge_type_indices,
            reverse_edge_type_indices=reverse_edge_type_indices)
        # Allow nodes to attend to themselves
        neighbor_mask = jnp.maximum(neighbor_mask,
                                    jnp.eye(graph_metadata.num_nodes))
        final_embeddings = transformer_steps(
            node_embeddings,
            edge_embeddings,
            neighbor_mask,
            num_real_nodes_per_graph=example.graph_metadata.num_nodes)
      elif model_type == "nri_encoder":
        final_embeddings = nri_steps(
            node_embeddings,
            edge_embeddings,
            num_real_nodes_per_graph=example.graph_metadata.num_nodes)
      return final_embeddings

    if nodewise:
      assert model_type != "nri_encoder", "Nodewise NRI model is not defined."
      # Add in a learned start node embedding, and broadcast node states out to
      # <float32[num_nodes, num_nodes, node_embedding_dim]>
      node_embedding_dim = node_embeddings.shape[-1]
      start_node_embedding = self.param(
          "start_node_embedding",
          shape=(node_embedding_dim,),
          initializer=jax.nn.initializers.normal())

      stacked_node_embeddings = jax.ops.index_add(
          jnp.broadcast_to(node_embeddings[None, :, :],
                           (graph_metadata.num_nodes, graph_metadata.num_nodes,
                            node_embedding_dim)),
          jax.ops.index[jnp.arange(graph_metadata.num_nodes),
                        jnp.arange(graph_metadata.num_nodes)],
          jnp.broadcast_to(start_node_embedding,
                           (graph_metadata.num_nodes, node_embedding_dim)),
      )

      # final_embeddings_from_each_source will be
      # [num_nodes, num_nodes, node_embedding_dim]
      if nodewise_loop_chunk_size and not self.is_initializing():
        grouped_stacked_node_embeddings = stacked_node_embeddings.reshape(
            (-1, nodewise_loop_chunk_size, graph_metadata.num_nodes,
             node_embedding_dim))
        grouped_final_embeddings_from_each_source = jax.lax.map(
            jax.vmap(run_steps), grouped_stacked_node_embeddings)
        final_embeddings_from_each_source = (
            grouped_final_embeddings_from_each_source.reshape(
                (graph_metadata.num_nodes,) +
                grouped_final_embeddings_from_each_source.shape[2:]))
      else:
        final_embeddings_from_each_source = jax.vmap(run_steps)(
            stacked_node_embeddings)

      # Extract predictions with a linear transformation.
      logits = flax.nn.Dense(
          final_embeddings_from_each_source,
          features=1,
          name="target_readout",
      ).squeeze(-1)

    elif model_type == "nri_encoder":
      # Propagate the node embeddings as-is, then use NRI to construct edges.
      final_embeddings = run_steps(node_embeddings)
      logits = graph_layers.NRIReadout(final_embeddings)

    else:
      # Propagate the node embeddings as-is, then extract edges pairwise.
      final_embeddings = run_steps(node_embeddings)
      logits = graph_layers.BilinearPairwiseReadout(final_embeddings)

    return logits

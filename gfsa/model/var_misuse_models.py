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
"""Core model for variable misuse task."""

from typing import List, Optional

import flax
import gin
import jax
import jax.numpy as jnp

from gfsa import automaton_builder
from gfsa import jax_util
from gfsa.datasets.var_misuse import example_definition
from gfsa.model import end_to_end_stack
from gfsa.model import graph_layers

# Flax adds name keyword arguments.
# pylint: disable=unexpected-keyword-arg


@flax.deprecated.nn.module
@gin.configurable
def token_graph_model_core(
    input_graph,
    static_metadata,
    encoding_info,
    node_embedding_dim = gin.REQUIRED,
    edge_embedding_dim = gin.REQUIRED,
    forward_edge_types = gin.REQUIRED,
    reverse_edge_types = gin.REQUIRED,
    components = gin.REQUIRED,
):
  """Transforms an input graph into final node embeddings, with no output head.

  Args:
    input_graph: Input graph for this example.
    static_metadata: Metadata about the padded size of this graph.
    encoding_info: How the example was encoded.
    node_embedding_dim: Dimension of node embedding space.
    edge_embedding_dim: Dimension of edge embedding space. If None, just
      concatenate each edge type's adjacency matrix.
    forward_edge_types: Edge types to use in the forward direction. As a list of
      lists to allow configuring groups of edges in config files; this will be
      flattened before use.
    reverse_edge_types: Edge types to use in the reverse direction. Note that
      reversed edge types are given a separate embedding from forward edge
      types; undirected edges should be represented by adding two edges in
      opposite directions and then only using `forward_edge_types`. Also a list
      of lists, as above.
    components: List of sublayer types. Each element should be the name of one
      of the components defined above.

  Returns:
    Final node embeddings after running all components.
  """
  num_node_types = len(encoding_info.builder.node_types)
  num_edge_types = len(encoding_info.edge_types)
  edge_types_to_indices = {
      edge_type: i for i, edge_type in enumerate(encoding_info.edge_types)
  }
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

  # Build initial node embeddings.
  node_embeddings = (
      graph_layers.PositionalAndTypeNodeEmbedding(
          node_types=input_graph.bundle.node_types,
          num_node_types=num_node_types,
          embedding_dim=node_embedding_dim,
      ) + graph_layers.TokenOperatorNodeEmbedding(
          operator=input_graph.tokens,
          vocab_size=encoding_info.token_encoder.vocab_size,
          num_nodes=static_metadata.num_nodes,
          embedding_dim=node_embedding_dim,
      ))

  if edge_embedding_dim is not None:
    # Learn initial edge embeddings.
    edge_embeddings = graph_layers.LearnableEdgeEmbeddings(
        edges=input_graph.bundle.edges,
        num_nodes=static_metadata.num_nodes,
        num_edge_types=num_edge_types,
        forward_edge_type_indices=forward_edge_type_indices,
        reverse_edge_type_indices=reverse_edge_type_indices,
        embedding_dim=edge_embedding_dim)
  else:
    # Build binary edge embeddings.
    edge_embeddings = graph_layers.binary_index_edge_embeddings(
        edges=input_graph.bundle.edges,
        num_nodes=static_metadata.num_nodes,
        num_edge_types=num_edge_types,
        forward_edge_type_indices=forward_edge_type_indices,
        reverse_edge_type_indices=reverse_edge_type_indices)

  # Run the core component stack.
  # Depending on whether edge_embedding_dim is provided, we either concatenate
  # new edge types or embed them (see end_to_end_stack for details).
  graph_context = end_to_end_stack.SharedGraphContext(
      bundle=input_graph.bundle,
      static_metadata=static_metadata,
      edge_types_to_indices=edge_types_to_indices,
      builder=encoding_info.builder,
      edges_are_embedded=edge_embedding_dim is not None)
  for component in components:
    component_fn = end_to_end_stack.ALL_COMPONENTS[component]
    node_embeddings, edge_embeddings = component_fn(graph_context,
                                                    node_embeddings,
                                                    edge_embeddings)

  return node_embeddings


@flax.deprecated.nn.module
def two_pointer_output_head(node_embeddings,
                            output_mask):
  """Computes two conditionally independent node pointers.

  Conceptually similar to Vasic et al. (2019), but simpler:
  - no masking is applied here, since the model should be able to learn that.
  - since we don't have a sequence model, we skip projecting back the final
    hidden state (as we don't have one) and instead simply compute logits from
    the node embeddings.

  Note that we still return a joint distribution for compatibility with other
  versions of output head.

  Args:
    node_embeddings: Final node embeddings.
    output_mask: Boolean mask for the bug and repair targets.

  Returns:
    NDarray <float32[num_nodes, num_nodes]> of log-probabilities (normalized
      over non-padding nodes).
  """
  logits = flax.deprecated.nn.Dense(node_embeddings, features=2, bias=False)
  logits = jnp.where(output_mask[:, None], logits, -jnp.inf)
  logits = jax.nn.log_softmax(logits, axis=0)
  # Convert from [nodes, 2] to [nodes, nodes] using an outer "product".
  return logits[:, 0, None] + logits[None, :, 1]


@flax.deprecated.nn.module
def bilinear_joint_output_head(
    node_embeddings,
    output_mask):
  """Computes a joint probability distribution with a bilinear transformation.

  Args:
    node_embeddings: Final node embeddings.
    output_mask: Boolean mask for the bug and repair targets.

  Returns:
    NDarray <float32[num_nodes, num_nodes]> of log-probabilities (normalized
      over non-padding nodes).
  """
  logits = graph_layers.BilinearPairwiseReadout(node_embeddings)
  logit_mask = output_mask[:, None] & output_mask[None, :]
  logits = jnp.where(logit_mask, logits, -jnp.inf)
  logits = jax.nn.log_softmax(logits, axis=(0, 1))
  return logits


@flax.deprecated.nn.module
def bug_conditional_output_head(
    node_embeddings,
    output_mask):
  """Computes a factorized joint probability distribution.

  First computes a normalized distribution over bug locations, then combines
  it with a normalized distribution over repairs given bug locations.

  Args:
    node_embeddings: Final node embeddings.
    output_mask: Boolean mask for the bug and repair targets.

  Returns:
    NDarray <float32[num_nodes, num_nodes]> of log-probabilities (normalized
      over non-padding nodes).
  """
  bug_logits = flax.deprecated.nn.Dense(
      node_embeddings, features=1, bias=False).squeeze(-1)
  bug_logits = jnp.where(output_mask, bug_logits, -jnp.inf)
  bug_logits = jax.nn.log_softmax(bug_logits, axis=0)

  repair_logits = graph_layers.BilinearPairwiseReadout(node_embeddings)
  repair_logits = jnp.where(output_mask[None, :], repair_logits, -jnp.inf)
  repair_logits = jax.nn.log_softmax(repair_logits, axis=1)

  return bug_logits[:, None] + repair_logits


VAR_MISUSE_OUTPUT_HEADS = {
    "two_pointer_output_head": two_pointer_output_head,
    "bilinear_joint_output_head": bilinear_joint_output_head,
    "bug_conditional_output_head": bug_conditional_output_head,
}


@flax.deprecated.nn.module
@gin.configurable
def var_misuse_model(
    padded_example,
    padding_config,
    encoding_info,
    head_type = gin.REQUIRED,
    mask_to_candidates = gin.REQUIRED):
  """Full var-misuse model.

  Args:
    padded_example: The example.
    padding_config: Metadata about the padded size of this graph.
    encoding_info: How the example was encoded.
    head_type: Output head to use.
    mask_to_candidates: Whether to mask output to candidate nodes.

  Returns:
    NDarray <float32[num_nodes, num_nodes]> of log-probabilities (normalized
      over non-padding nodes). The first dimension iterates over possible buggy
      nodes, and the second iterates over possible repair nodes.
  """
  node_embeddings = token_graph_model_core(
      padded_example.input_graph,
      padding_config.bundle_padding.static_max_metadata, encoding_info)
  if mask_to_candidates:
    output_mask = padded_example.candidate_node_mask.astype(bool)
  else:
    num_nodes = padded_example.input_graph.bundle.graph_metadata.num_nodes
    output_mask = jnp.arange(node_embeddings.shape[0]) < num_nodes

  return VAR_MISUSE_OUTPUT_HEADS[head_type](node_embeddings, output_mask)

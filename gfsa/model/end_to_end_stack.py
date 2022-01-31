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
"""Model components for an end-to-end-trainable graph/automaton hybrid.

The components defined in this module share a common interface

  (input graph, node embeddings, edge embeddings)
    -> (node embeddings, edge embeddings)

which allows them to be composed with each other. Note that most components
either modify node embeddings or edge embeddings but not both. Also note that
the output edge embeddings are allowed to be of a different size; in particular,
the components that add new edge types use SharedGraphContext.edges_are_embedded
to determine how to modify the edge embeddings.
"""

from typing import Dict, List, Optional, Tuple

import dataclasses
import flax
import gin
import jax
import jax.numpy as jnp

from gfsa import automaton_builder
from gfsa import jax_util
from gfsa.datasets import graph_bundle
from gfsa.model import automaton_layer
from gfsa.model import edge_supervision_models
from gfsa.model import graph_layers
from gfsa.model import model_util
from gfsa.model import side_outputs

# TODO(ddjohnson) Move common layers out of `edge_supervision_models`.

# Flax adds name keyword arguments.
# pylint: disable=unexpected-keyword-arg

NodeAndEdgeEmbeddings = Tuple[jax_util.NDArray, jax_util.NDArray]


@dataclasses.dataclass
class SharedGraphContext:
  """Shared information about the input graph.

  Attributes:
    bundle: The input graph.
    static_metadata: Padded size of the graph.
    edge_types_to_indices: Mapping from string edge names to edge type indices.
    builder: Automaton builder associated with this graph.
    edges_are_embedded: Whether the "edge_embeddings" represent edge types that
      are embedded into vectors (True), or just edge type adjacency matrices
      that are concatenated together.
  """
  bundle: graph_bundle.GraphBundle
  static_metadata: automaton_builder.EncodedGraphMetadata
  edge_types_to_indices: Dict[str, int]
  builder: automaton_builder.AutomatonBuilder
  edges_are_embedded: bool


def _add_edges(old_edge_array,
               new_edge_types,
               edges_are_embedded,
               add_reverse = True):
  """Helper function to add edges of a new edge type.

  If edges_are_embedded=True, we assume `old_edge_dim` is an embedding matrix;
  the new edges are embedded and then added into this matrix. Otherwise, we
  assume `old_edge_dim` is a stacked set of adjacency matrices, and concatenate
  the new types.

  Args:
    old_edge_array: <float32[num_nodes, num_nodes, old_edge_dim]>
    new_edge_types: <float32[num_nodes, num_nodes, new_edge_types]>, which
      should be between 0 and 1, one for each added edge type.
    edges_are_embedded: Whether edge types are embedded.
    add_reverse: Whether to add reverse edges as well with a different type.

  Returns:
    <float32[num_nodes, num_nodes, output_edge_dim]>, where
    output_edge_dim = old_edge_dim if edges_are_embedded=True, and otherwise
    output_edge_dim = old_edge_dim + new_edge_types
  """
  if add_reverse:
    new_edge_types = jnp.concatenate(
        [new_edge_types, new_edge_types.transpose((1, 0, 2))], -1)
  if edges_are_embedded:
    # Project the outputs into new edge embeddings.
    # (No bias is used so that an absorbing probability of 0 produces no change
    # in the edge embeddings.)
    new_edge_type_embeddings = flax.deprecated.nn.Dense(
        new_edge_types,
        features=old_edge_array.shape[-1],
        bias=False,
        name="new_edge_type_embeddings")
    output_edge_array = old_edge_array + new_edge_type_embeddings
  else:
    # Concatenate new embedding.
    output_edge_array = jnp.concatenate([old_edge_array, new_edge_types],
                                        axis=-1)

  return output_edge_array


def _shared_automaton_logic(
    graph_context, node_embeddings,
    edge_embeddings,
    variant_weights):
  """Helper function for shared automaton logic."""

  # Run the automaton.
  edge_weights = automaton_layer.FiniteStateGraphAutomaton(
      encoded_graph=graph_context.bundle.automaton_graph,
      variant_weights=variant_weights,
      dynamic_metadata=graph_context.bundle.graph_metadata,
      static_metadata=graph_context.static_metadata,
      builder=graph_context.builder)

  return (node_embeddings,
          _add_edges(edge_embeddings, edge_weights.transpose([1, 2, 0]),
                     graph_context.edges_are_embedded))


@flax.deprecated.nn.module
@gin.configurable
def variantless_automaton(
    graph_context, node_embeddings,
    edge_embeddings):
  """Runs an automaton without variants.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>

  Returns:
    New node and edge embeddings. Node embeddings will not be modified. Edge
    embeddings will be modified by adding a new edge type (either embedded or
    concatenated based on graph_context.edges_are_embedded).
  """
  return _shared_automaton_logic(
      graph_context, node_embeddings, edge_embeddings, variant_weights=None)


@flax.deprecated.nn.module
@gin.configurable
def edge_variant_automaton(
    graph_context,
    node_embeddings,
    edge_embeddings,
    variant_edge_types = gin.REQUIRED):
  """Runs an automaton with variants based on edges in the input graph.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>
    variant_edge_types: List of edge types used as variants.

  Returns:
    New node and edge embeddings. Node embeddings will not be modified. Edge
    embeddings will be modified by adding a new edge type (either embedded or
    concatenated based on graph_context.edges_are_embedded).
  """
  # Set up variants from edge types.
  variant_edge_type_indices = [
      graph_context.edge_types_to_indices[type_str]
      for type_str in variant_edge_types
  ]
  num_edge_types = len(graph_context.edge_types_to_indices)
  variant_weights = edge_supervision_models.variants_from_edges(
      graph_context.bundle, graph_context.static_metadata,
      variant_edge_type_indices, num_edge_types)

  return _shared_automaton_logic(graph_context, node_embeddings,
                                 edge_embeddings, variant_weights)


@flax.deprecated.nn.module
@gin.configurable
def embedding_variant_automaton(
    graph_context,
    node_embeddings,
    edge_embeddings,
    num_variants = gin.REQUIRED):
  """Runs an automaton with variants based on node embeddings.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>
    num_variants: How many variants to use.

  Returns:
    New node and edge embeddings. Node embeddings will not be modified. Edge
    embeddings will be modified by adding a new edge type (either embedded or
    concatenated based on graph_context.edges_are_embedded).
  """
  if num_variants <= 1:
    raise ValueError(
        "Must have at least one variant to use embedding_variant_automaton.")
  # Generate variants using a pairwise readout of the node embeddings.
  variant_logits = graph_layers.BilinearPairwiseReadout(
      node_embeddings, num_variants, name="variant_logits")

  variant_logits = side_outputs.encourage_discrete_logits(
      variant_logits, distribution_type="categorical", name="variant_logits")
  variant_weights = jax.nn.softmax(variant_logits)

  return _shared_automaton_logic(graph_context, node_embeddings,
                                 edge_embeddings, variant_weights)


@flax.deprecated.nn.module
@gin.configurable
def nri_encoder_readout(
    graph_context,
    node_embeddings,
    edge_embeddings,
    num_edge_types = gin.REQUIRED):
  """Modifies edge embeddings using an NRI encoder.

  Note that we use a sigmoid rather than a softmax, because we don't necessarily
  want to enforce having exactly one edge type per pair of nodes.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>
    num_edge_types: How many edge types to produce.

  Returns:
    New node and edge embeddings. Node embeddings will not be modified. Edge
    embeddings will be modified by adding a new edge type (either embedded or
    concatenated based on graph_context.edges_are_embedded).
  """
  # Run the NRI readout layer.
  logits = graph_layers.NRIReadout(
      node_embeddings=node_embeddings, readout_dim=num_edge_types)
  new_edge_weights = jax.nn.sigmoid(logits)

  mask = (
      jnp.arange(new_edge_weights.shape[0]) <
      graph_context.bundle.graph_metadata.num_nodes)
  new_edge_weights = jnp.where(mask[:, None, None], new_edge_weights,
                               jnp.zeros_like(new_edge_weights))

  return (node_embeddings,
          _add_edges(edge_embeddings, new_edge_weights,
                     graph_context.edges_are_embedded))


class UniformRandomWalk(flax.deprecated.nn.Module):
  """Adds edges according to a uniform random walk along the graph."""

  @gin.configurable("UniformRandomWalk")
  def apply(
      self,
      graph_context,
      node_embeddings,
      edge_embeddings,
      forward_edge_types = gin.REQUIRED,
      reverse_edge_types = gin.REQUIRED,
      walk_length_log2 = gin.REQUIRED,
  ):
    """Modifies edge embeddings using a uniform random walk.

    Uses an efficient repeated-squaring technique to compute the absorbing
    distribution.

    Args:
      graph_context: Input graph for this example.
      node_embeddings: Current node embeddings, as <float32[num_nodes,
        node_embedding_dim]>
      edge_embeddings: Current edge embeddings, as <float32[num_nodes,
        num_nodes, edge_embedding_dim]>
      forward_edge_types: Edge types to use in the forward direction. As a list
        of lists to allow configuring groups of edges in config files; this will
        be flattened before use.
      reverse_edge_types: Edge types to use in the reverse direction. Note that
        reversed edge types are given a separate embedding from forward edge
        types; undirected edges should be represented by adding two edges in
        opposite directions and then only using `forward_edge_types`. Also a
        list of lists, as above.
      walk_length_log2: Base-2 logarithm of maximum walk length; this determines
        how many times we will square the transition matrix (doubling the walk
        length).

    Returns:
      New node and edge embeddings. Node embeddings will not be modified. Edge
      embeddings will be modified by adding a new edge type (either embedded or
      concatenated based on graph_context.edges_are_embedded).
    """
    num_nodes = node_embeddings.shape[0]

    # pylint: disable=g-complex-comprehension
    forward_edge_type_indices = [
        graph_context.edge_types_to_indices[type_str]
        for group in forward_edge_types
        for type_str in group
    ]
    reverse_edge_type_indices = [
        graph_context.edge_types_to_indices[type_str]
        for group in reverse_edge_types
        for type_str in group
    ]
    # pylint: enable=g-complex-comprehension

    adjacency = graph_layers.edge_mask(
        edges=graph_context.bundle.edges,
        num_nodes=num_nodes,
        num_edge_types=len(graph_context.edge_types_to_indices),
        forward_edge_type_indices=forward_edge_type_indices,
        reverse_edge_type_indices=reverse_edge_type_indices)
    adjacency = jnp.maximum(adjacency, jnp.eye(num_nodes))

    absorbing_logit = self.param(
        "absorbing_logit",
        shape=(),
        initializer=lambda *_: jax.scipy.special.logit(0.1))

    absorbing_prob = jax.nn.sigmoid(absorbing_logit)
    nonabsorbing_prob = jax.nn.sigmoid(-absorbing_logit)
    walk_matrix = nonabsorbing_prob * adjacency / jnp.sum(
        adjacency, axis=1, keepdims=True)

    # A, I
    # A^2, A + I
    # (A^2)^2 = A^4, (A + I)A^2 + (A + I) = A^3 + A^2 + A + I
    # ...

    def step(state, _):
      nth_power, nth_partial_sum = state
      return (nth_power @ nth_power,
              nth_power @ nth_partial_sum + nth_partial_sum), None

    (_, partial_sum), _ = jax.lax.scan(
        step, (walk_matrix, jnp.eye(num_nodes)), None, length=walk_length_log2)

    approx_visits = absorbing_prob * partial_sum
    logits = model_util.safe_logit(approx_visits)
    logits = model_util.ScaleAndShift(logits)
    edge_weights = jax.nn.sigmoid(logits)

    return (node_embeddings,
            _add_edges(edge_embeddings, edge_weights[:, :, None],
                       graph_context.edges_are_embedded))


@flax.deprecated.nn.module
def ggnn_adapter(graph_context,
                 node_embeddings,
                 edge_embeddings):
  """Adapter function to run GGNN steps.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>

  Returns:
    New node and edge embeddings. Node embeddings are processed by a GGNN,
    and edge embeddings are returned unchanged.
  """
  del graph_context
  return (
      edge_supervision_models.ggnn_steps(node_embeddings, edge_embeddings),
      edge_embeddings,
  )


@flax.deprecated.nn.module
def transformer_adapter(
    graph_context, node_embeddings,
    edge_embeddings):
  """Adapter function to run transformer blocks.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>

  Returns:
    New node and edge embeddings. Node embeddings are processed by a
    transformer, and edge embeddings are returned unchanged.
  """
  return (
      edge_supervision_models.transformer_steps(
          node_embeddings,
          edge_embeddings,
          neighbor_mask=None,
          num_real_nodes_per_graph=(
              graph_context.bundle.graph_metadata.num_nodes),
          mask_to_neighbors=False),
      edge_embeddings,
  )


@flax.deprecated.nn.module
def nri_adapter(graph_context,
                node_embeddings,
                edge_embeddings):
  """Adapter function to run NRI blocks.

  Args:
    graph_context: Input graph for this example.
    node_embeddings: Current node embeddings, as <float32[num_nodes,
      node_embedding_dim]>
    edge_embeddings: Current edge embeddings, as <float32[num_nodes, num_nodes,
      edge_embedding_dim]>

  Returns:
    New node and edge embeddings. Node embeddings are processed by a NRI-style
    model, and edge embeddings are returned unchanged.
  """
  return (
      edge_supervision_models.nri_steps(
          node_embeddings,
          edge_embeddings,
          num_real_nodes_per_graph=(
              graph_context.bundle.graph_metadata.num_nodes)),
      edge_embeddings,
  )


ALL_COMPONENTS = {
    "variantless_automaton": variantless_automaton,
    "edge_variant_automaton": edge_variant_automaton,
    "embedding_variant_automaton": embedding_variant_automaton,
    "nri_encoder_readout": nri_encoder_readout,
    "ggnn_adapter": ggnn_adapter,
    "transformer_adapter": transformer_adapter,
    "nri_adapter": nri_adapter,
    "UniformRandomWalk": UniformRandomWalk,
}

# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Defines models."""

from typing import Callable, Sequence, Optional

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import jraph


class MultiLayerPerceptron(nn.Module):
  """A multi-layer perceptron (MLP)."""

  latent_sizes: Sequence[int]
  activation: Optional[Callable[[chex.Array], chex.Array]]
  skip_connections: bool = False
  activate_final: bool = False

  @nn.compact
  def __call__(self, inputs):
    for index, dim in enumerate(self.latent_sizes):
      next_inputs = nn.Dense(dim)(inputs)

      if index != len(self.latent_sizes) - 1 or self.activate_final:
        if self.activation is not None:
          next_inputs = self.activation(next_inputs)

      if self.skip_connections and next_inputs.shape == inputs.shape:
        next_inputs = next_inputs + inputs

      inputs = next_inputs
    return inputs


class GraphMultiLayerPerceptron(nn.Module):
  """A multi-layer perceptron (MLP) applied to the node features."""

  dimensions: Sequence[int]
  activation: Callable[[chex.Array], chex.Array]

  @nn.compact
  def __call__(self, graph):
    mlp = MultiLayerPerceptron(
        self.dimensions,
        self.activation,
        skip_connections=False,
        activate_final=False)
    return graph._replace(nodes=mlp(graph.nodes))


class OneHopGraphConvolution(nn.Module):
  """Performs one hop of a graph convolution with weighted edges."""

  update_fn: Callable[[chex.Array], chex.Array]
  num_partitions: int = 5

  @nn.compact
  def __call__(self, graph):
    # Message-passing occurs against the direction of the input edges.
    senders, receivers = graph.receivers, graph.senders
    if senders is None:
      raise ValueError('Graph must have senders and receivers.')

    num_nodes = jax.tree.leaves(graph.nodes)[0].shape[0]
    num_edges = senders.shape[0]

    # Compute the convolution by partitioning the edges.
    # This saves a significant amount of memory.
    num_partitions = min(num_edges, self.num_partitions)
    partition_size = num_edges // num_partitions
    convolved_nodes = jnp.zeros_like(graph.nodes)
    for step in range(num_edges // partition_size + 1):
      partition_start = partition_size * step
      partition_end = partition_size * (step + 1)
      partition_end = min(partition_end, num_edges)
      partition_edges = graph.edges[partition_start:partition_end]
      partition_senders = senders[partition_start:partition_end]
      partition_receivers = receivers[partition_start: partition_end]
      weighted_edges = partition_edges * graph.nodes[partition_senders]
      convolved_nodes += jraph.segment_sum(
          weighted_edges,
          partition_receivers,
          num_nodes,
          indices_are_sorted=False)

    # Update node features.
    convolved_nodes = self.update_fn(convolved_nodes)
    return graph._replace(nodes=convolved_nodes)


class GraphConvolutionalNetwork(nn.Module):
  """A graph convolutional neural network from Kipf, et al. (2016)."""

  latent_size: int
  num_classes: int
  num_message_passing_steps: int
  num_encoder_layers: int
  num_decoder_layers: int
  activation: Callable[[chex.Array], chex.Array]

  @nn.compact
  def __call__(self, graph):
    # Encoder.
    encoder = MultiLayerPerceptron(
        [self.latent_size] * self.num_encoder_layers,
        self.activation,
        skip_connections=False,
        activate_final=True,
        name='encoder')
    graph = jraph.GraphMapFeatures(embed_node_fn=encoder)(graph)

    # Core.
    for hop in range(self.num_message_passing_steps):
      node_update_fn = MultiLayerPerceptron([self.latent_size],
                                            self.activation,
                                            skip_connections=True,
                                            activate_final=True,
                                            name=f'core_{hop}')
      core = OneHopGraphConvolution(update_fn=node_update_fn)
      graph = core(graph)

    # Decoder.
    decoder = MultiLayerPerceptron(
        [self.latent_size] * (self.num_decoder_layers - 1) + [self.num_classes],
        self.activation,
        skip_connections=False,
        activate_final=False,
        name='decoder')
    graph = jraph.GraphMapFeatures(embed_node_fn=decoder)(graph)
    return graph

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

"""Agents that define transition probabilities over nodes and neighbors.

These agents implicitly define an adjacency matrix on a graph.
"""

import abc
import functools

import flax
from flax import struct
import flax.linen as nn
import jax
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

import numpy as np

from jaxsel._src import graph_api



flax_dataclass = flax.struct.dataclass

# Helper functions


def _make_adjacency_mat_row_indices(node_id,
                                    neighbor_node_ids):
  """Turns outgoing node ids and neighbor ids into sparse matrix indices.

  This allows us to fill in a sparse matrix from the node_ids.

  Args:
    node_id: id of the row to be filled.
    neighbor_node_ids: ids of the neighbors of the initial node. These will be
      column ids.

  Returns:
    indices: the indices in the adjacency matrix of the weights from
      node_id to its neighbors. All these elements are in the same row
      of the adjacency matrix, since they share the outgoing node.
  """
  n_neighbors = neighbor_node_ids.shape[0]
  return jnp.stack(
      (
          jnp.full(n_neighbors, node_id, dtype="int32"),
          jnp.array(neighbor_node_ids),
      ),
      axis=-1,
  )


################
# Agent models #
################


class Agent(abc.ABC):
  """Abstract base class for all agents.

  An agent defines an adjacency matrix on a given graph.
  """

  @abc.abstractmethod
  def __call__(self, task_features, node_features, neighbor_relations,
               neighbor_features):
    """Maps a node and neighbor features to logits (one per neighbor).

    Args:
      task_features: int or float tensor representing fixed features of the task
      node_features: int or float tensor representing fixed features of the
        current node
      neighbor_relations: int tensor of size `num_outgoing_edges`. It encodes
        the type of edge between node and neighbor.
      neighbor_features: int or float tensor of size `num_outgoing_edges x
        feature_dim` representing features of the neighbor nodes

    Returns:
      A float tensor of size `num_outgoing_edges` with logits associated with
        the probability of transitioning to that neighbor.
    """
    Ellipsis

  def walk(self, graph):
    """Performs a random walk on the graph.

    This is just a demonstration. We don't intend to use it for
    anything serious at the moment.

    Args:
      graph: The underlying graph to walk.

    Returns:
      A list of node IDs visited on a random walk.
    """
    cur_id = graph.sample_start_node_id()

    node_ids = [cur_id]
    for _ in range(50):
      neighbor_relations, neighbor_features, neighbor_ids = (
          graph.outgoing_neighbors_and_features(cur_id))
      num_neighbors = neighbor_ids.shape[0]
      node_features = jnp.tile(graph.node_features(cur_id), [num_neighbors, 1])
      neighbor_logits = self(None, node_features, neighbor_relations,
                             neighbor_features)

      # Sample neighbor with Gumbel-max.
      gumbels = -np.log(-np.log(np.random.rand(*neighbor_logits.shape)))
      neighbor_id = np.argmax(neighbor_logits + gumbels)
      cur_id = neighbor_id
      node_ids.append(cur_id)
    return node_ids

  @functools.partial(
      nn.vmap,
      variable_axes={"params": None},
      in_axes=(0, None, None),
      split_rngs={"params": False})
  def _fill_rows_of_sparse_adjacency_matrix(
      self, node_ids, graph,
      max_graph_size):
    """Computes rows corresponding to `node_ids` of the sparse adjacency matrix."""
    neighbor_relations, neighbor_features, neighbor_ids = (
        graph.outgoing_neighbors_and_features(node_ids))
    num_neighbors = neighbor_ids.shape[0]
    node_features = jnp.tile(graph.node_features(node_ids), [num_neighbors, 1])
    logits = self(graph.task_features(), node_features, neighbor_relations,
                  neighbor_features)
    probs = jax.nn.softmax(logits)
    sparse_rows = jsparse.BCOO(
        (probs, _make_adjacency_mat_row_indices(node_ids, neighbor_ids)),
        shape=(max_graph_size, max_graph_size))
    return sparse_rows

  def fill_sparse_adjacency_matrix(self, q,
                                   graph):
    """Computes outgoing transition probabilities from nodes in the subgraph.

    This method takes node weights represented by sparse vector `q` and returns
    the local adjacency matrix: the restriction of the full adjacency matrix to
    rows corresponding to nonzero elements of `q`.

    Args:
      q: Sparse representation of the node weights. Indices of nonzero elements
        represent the subgraph to restrict the graph to.
      graph: The underlying unweighted graph. Allows to query neighbors.

    Returns:
      sparse_adjacency_submat: A sparse representation of the adjacency matrix
        where each edge starts from a node with non zero weight in `q`.
    """
    # TODO(gnegiar): use XOR on boolean mask from q_new and q_old
    # Also check if q_data is nonzero
    # to fill only needed rows

    # Deduplicate: a given node in q may have been computed as the neighbor of
    # multiple nodes. In that case, it will appear twice in q's sparse repr.
    q = q.sum_duplicates(q.nse)
    node_ids = q.indices.flatten()
    adjacency_matrix = self._fill_rows_of_sparse_adjacency_matrix(
        node_ids, graph, q.shape[0]).sum(0)
    return adjacency_matrix.sum_duplicates(adjacency_matrix.nse)


########################
#  Flax based models   #
########################


def product(iterable, start=1.):
  return functools.reduce(jnp.multiply, iterable, start)


@struct.dataclass
class AgentConfig:
  graph_parameters: graph_api.GraphParameters
  embedding_dim: int
  hidden_dim: int


class SimpleFiLMedAgentModel(nn.Module, Agent):
  """An agent mixing task, node, relation and neighbor embeddings.

  This is a simple version, where only 2 streams are mixed multiplicatively.

  Attributes:
    graph_parameters: Represents the graphs to be weighted.
    embedding_dim: dimension of initial embeddings.
    hidden_dim: dimension of the FiLMed layer.
  """
  config: AgentConfig

  def setup(self):
    # Embeddings
    self.node_embedding = nn.Embed(
        self.config.graph_parameters.node_vocab_size,
        self.config.embedding_dim,
        name="Node feature embedding")
    self.relation_embedding = nn.Embed(
        self.config.graph_parameters.num_relation_types,
        self.config.embedding_dim,
        name="Relation embedding")
    self.neighbor_embedding = nn.Embed(
        self.config.graph_parameters.node_vocab_size,
        self.config.embedding_dim,
        name="Neighbor feature embedding")
    self.input_embedding = (self.node_embedding, self.relation_embedding,
                            self.neighbor_embedding)

    # Hidden layers for multiplication in FiLM-like layer
    self.node_hidden = nn.Dense(
        self.config.hidden_dim, name="Node hidden layer")
    self.relation_hidden_mul = nn.Dense(
        self.config.hidden_dim, name="Relation hidden layer Mul")

    # Hidden layers for bias in FiLM-like layer
    self.relation_hidden_sum = nn.Dense(
        self.config.hidden_dim, name="Relation hidden layer sum")

    self.output_layer = nn.Dense(1, name="Output layer")

  def __call__(self, task_features, node_features, neighbor_relations,
               neighbor_features):
    """Maps task, node, edge and neighbors to logits.

    We use both multiplicative and additive dependency between
      representations of task, node, neighbors, neighbor_features.

    Args:
      task_features: Not used.
      node_features: Integer tensor of size `num_neighbors x num_node_features`
        with (duplicated) features of the current node.
      neighbor_relations: Integer tensor of size `num_neighbors`.
      neighbor_features: Integer tensor of size `num_neighbors x
        num_node_features` of neighbor features.

    Returns:
      A float tensor of size `num_neighbors` with logits associated with
      the probability of transitioning to that neighbor.
    """
    del task_features
    num_neighbors = len(neighbor_relations)
    # Embed all inputs
    inputs = (node_features, neighbor_relations, neighbor_features)
    embeddings = [emb(x) for emb, x in zip(self.input_embedding, inputs)]

    # Reshape
    node_embs, relation_embs, neighbor_embs = embeddings
    node_embs = node_embs.reshape(num_neighbors, -1)
    relation_embs = relation_embs.reshape(num_neighbors, -1)
    neighbor_embs = neighbor_embs.reshape(num_neighbors, -1)

    local_feature_embs = jnp.column_stack((node_embs, neighbor_embs))
    hidden_local_features = self.node_hidden(local_feature_embs)
    hidden_relation_embs = self.relation_hidden_mul(relation_embs)
    hidden_relation_bias = self.relation_hidden_sum(relation_embs)

    # FiLM-like layer to mix representations of neighborhood (node+neighbor)
    # and edge
    mixed_representation = hidden_local_features * hidden_relation_embs + hidden_relation_bias

    mixed_representation = nn.relu(mixed_representation)
    neighbor_logits = self.output_layer(mixed_representation)
    return neighbor_logits.squeeze(-1)


class GeneralizedMultiplicativeAgentModel(nn.Module, Agent):
  """An agent mixing task, node, relation and neighbor embeddings.

  Attributes:
    graph_parameters: Represents the graphs to be weighted.
    embedding_dim: dimension of initial embeddings.
    hidden_dim: dimension of the FiLMed layer.
  """
  config: AgentConfig

  def setup(self):
    # Embeddings
    self.task_embedding = nn.Embed(
        self.config.graph_parameters.task_vocab_size,
        self.config.embedding_dim,
        name="Task embedding")
    self.node_embedding = nn.Embed(
        self.config.graph_parameters.node_vocab_size,
        self.config.embedding_dim,
        name="Node feature embedding")
    self.relation_embedding = nn.Embed(
        self.config.graph_parameters.num_relation_types,
        self.config.embedding_dim,
        name="Relation embedding")
    self.neighbor_embedding = nn.Embed(
        self.config.graph_parameters.node_vocab_size,
        self.config.embedding_dim,
        name="Neighbor feature embedding")
    self.input_embedding = (self.task_embedding, self.node_embedding,
                            self.relation_embedding, self.neighbor_embedding)

    # Hidden layers for multiplication in FiLM-like layer
    self.task_hidden_mul = nn.Dense(
        self.config.hidden_dim, name="Task hidden layer Mul")
    self.node_hidden_mul = nn.Dense(
        self.config.hidden_dim, name="Node hidden layer Mul")
    self.relation_hidden_mul = nn.Dense(
        self.config.hidden_dim, name="Relation hidden layer Mul")
    self.neighbor_hidden_mul = nn.Dense(
        self.config.hidden_dim, name="Neighbor hidden layer Mul")
    self.hidden_layers_mul = (self.task_hidden_mul, self.node_hidden_mul,
                              self.relation_hidden_mul,
                              self.neighbor_hidden_mul)

    # Hidden layers for bias in FiLM-like layer
    self.task_hidden_sum = nn.Dense(
        self.config.hidden_dim, name="Task hidden layer sum")
    self.node_hidden_sum = nn.Dense(
        self.config.hidden_dim, name="Node hidden layer sum")
    self.relation_hidden_sum = nn.Dense(
        self.config.hidden_dim, name="Relation hidden layer sum")
    self.neighbor_hidden_sum = nn.Dense(
        self.config.hidden_dim, name="Neighbor hidden layer sum")
    self.hidden_layers_sum = (self.task_hidden_sum, self.node_hidden_sum,
                              self.relation_hidden_sum,
                              self.neighbor_hidden_sum)

    self.output_layer = nn.Dense(1, name="Output layer")

  def __call__(self, task_features, node_features, neighbor_relations,
               neighbor_features):
    """Maps task, node, edge and neighbors to logits.

    We use both multiplicative and additive dependency between
      representations of task, node, neighbors, neighbor_features.

    Args:
      task_features: Integer tensor of size `num_task_features` task features.
      node_features: Integer tensor of size `num_neighbors x num_node_features`
        with (duplicated) features of the current node.
      neighbor_relations: Integer tensor of size `num_neighbors`.
      neighbor_features: Integer tensor of size `num_neighbors x
        num_node_features` of neighbor features.

    Returns:
      A float tensor of size `num_neighbors` with logits associated with
      the probability of transitioning to that neighbor.
    """
    num_neighbors = len(neighbor_relations)
    # Embed all inputs
    inputs = (task_features, node_features, neighbor_relations,
              neighbor_features)
    embeddings = [emb(x) for emb, x in zip(self.input_embedding, inputs)]

    # Reshape
    task_embs, node_embs, relation_embs, neighbor_embs = embeddings
    task_embs = task_embs.reshape(1, -1)
    task_embs = jnp.repeat(task_embs, num_neighbors,
                           0)  # Duplicate task embs for each neighbor
    node_embs = node_embs.reshape(num_neighbors, -1)
    relation_embs = relation_embs.reshape(num_neighbors, -1)
    neighbor_embs = neighbor_embs.reshape(num_neighbors, -1)

    reshaped_embeddings = (task_embs, node_embs, relation_embs, neighbor_embs)

    hidden_mul = [
        hidden_layer_mul(x) for hidden_layer_mul, x in zip(
            self.hidden_layers_mul, reshaped_embeddings)
    ]
    hidden_sum = [
        hidden_layer_sum(x) for hidden_layer_sum, x in zip(
            self.hidden_layers_sum, reshaped_embeddings)
    ]
    # FiLM-like layer to mix representations of task, node, neighbor
    mixed_representation = product(hidden_mul) + sum(hidden_sum)
    mixed_representation = nn.relu(mixed_representation)
    neighbor_logits = self.output_layer(mixed_representation)
    return neighbor_logits.squeeze(-1)



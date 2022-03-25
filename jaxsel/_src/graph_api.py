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

"""Abstract base class for implicit access to a potentially giant graph."""
import abc
import enum

from typing import Any, Mapping, Sequence, Text, Tuple

from flax import struct
import jax
import numpy as np


class NodeFeatureKind(enum.Enum):
  CATEGORICAL = 0  # Each feature is a categorical value.
  REAL = 1  # Each feature is real-valued.


@struct.dataclass
class GraphParameters:
  """Holds a graph's characteristics.

  Attributes:
    node_vocab_size: Number of different node types.
    num_relation_types: Number of different edge types.
    node_feature_dim: Dimension of the fixed representation of each node.
    node_feature_kind: Determines whether the node features are categorical or
      real.
    task_vocab_size: Number of different types of tasks.
    task_feature_dim: Dimension of the fixed representation of each task.
    task_feature_kind: Determines whether the task features are categorical or
      real.
  """
  node_vocab_size: int = struct.field(pytree_node=False)
  num_relation_types: int = struct.field(pytree_node=False)
  node_feature_dim: int = struct.field(pytree_node=False)
  node_feature_kind: NodeFeatureKind = struct.field(pytree_node=False)
  task_vocab_size: int = struct.field(pytree_node=False)
  task_feature_dim: int = struct.field(pytree_node=False)
  task_feature_kind: NodeFeatureKind = struct.field(pytree_node=False)


class GraphAPI(abc.ABC):
  """API that we expect for implicit access to a giant graph.

  Node IDs do not need to be contiguous.
  """

  ########################################################################
  #
  # Core API.
  #
  ########################################################################

  @abc.abstractmethod
  def sample_start_node_id(self, seed=0):
    """Samples an ID of the start node.

    Each call may return a different value.

    Args:
      seed: Random seed.

    Returns:
      A node ID.
    """
    pass

  @abc.abstractmethod
  def task_features(self):
    """."""
    pass

  @abc.abstractmethod
  def node_features(self, node_id):
    """Gets a feature vector associated with `node_id`."""
    pass

  @abc.abstractmethod
  def outgoing_edges(self, node_id):
    """Gets list of edges outgoing from `node_id`.

    Args:
      node_id: Integer ID of node to get outgoing edges from.

    Returns:
      A list of (relation_id, neighbor_id) pairs specifying that there is a
      directed edge of type `relation_id` from `node_id` to `neighbor_id`.
    """
    Ellipsis

  @abc.abstractmethod
  def node_metadata(self, node_id):
    """Uninterpreted dictionary of metadata about a given `node_id`.

    Meant to be used for debugging and visualization.

    Args:
      node_id: Integer ID of node to get metadata for.

    Returns:
      A dictionary with string keys specifying the kind of metadata, associated
      with arbitrary values.
    """
    Ellipsis

  @abc.abstractmethod
  def graph_parameters(self):
    Ellipsis

  ########################################################################
  #
  # Additional functionality derived from the core API.
  #
  ########################################################################

  def outgoing_neighbors_and_features(self, node_id):
    """Gets neighbors where there is an edge from `node_id` to the neighbor.

    Args:
      node_id: Integer ID of node to get neighbors and features for.

    Returns:
      A tuple of
        neighbor_relations: A `num_neighbors` integer tensor of neighbor
          relation ids (edge types).
        neighbor_features: A `num_neighbors x num_features` node of features
          for each neighbor.
        neighbor_ids: A `num_neighbors` integer tensor of neighbor node IDs.
    """
    neighbor_relations, neighbor_ids = self.outgoing_edges(node_id)
    # TODO(gnegiar): The vmap seems to interfere with pytype
    neighbor_features = jax.vmap(self.node_features)(neighbor_ids)  # pytype: disable=wrong-arg-types

    return neighbor_relations, neighbor_features, neighbor_ids

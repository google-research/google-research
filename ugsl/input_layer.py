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

"""Input layer of the GSL Model.

The input layer in the GSL model takes as input the name of the dataset and can
return node features corresponding to the dataset and the initial adjacency.
"""

from typing import Any, Mapping, Tuple

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_gnn as tfgnn

from ugsl import datasets
from ugsl import position_encoders


@tf.keras.utils.register_keras_serializable(package='GSL')
class InputLayer:
  """Creates an input layer to provide the necessary inputs for GSL modules.

  Attributes:
    graph_data: the gsl in-memory graph data for a given dataset name.
    wl_roles: A WL coloring for each edge set.
    shortest_paths: The shortest path distance between pairs of nodes for each
      edge set. Unset until get_wl_roles() is called.
    shortest_path_limit: The maximum path length considered when shortest paths
      was computed. Unset until get_shortest_paths() is called.
  """

  def __init__(
      self,
      name,
      remove_noise_ratio,
      add_noise_ratio,
      add_wl_position_encoding,
      add_spectral_encoding,
  ):
    self._graph_data = datasets.get_in_memory_graph_data(
        name,
        remove_noise_ratio,
        add_noise_ratio,
    )
    self._add_wl_position_encoding = add_wl_position_encoding
    self._add_spectral_encoding = add_spectral_encoding
    self._wl_roles = None
    self._shortest_paths = None
    self._shortest_path_limit = None
    self._name = name
    self._adjacency_matrix = None
    self._graph_bert_intimacy = None
    self._graph_spectrum = None
    self._graph_spectrum_dim = None

  def get_initial_node_features(self):
    """Give initial node features, optionally with position encodings."""
    raw_features = self._graph_data.node_features_dicts()['nodes']['feat']
    if self._add_wl_position_encoding:
      positions = self.get_wl_roles()
      for roles in positions.values():
        raw_features += tf.cast(
            position_encoders.get_sinusoidal_encoding_tensor(
                roles, raw_features.shape[1]
            ),
            dtype=tf.float32,
        )
    if self._add_spectral_encoding:
      for spectrum in self.get_graph_spectrum(raw_features.shape[1]).values():
        raw_features += spectrum

    return raw_features

  def get_initial_adj(self):
    return self._graph_data.edge_lists()

  def get_number_of_nodes(self):
    return len(self._graph_data.node_features_dicts()['nodes']['feat'])

  def get_number_of_classes(self):
    return self._graph_data.num_classes()

  def get_labels(self):
    return self._graph_data.labels()

  def get_all_labels(self):
    return self._graph_data.test_labels()

  def get_graph_data(self):
    return self._graph_data

  def get_node_split(self):
    return self._graph_data.node_split()

  def get_input_graph_tensor(self):
    return self._graph_data.as_graph_tensor()

  def get_input_graph_tensor_with_noise(
      self, remove_noise_ratio, add_noise_ratio
  ):
    return self._graph_data.as_graph_tensor_noisy_adjacency(
        remove_noise_ratio=remove_noise_ratio, add_noise_ratio=add_noise_ratio
    )

  def get_dataset_statistics(self):
    """Creates a dictionary with some statistics on the dataset."""
    statistics = {}
    statistics['number_of_nodes'] = self.get_number_of_nodes()
    statistics['number_of_features'] = self.get_initial_node_features().shape[1]
    statistics['number_of_classes'] = self.get_number_of_classes()
    statistics['number_of_edges'] = (
        self.get_input_graph_tensor()
        .edge_sets['edges']
        .adjacency.source.shape[0]
    )
    statistics['number_of_train_nodes'] = len(
        self._graph_data.node_split().train
    )
    statistics['number_of_validation_nodes'] = len(
        self._graph_data.node_split().validation
    )
    statistics['number_of_test_nodes'] = len(self._graph_data.node_split().test)
    statistics['label_rate'] = (
        statistics['number_of_train_nodes']
    ) / statistics['number_of_nodes']
    return statistics

  def get_wl_roles(
      self,
  ):
    """Computes a WL coloring for each set of edges independently.

    Output is memoized, so subsequent calls will be much faster than the first.

    Returns:
      A mapping from each (src_node_set, edge_set, targ_node_set) to dict
      mapping node ids to sequential color ids.
    """
    if self._wl_roles is not None:
      return self._wl_roles
    self._wl_roles = {}
    for edge_set_id, edge_list in self.get_initial_adj().items():
      adj_list = [[] for _ in range(self.get_number_of_nodes())]
      for i in range(edge_list.shape[-1]):
        adj_list[edge_list[0, i]].append(edge_list[1, i])
      self._wl_roles[edge_set_id] = position_encoders.wl_coloring(adj_list)
    return self._wl_roles

  def get_shortest_paths(
      self, max_dist
  ):
    """Gives the shortest paths between all pairs of nodes.

    Output is memoized, so subsequent calls will be much faster than the first.
    If max_dist is not the same as the one used for the memoized result, we
    recompute.

    Args:
      max_dist: The maximum distance to consider. Nodes which are further apart
        than this will not have a distance computed.

    Returns:
      A mapping from each (src_node_set, edge_set, targ_node_set) to dict
      mapping node ids to sequential color ids.
    """
    if (
        self._shortest_paths is not None
        and max_dist == self._shortest_path_limit
    ):
      return self._shortest_paths
    self._shortest_paths = {}
    self._shortest_path_limit = max_dist
    for edge_set_id, edge_list in self.get_initial_adj().items():
      adj_dict = {}
      for u, v in edge_list:
        if u not in adj_dict:
          adj_dict[u] = [v]
        else:
          adj_dict[u].append(v)
      self._shortest_paths[edge_set_id] = (
          position_encoders.all_pairs_shortest_path(adj_dict, max_dist)
      )
    return self._shortest_paths

  def get_config(self):
    return dict(
        name=self._name,
    )

  def get_adjacency_matrix(
      self,
  ):
    """Gets the adjacency matrix, per edge set.

    Returns:
      A dict mapping edge set ids to sparse adjacency matrices.
    """
    if self._adjacency_matrix is None:
      self._adjacency_matrix = {}
      for edge_set_id, edge_list in self.get_initial_adj().items():
        adj_list = [
            (edge_list[0, i], edge_list[1, i])
            for i in range(edge_list.shape[-1])
        ]
        self._adjacency_matrix[edge_set_id] = (
            position_encoders.edge_list_to_adj_matrix(adj_list)
        )
    return self._adjacency_matrix

  def get_graph_bert_intimacy(
      self,
  ):
    """Gives the 'intimacy' scores following the GraphBert paper.

    Returns:
      A matrix of the pairwise intimacy scores for each edge set, following the
      index ordering used in self._adjacency_matrix_indices.
    """
    if self._graph_bert_intimacy is None:
      self._graph_bert_intimacy = {}
      adj_matrix = self.get_adjacency_matrix()
      for edge_set_id, adj in adj_matrix.items():
        self._graph_bert_intimacy[edge_set_id] = (
            position_encoders.graph_bert_intimacy(adj)
        )
    return self._graph_bert_intimacy

  def get_graph_spectrum(
      self, dim
  ):
    """Computes the top eigenvectors of a each edge sets adjacency matrix.

    Args:
      dim: The number of eigenvectors to compute, per edge set.

    Returns:
      A mapping from edge set to eigenvectors as an N x dim array. (Where N
      is the number of nodes). The indices in the array correspond to the values
      in self._adjacency_matrix_indices.
    """
    if self._graph_spectrum is None or dim != self._graph_spectrum_dim:
      self._graph_spectrum = {}
      for edge_set_id, adj in self.get_adjacency_matrix().items():
        self._graph_spectrum[edge_set_id] = position_encoders.graph_spectrum(
            adj, dim
        )
      self._graph_spectrum_dim = dim
    return self._graph_spectrum

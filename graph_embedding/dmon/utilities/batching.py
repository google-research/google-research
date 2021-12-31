# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""TODO(tsitsulin): add headers, tests, and improve style."""

from typing import List

import numpy as np
import scipy.sparse


def subgraph(graph, seed, n_neighbors):  # pylint: disable=missing-function-docstring
  total_matrix_size = 1 + np.cumprod(n_neighbors).sum()
  picked_nodes = {seed}
  last_layer_nodes = {seed}
  # Number of nodes to pick at each layer. Initially, only the seed is picked.
  to_pick = 1
  for n_neighbors_current in n_neighbors:
    to_pick = to_pick * n_neighbors_current
    neighbors = graph[list(last_layer_nodes), :].nonzero()[1]
    neighbors = list(set(
        neighbors))  # Make a set neighbors of all nodes from the last layer.
    n_neigbors_real = min(
        to_pick,
        len(neighbors))  # Handle case there are fewer neighbors than desired.
    last_layer_nodes = set(
        np.random.choice(neighbors, n_neigbors_real, replace=False))
    picked_nodes |= last_layer_nodes
  indices = [seed] + list(sorted(picked_nodes - {seed}))
  matrix = graph[indices, :][:, indices]
  matrix.resize((total_matrix_size, total_matrix_size))
  return matrix.todense().A1.reshape(total_matrix_size,
                                     total_matrix_size), indices


def make_batch(graph, features,  # pylint: disable=missing-function-docstring
               batch_nodes, n_neighbors):
  total_matrix_size = 1 + np.cumprod(n_neighbors).sum()
  batch_size = len(batch_nodes)
  graph_ss = np.zeros((batch_size, total_matrix_size,
                       total_matrix_size))  # Subsampled graph matrix.
  features_ss = np.zeros((batch_size, total_matrix_size,
                          features.shape[1]))  # Subsampled feature matrix.
  subgraph_sizes = np.zeros(batch_size, dtype=np.int)
  for index, node in enumerate(batch_nodes):
    graph_ss[index, :, :], indices = subgraph(graph, node, n_neighbors)
    subgraph_sizes[index] = len(indices)
    features_ss[index, :subgraph_sizes[index], :] = features[indices, :]
  return graph_ss, features_ss, subgraph_sizes


def full_graph_batch(graph, features,
                     n_neighbors):
  node_ids = np.arange(graph.shape[0])
  graph_ss, features_ss, subgraph_sizes = make_batch(graph, features, node_ids,
                                                     n_neighbors)
  return graph_ss, features_ss, node_ids, subgraph_sizes


def random_batch(graph, features,
                 batch_size, n_neighbors):
  node_ids = np.random.randint(graph.shape[0], size=batch_size)
  graph_ss, features_ss, subgraph_sizes = make_batch(graph, features, node_ids,
                                                     n_neighbors)
  return graph_ss, features_ss, node_ids, subgraph_sizes

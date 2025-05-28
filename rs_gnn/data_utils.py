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

"""Data utils for RS-GNN."""

import jax.numpy as jnp
import jraph
import numpy as np
from scipy.sparse import csr_matrix


def onehot(labels):
  classes = set(labels)
  return jnp.identity(len(classes))[jnp.array(labels)]


def load_from_npz(path, dataset):
  """Loads datasets from npz files."""
  file_name = path + dataset + '.npz'
  with np.load(open(file_name, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)
    adj_matrix = csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    if 'attr_data' in loader:
      # Attributes are stored as a sparse CSR matrix
      attr_matrix = csr_matrix(
          (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
          shape=loader['attr_shape']).todense()
    elif 'attr_matrix' in loader:
      # Attributes are stored as a (dense) np.ndarray
      attr_matrix = loader['attr_matrix']
    else:
      raise Exception('No attributes in the data file', file_name)

    if 'labels_data' in loader:
      # Labels are stored as a CSR matrix
      labels = csr_matrix((loader['labels_data'], loader['labels_indices'],
                           loader['labels_indptr']),
                          shape=loader['labels_shape'])
      labels = labels.nonzero()[1]
    elif 'labels' in loader:
      # Labels are stored as a numpy array
      labels = loader['labels']
    else:
      raise Exception('No labels in the data file', file_name)

  return adj_matrix, attr_matrix, onehot(labels)


def symmetrize(edges):
  """Symmetrizes the adjacency."""
  inv_edges = {(d, s) for s, d in edges}
  return edges.union(inv_edges)


def add_self_loop(edges, n_node):
  """Adds self loop."""
  self_loop_edges = {(s, s) for s in range(n_node)}
  return edges.union(self_loop_edges)


def get_graph_edges(adj, features):
  rows = adj.tocoo().row
  cols = adj.tocoo().col
  edges = {(row, col) for row, col in zip(rows, cols)}
  edges = symmetrize(edges)
  edges = add_self_loop(edges, features.shape[0])
  return edges, len(edges)


def create_jraph(data_path, dataset):
  """Creates a jraph graph for a dataset."""
  adj, features, labels = load_from_npz(data_path, dataset)
  edges, n_edge = get_graph_edges(adj, np.array(features))
  n_node = len(features)
  features = jnp.asarray(features)
  graph = jraph.GraphsTuple(
      n_node=jnp.asarray([n_node]),
      n_edge=jnp.asarray([n_edge]),
      nodes=features,
      edges=None,
      globals=None,
      senders=jnp.asarray([edge[0] for edge in edges]),
      receivers=jnp.asarray([edge[1] for edge in edges]))

  return graph, np.asarray(labels), labels.shape[1]

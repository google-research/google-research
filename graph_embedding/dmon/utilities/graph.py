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

"""Graph construction utility functions.

Functions for graph manipulation and creation.
TODO(tsitsulin): add headers, tests, and improve style.
"""

import pickle
import sys

import networkx as nx
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.base import spmatrix
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf


def construct_knn_graph(data, k = 15, symmetrize = True):
  graph = kneighbors_graph(data, k)
  if symmetrize:
    graph = graph + graph.T
    graph.data = np.ones(graph.data.shape)
  return graph


def normalize_graph(graph,  # pylint: disable=missing-function-docstring
                    normalized,
                    add_self_loops = True):
  if add_self_loops:  # Bröther may i have some self-lööps
    graph = graph + scipy.sparse.identity(graph.shape[0])
  degree = np.squeeze(np.asarray(graph.sum(axis=1)))
  if normalized:
    with np.errstate(divide='ignore'):
      degree = 1. / np.sqrt(degree)
    degree[degree == np.inf] = 0
    degree = scipy.sparse.diags(degree)
    return degree @ graph @ degree
  else:
    with np.errstate(divide='ignore'):
      degree = 1. / degree
    degree[degree == np.inf] = 0
    degree = scipy.sparse.diags(degree)
    return degree @ graph


def scipy_to_tf(matrix):
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)


def load_npz_to_sparse_graph(file_name):  # pylint: disable=missing-function-docstring
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
      label_mask = labels.nonzero()[0]
      labels = labels.nonzero()[1]
    elif 'labels' in loader:
      # Labels are stored as a numpy array
      labels = loader['labels']
      label_mask = np.ones(labels.shape, dtype=np.bool)
    else:
      raise Exception('No labels in the data file', file_name)

  return adj_matrix, attr_matrix, labels, label_mask


def _parse_index_file(filename):
  index = []
  for line in open(filename):
    index.append(int(line.strip()))
  return index


def _sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=np.bool)


def load_kipf_data(path_str, dataset_str):  # pylint: disable=missing-function-docstring
  names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
  objects = []
  for i in range(len(names)):
    with open('{}/ind.{}.{}'.format(path_str, dataset_str, names[i]),
              'rb') as f:
      if sys.version_info > (3, 0):
        objects.append(pickle.load(f, encoding='latin1'))
      else:
        objects.append(pickle.load(f))

  x, y, tx, ty, allx, ally, graph = tuple(objects)  # pylint: disable=unbalanced-tuple-unpacking
  test_idx_reorder = _parse_index_file('{}/ind.{}.test.index'.format(
      path_str, dataset_str))
  test_idx_range = np.sort(test_idx_reorder)

  if dataset_str == 'citeseer':
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(
        min(test_idx_reorder),
        max(test_idx_reorder) + 1)
    tx_extended = lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended

  features = scipy.sparse.vstack((allx, tx)).tolil()
  features[test_idx_reorder, :] = features[test_idx_range, :]
  adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

  labels = np.vstack((ally, ty))
  labels[test_idx_reorder, :] = labels[test_idx_range, :]

  idx_test = test_idx_range.tolist()
  idx_train = range(len(y))
  idx_val = range(len(y), len(y) + 500)

  train_mask = _sample_mask(idx_train, labels.shape[0])
  val_mask = _sample_mask(idx_val, labels.shape[0])
  test_mask = _sample_mask(idx_test, labels.shape[0])

  y_train = np.zeros(labels.shape)
  y_val = np.zeros(labels.shape)
  y_test = np.zeros(labels.shape)
  y_train[train_mask, :] = labels[train_mask, :]
  y_val[val_mask, :] = labels[val_mask, :]
  y_test[test_mask, :] = labels[test_mask, :]

  labels = (y_train + y_val + y_test).nonzero()[1]
  label_mask = (y_train + y_val + y_test).nonzero()[0]

  return adj, features.todense(), labels, label_mask

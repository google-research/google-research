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

"""Data utils for the clustering with strong and weak signals."""

import gin
import numpy as np
from scipy.sparse import csr_matrix


def read_numpy(filename, allow_pickle=False):
  with open(filename, 'rb') as f:
    return np.load(f, allow_pickle=allow_pickle)


def load_weak_and_strong_signals(path, name):
  """Loads weak and strong signals for dataset class."""
  if name in ['stackoverflow', 'search_snippets']:
    features = read_numpy(f'{path}/{name}_weak_signal')
    weak_signal = features.dot(features.T)
    strong_signal = read_numpy(f'{path}/{name}_strong_signal', True)
    strong_signal = strong_signal[()].toarray()
  else:
    weak_signal = read_numpy(f'{path}/{name}_weak_signal')
    strong_signal = read_numpy(f'{path}/{name}_strong_signal')
  return weak_signal, strong_signal


@gin.configurable
class Dataset:
  """Dataset class."""

  def __init__(self, path, name):
    self.get_weak_and_strong_signals(path, name)

  def get_weak_and_strong_signals(self, path, name):
    # Assumption: weak_signal is a NxN numpy array of similarities.
    # Assumption: strong signal is a NxN numpy array of similarities.
    self.weak_signal, self.strong_signal = load_weak_and_strong_signals(
        path, name
    )
    assert self.weak_signal.shape == self.strong_signal.shape
    self.is_graph = False
    self.is_sparse = False

  @property
  def num_examples(self):
    return self.strong_signal.shape[0]

  @property
  def num_pairs(self):
    return (self.num_examples * (self.num_examples - 1)) / 2

  def same_cluster(self, ex_id1, ex_id2):
    return self.strong_signal[ex_id1][ex_id2] == 1

  def pair_same_cluster_iterator(self):
    for ex_id1 in range(self.num_examples):
      for ex_id2 in range(ex_id1 + 1, self.num_examples):
        yield ex_id1, ex_id2, self.same_cluster(ex_id1, ex_id2)

  def most_similar_pairs(self, k):
    """Selects the top_k most similar indices from the weak_signal matrix.

    Consider a similarity matrix as follows:
    [[ 1  2  3  4]
     [ 2  4  5  6]
     [ 3  5  2  3]
     [ 4  6  3  2]]
    And assume we want to select the indices of the top 3 values.
    We first triangualrize the matrix:
    [[0 2 3 4]
     [0 0 5 6]
     [0 0 0 3]
     [0 0 0 0]]
    The ravel function then linearizes the data:
    [ 0 -2 -3 -4  0  0 -5 -6  0  0  0 -3  0  0  0  0]
    The argpartition selects the index of the top 3 elements:
    [3 6 7]
    The unravel function then turns the 1d indexes into 2d indexes:
    [[0 3], [1 2], [1 3]]

    Args:
      k: for top_k

    Returns:
      Indices of the top k elements
    """
    triangular_sims = np.triu(self.weak_signal, k=1)
    idx = np.argpartition(-triangular_sims.ravel(), k)[:k]
    return list(np.column_stack(np.unravel_index(idx, triangular_sims.shape)))

  def argsort_similarities(self, similarities):
    """Arg sorts an array of similarities row-wise in decreasing order."""
    return np.argsort(similarities)[:, ::-1]

  def k_nearest_neighbors_weak_signal(self, input_nodes, k):
    """Returns k nearest neighbors of input nodes according to weak signal."""
    # Weak signal is cosine similarities of embeddings.
    # Higher values indicate closer neighbors.
    cosine_similarities = self.weak_signal[input_nodes, :]
    if self.is_sparse:
      cosine_similarities = cosine_similarities.toarray()
    return self.argsort_similarities(cosine_similarities)[:, 0 : k + 1]

  def symmetrize_graph(self, graph):
    """Makes graph (csr matrix) symmetric."""
    rows, cols = graph.nonzero()
    graph[cols, rows] = graph[rows, cols]
    return graph

  def construct_weak_signal_knn_graph(self, k):
    """Make k-nn graph according to weak_signal similarity matrix."""
    rows, cols = [], []
    for i in range(self.num_examples):
      k_neighbors = self.k_nearest_neighbors_weak_signal([i], k)[0, :]
      for neighbor in k_neighbors:
        if neighbor != i:
          rows.append(i)
          cols.append(neighbor)
    graph = csr_matrix(
        ([1] * len(rows), (rows, cols)),
        shape=(self.num_examples, self.num_examples),
    )
    # Procedure to make graph symmetric.
    graph = self.symmetrize_graph(graph)
    return graph

  def reweight_graph_using_strong_signal(self, possible_edges):
    """Reweight given edges using strong signal."""
    filtered_rows = []
    filtered_columns = []
    for v, u in possible_edges:
      if self.strong_signal[v, u]:
        filtered_rows.append(v)
        filtered_columns.append(u)
    graph = csr_matrix(
        ([1] * len(filtered_rows), (filtered_rows, filtered_columns)),
        shape=(self.num_examples, self.num_examples),
    )
    graph = self.symmetrize_graph(graph)
    return graph

  def construct_weighted_knn_graph(self, k):
    """Get weak signal knn graph and reweight edges using strong signal."""
    weak_signal_knn_graph = self.construct_weak_signal_knn_graph(k)
    rows, columns = weak_signal_knn_graph.nonzero()
    strong_signal_weighted_knn_graph = self.reweight_graph_using_strong_signal(
        zip(rows, columns)
    )
    return strong_signal_weighted_knn_graph


class AdhocDataset:
  """Adhoc dataset class for creating a graph dataset on the fly."""

  def __init__(self, strong_signal, features):
    self.strong_signal = strong_signal
    self.is_graph = True
    self.features = features
    self.num_examples = strong_signal.shape[0]

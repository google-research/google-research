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

"""Functions for precomputing graph properties.

This module implements functions intended to be called before model training,
to compute graph properties needed by some models.
"""

import collections
from typing import Sequence, Tuple
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def wl_coloring(adj_list):
  """Computes the graph coloring used by the Wesifeiler-Lehman algorithm.

  Args:
    adj_list: The graph adjacency represented as a list where the i-th entry
      contains the indices of the neighbors of the i-th node.

  Returns:
    A list where the i-th index gives the color for the i-th node.
  """
  colors = [1] * len(adj_list)
  while True:
    next_colors = [
        hash(
            tuple(sorted(collections.Counter(colors[y] for y in nbrs).items()))
        )
        for nbrs in adj_list
    ]
    color_counts = collections.Counter(next_colors)
    color_sort = sorted(color_counts.keys(), key=lambda x: color_counts[x])
    color_mapping = {x: i for (i, x) in enumerate(color_sort)}
    next_colors = [color_mapping[c] for c in next_colors]
    if next_colors == colors:
      return colors
    colors = next_colors


def all_pairs_shortest_path(
    adj_dict, max_dist
):
  """Computes the length of the shortest path between all pairs of nodes.

  Args:
    adj_dict: The graph adjacency represented as a dict mapping node ids to
      lists of neighbors.
    max_dist: The maximum distance to consider. Pairs with distance further than
      this will not be included in the output.

  Returns:
    A dict mapping node ids to the distances to other nodes in the graph.
    Self-distances are not included.
  """
  distances = {
      u: {v: 1 for v in nbrs if v != u} for u, nbrs in adj_dict.items()
  }
  frontier = {u: set(nbrs.keys()) for u, nbrs in distances.items()}
  for i in range(2, max_dist + 1):
    next_frontier = collections.defaultdict(set)
    for u, ancestors in frontier.items():
      u_dist = distances[u]
      for v in ancestors:
        for nbr in adj_dict[v]:
          if nbr == u:
            continue
          if nbr not in u_dist:
            u_dist[nbr] = i
            next_frontier[u].add(nbr)
    frontier = next_frontier
    if not frontier:
      break
  return distances


def edge_list_to_adj_matrix(
    edge_list
):
  """Converts a graph represented as a dict to an adjacency matrix.

  Args:
    edge_list: The graph represented as a list of edges, as pairs of node ids.

  Returns:
    The graph as an adjacency matrix.
  """
  rows = []
  cols = []
  for src, targ in edge_list:
    rows.append(src)
    cols.append(targ)
    rows.append(targ)
    cols.append(src)
  data = np.ones_like(rows, dtype=np.float32)
  return sp.coo_array((data, (rows, cols)))


def graph_bert_intimacy(
    adj_matrix, alpha = 0.15
):
  """Computes the initimacy scores associated with a given adjacency matrix.

  Uses the definition from the GraphBert paper:
    S = alpha (1 - (1-alpha) A D^-1)^-1

  Args:
    adj_matrix: Adjacency matrix for a graph (from adj_dict_to_matrix)
    alpha: The teleportation parameter from PageRank

  Returns:
    A dict mapping node names to PageRank values.
  """
  s = alpha * np.linalg.inv(
      np.eye(adj_matrix.shape[0])
      - (1 - alpha) * adjacency_to_stochastic(adj_matrix)
  )
  return s


def adjacency_to_stochastic(
    adj_matrix,
):
  """Computes the stochastic transition matrix from a graph adjacency."""
  norms = 1.0 / adj_matrix.sum(axis=0).flatten()
  stochastic_matrix = adj_matrix
  selected_norms = np.take(norms, stochastic_matrix.row).ravel()
  stochastic_matrix.data *= selected_norms
  return stochastic_matrix


def graph_spectrum(adj_matrix, dim):
  """Gives a matrix of the top k eigenvectors of the normalized graph laplacian.

  Assumes that the adjecency matrix is symmetric.

  Args:
    adj_matrix: Adjacency matrix for a graph (from adj_dict_to_matrix)
    dim: The number of eigenvectors to compute / the dimensionality of the per
      node 'embeddings'

  Returns:
    An N x dim matrix of containing the top `dim` eigenvectors.
  """
  laplacian = sp.eye(adj_matrix.shape[0]) - adjacency_to_stochastic(adj_matrix)
  _, eigvecs = sp.linalg.eigsh(laplacian, dim)
  return eigvecs


def get_sinusoidal_encoding(position, dim):
  angles = [position * np.power(10000.0, -2 * i / dim) for i in range(dim)]
  return np.array(
      [np.sin(a) if i % 2 == 0 else np.cos(a) for i, a in enumerate(angles)]
  )


def get_sinusoidal_encoding_tensor(positions, dim):
  return tf.constant([get_sinusoidal_encoding(p, dim) for p in positions])

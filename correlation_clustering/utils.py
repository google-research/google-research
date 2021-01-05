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

"""Utils for the fair correlation clustering algorithm.
"""

import collections
import math
import numpy as np
import sklearn.metrics


def BooleanVectorsFromGraph(graph):
  """Create a boolean encoding for the nodes in the graph.

  Starting from the graph, it creates a set of boolean vectors where u,v,
  has an entry 1 for each positive edge (0 for negative edge). Selfloops
  are assumed positive.

  Args:
    graph: graph in nx.Graph format.
  Returns:
    the nxn bolean matrix with the encoding.
  """
  n = graph.number_of_nodes()
  vectors = np.identity(n)
  for u, v, d in graph.edges(data=True):
    if d['weight'] > 0:
      vectors[u][v] = 1
      vectors[v][u] = 1
  return vectors


def PairwiseFairletCosts(graph):
  """Create a matrix with the fairlet cost.

  Args:
    graph: graph in nx.Graph format.
  Returns:
    the nxn matrix with the fairlet cost for each pair of nodes.
  """
  assert max(list(graph.nodes())) == graph.number_of_nodes() - 1
  assert min(list(graph.nodes())) == 0

  bool_vectors = BooleanVectorsFromGraph(graph)
  distance_matrix = sklearn.metrics.pairwise_distances(
      bool_vectors, metric='l1')
  # This counts twice the negative edge inside each u,v fairlet, so we deduct
  # one for each such pair.
  for u, v, d in graph.edges(data=True):
    if d['weight'] < 0:
      distance_matrix[u][v] -= 1
      distance_matrix[v][u] -= 1
  return distance_matrix


def ClusterIdMap(solution):
  """Create a map from node to cluster id.

  Args:
    solution: list of clusters.
  Returns:
    the map from node id to cluster id.
  """
  clust_assignment = {}
  for i, clust in enumerate(solution):
    for elem in clust:
      clust_assignment[elem] = i
  return clust_assignment


def FractionalColorImbalance(graph, solution, alpha):
  """Evaluates the color imbalance of solution.

  Computes the fraction of nodes that are above the threshold for color
  representation.

  Args:
    graph: in nx.Graph format.
    solution: list of clusters.
    alpha: representation constraint.
  Returns:
    the fraction of nodes that are above the threshold for color.
  """
  total_violation = 0
  nodes = 0
  for cluster in solution:
    color_count = collections.defaultdict(int)
    for elem in cluster:
      color_count[graph.nodes[elem]['color']] += 1
    for count in color_count.values():
      imbalance = max(0, count - math.floor(float(len(cluster)) * alpha))
      total_violation += imbalance
    nodes += len(cluster)
  return 1.0 * total_violation / nodes


def CorrelationClusteringError(graph, solution):
  """Evaluates  the correlation clustering error of solution.

  Computes the fraction of edges that are misclassified by the algorithm.

  Args:
    graph: in nx.Graph format.
    solution: list of clusters.
  Returns:
    the fraction of edges that are incorrectly classified.
  """
  clust_assignment = ClusterIdMap(solution)
  errors = 0
  corrects = 0
  for u, v, d in graph.edges(data=True):
    if (d['weight'] > 0 and clust_assignment[u] != clust_assignment[v]) or \
        (d['weight'] < 0 and clust_assignment[u] == clust_assignment[v]):
      errors += 1
    elif (d['weight'] > 0 and clust_assignment[u] == clust_assignment[v]) or \
        (d['weight'] < 0 and clust_assignment[u] != clust_assignment[v]):
      corrects += 1
  return float(errors) / (errors + corrects)

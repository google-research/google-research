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

"""Implementations of standard correlation clustering algorithm.
"""

import collections
import random
from .utils import CorrelationClusteringError


def PivotAlgorithm(graph):
  """The well-known pivot algorithm for correlation clustering.

  Run the pivot algorithm on graph.
  Args:
    graph: the graph in nx.Graph format.
  Returns:
    The solution.
  """
  # This is to ensure consistency of random runs with same seed.
  nodes = sorted(list(graph.nodes()))
  random.shuffle(nodes)
  clusters = []
  clustered = set()

  for node in nodes:
    if node in clustered:
      continue
    cluster = [node]
    clustered.add(node)

    for neighbor in graph.neighbors(node):
      if graph.edges[node,
                     neighbor]['weight'] > 0 and neighbor not in clustered:
        cluster.append(neighbor)
        clustered.add(neighbor)
    clusters.append(cluster)
  assert len(clustered) == sum(len(c) for c in clusters)
  assert clustered == set(nodes)
  return clusters


def LocalSearchAlgorithm(graph, attempts=10):
  """Run the local search heuristic for correlation clustering.

  The algorithm is a simple local search heuristic that tries to improve the
  clustering by local moves of individual nodes until a certain number of
  iterations over the graph are completed.

  Args:
    graph: the graph in nx.Graph format.
    attempts: number of times local search is run.
  Returns:
    The solution.
  """
  best_sol = None
  best_sol_value = None
  for _ in range(attempts):
    ls = LocalSearchCorrelationClustering(graph, 20)
    sol = ls.RunClustering()
    cost = CorrelationClusteringError(graph, sol)
    if best_sol_value is None or best_sol_value > cost:
      best_sol_value = cost
      best_sol = sol
  return best_sol


class LocalSearchCorrelationClustering(object):
  """Single run of the the local search heuristic for correlation clustering.

  The algorithm performs a series of passes over the nodes in the graph in
  arbitrary order.
  For each node in the order, it checks if the solution can be improved by
  moving the node to another cluster.
  """

  def __init__(self, graph, iterations):
    self.graph = graph
    self.iterations = iterations
    self.node_to_cluster_id = {}
    self.cluster_uid = 0
    self.cluster_id_nodes = collections.defaultdict(set)
    for node in self.graph.nodes():
      self.node_to_cluster_id[node] = self.cluster_uid
      self.cluster_id_nodes[self.cluster_uid].add(node)
      self.cluster_uid += 1

  def MoveNodeToCluster(self, node, cluster_id):
    """Moves a node to a cluster."""
    self.cluster_id_nodes[self.node_to_cluster_id[node]].remove(node)
    self.node_to_cluster_id[node] = cluster_id
    self.cluster_id_nodes[cluster_id].add(node)

  def DoOnePassMoves(self):
    """Completes one pass over the graph."""
    nodes = sorted(list(self.graph.nodes()))
    random.shuffle(nodes)
    for node in nodes:
      positive_to_clusters = collections.defaultdict(int)
      best_cluster = None
      best_cluster_cost = self.graph.number_of_nodes() + 1
      positives = 0
      for neighbor in self.graph.neighbors(node):
        if self.graph.edges[node, neighbor]['weight'] > 0:
          positives += 1
          positive_to_clusters[self.node_to_cluster_id[neighbor]] += 1
      curr_cluster = self.node_to_cluster_id[node]
      curr_cluster_cost = positives + len(
          self.cluster_id_nodes[curr_cluster]
          )-1-2*positive_to_clusters[curr_cluster]
      for c, pos in positive_to_clusters.items():
        if c != curr_cluster:
          cluster_cost = positives + len(self.cluster_id_nodes[c]) - 2*pos
          if cluster_cost < best_cluster_cost:
            best_cluster_cost = cluster_cost
            best_cluster = c

      if best_cluster_cost < curr_cluster_cost:
        self.MoveNodeToCluster(node, best_cluster)

  def RunClustering(self):
    for _ in range(self.iterations):
      self.DoOnePassMoves()
    return self.cluster_id_nodes.values()

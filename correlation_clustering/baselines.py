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

"""Implementations of fair and standard correlation clustering baselines."""
import collections
import random
import networkx as nx


def BaselineAllTogether(graph):
  """Trivial baseline consisting in outputing a single cluster.

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'.

  Returns:
    A list of lists represeting the clusters.
  """
  return [list(graph.nodes())]


def BaselineRandomFairEqual(graph):
  """Fair baseline of random size c clusters with one point per color.

  The clusters have alpha = 1/C balance (equal color representation).

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'.

  Returns:
    A list of lists represeting the clusters.
  """
  # Create a mapping from color to nodes.
  color_nodes = collections.defaultdict(list)
  for u, d in graph.nodes(data=True):
    color_nodes[d['color']].append(u)
  nodes_by_color = [nodes for nodes in color_nodes.values()]
  nodes_per_color = len(nodes_by_color[0])
  for l in nodes_by_color:
    random.shuffle(l)

  clusters = []
  for _ in range(nodes_per_color):
    cluster = []
    for i in range(len(nodes_by_color)):
      cluster.append(nodes_by_color[i].pop())
    clusters.append(cluster)

  return clusters


def BaselineRandomFairOneHalf(graph):
  """Fair baseline which consists in random size 2 clusters.

  The clusters have alpha = 1/2 balance.

  Args:
    graph: The undirected graph represented nx.Graph. Nodes must have the
      attribute 'color'

  Returns:
    A list of lists represeting the clusters.
  """
  # To get a random unifom matching we use max weight max cardinatlity matching
  # over a randomly weighted graph.
  matching_graph = nx.Graph()
  for i in range(graph.number_of_nodes()):
    for j in range(i + 1, graph.number_of_nodes()):
      if graph.nodes[i]['color'] != graph.nodes[j]['color']:
        matching_graph.add_edge(i, j, weight=random.random())
  matching = nx.max_weight_matching(matching_graph, maxcardinality=True)
  return [list(m) for m in matching]

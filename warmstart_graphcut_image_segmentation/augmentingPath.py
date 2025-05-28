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

"""This file implements the cold-start FF algorithm with shortest augmenting path.

The augmenting path is found by running BFS each time. After FF terminates,
DFS is then used to find the cuts.
This file is adapted from "augmentingPath.py" in the github repo
"image-segmentation".
The functions in this file are re-implemented from the functions with the same
name from the original file.
"""

from collections import defaultdict
from copy import deepcopy
from queue import Queue
import numpy as np
"""
Re-implemented from bfs() in the original "augmentingPath.py" but slightly changed to adapt to the data type of some_graph.
In the original implementation the input graph is a matrix. However this does not take advantage of the sparsity of the graph.
We reimplemented the algorithm and made the input graph a dictionary of {node:adjacency list} instead,
where each adjacency list is another default dictionary of {neighbor: arc capacity on (node, neighbor)}.
"""


def bfs(some_graph, V, s, t, parent):
  q = Queue()
  visited = [False] * V
  q.put(s)
  visited[s] = True
  parent[s] = -1

  while not q.empty():
    u = q.get()

    for v in some_graph[u]:
      if (not visited[v]) and some_graph[u][v] > 0:
        q.put(v)
        parent[v] = u
        visited[v] = True
        if v == t:
          return True
  return False


"""
Re-implemented from the dfs() in the original "augmentingPath.py".
"""


def dfs(some_graph, V, s):
  visited = [False] * V
  stack = [s]
  while stack:
    v = stack.pop()
    if not visited[v]:
      visited[v] = True
      stack.extend([u for u in some_graph[v] if some_graph[v][u] > 0])
  return visited


"""
Similar to augmentingPath() in the original "augmentingPath.py" with changes.
flow_value and rGraph: can potentially be used to warmstart.
    - If None, flows is initialized to be all 0's on the arcs, and rGraph is equivalent to the capacities on each arc.
    - Otherwises, start finding augmenting paths and increase the flow based on the given flow and rGraph.
Must be feasible, satisfying both capacity and flow conservation constraints.
If None, it means start from scratch.
"""


def augmentingPath(graph, V, s, t, flows=None, rGraph=None):
  print("Running augmenting path algorithm")
  path_counter = 0
  total_path_len = 0
  # this keeps track of the flow value on each edge
  # flow_value = np.zeros((V, V), dtype='int32')
  if flows is None:
    flows = {i: defaultdict(int) for i in range(V)}
    rGraph = deepcopy(graph)

  flow = sum([flows[s][j] for j in flows[s]])

  parent = np.zeros(V, dtype="int32")
  while bfs(rGraph, V, s, t, parent):
    path_counter += 1
    pathFlow = float("inf")
    v = t
    while v != s:
      u = parent[v]
      total_path_len += 1
      pathFlow = min(pathFlow, rGraph[u][v])
      v = parent[v]
    flow += pathFlow
    v = t
    while v != s:
      u = parent[v]
      rGraph[u][v] -= pathFlow
      rGraph[v][u] += pathFlow
      flows[u][v] += max(0, pathFlow - flows[v][u])
      flows[v][u] = max(0, flows[v][u] - pathFlow)
      v = parent[v]

  visited = dfs(rGraph, V, s)
  cuts = []

  for i in range(V):
    for j in graph[i]:
      if visited[i] and not visited[j] and graph[i][j] > 0:
        cuts.append((i, j))

  if path_counter > 0:
    average_aug_path_len = float(total_path_len) / path_counter
  else:
    average_aug_path_len = 0
  return flows, cuts, path_counter, average_aug_path_len

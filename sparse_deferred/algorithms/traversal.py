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

"""Algorithms for traversing graphs, written with tensor ops, for GPU/TPU."""

from sparse_deferred.implicit import matrix

Matrix = matrix.Matrix
Tensor = matrix.Tensor


def one_hop_reachable(
    directed_adjacency,
    start_nodes
    ):
  """Returns nodes that are exactly one step away from any of `start_nodes`.

  Args:
    directed_adjacency: Directed square adjacency matrix.
    start_nodes: boolean tensor with
      `start_nodes.shape[0] == directed_adjacency.shape[0]`
      (` == directed_adjacency.shape[1]`, as it is must be square)

  Returns:
    boolean tensor with same shape as `start_nodes`. If `start_nodes is a
    vector, then the return will be vector marking every nodes with True if it
    is one hop away from any node that is marked `True` in `start_nodes`.
  """
  start_nodes = directed_adjacency.engine.cast(start_nodes, 'int16')
  return directed_adjacency @ start_nodes != 0


def multi_hop_reachable(
    directed_adjacency,
    start_nodes,
    hops = 1,
    include_transpose = False,
    ):
  """Calculates nodes that are `<= hops` away from `start_nodes`.

  The `<=` indicates that all nodes touched on traversal, including
  `start_nodes`, will be marked as "reached". NOTE: Replacing line
  `nodes = engine.reduce_any(reachable, axis=0)` with `nodes = reachable[-1]`
  makes the function return list of nodes *exactly* at distance `hops`.

  Args:
    directed_adjacency: Square adjacency matrix to traverse. See also
      `include_transpose`.
    start_nodes: Must be boolean tensor that indicates the start nodes of
      traversal. Leading dimension must be directed_adjacency.shape[0].
    hops: Numer of hops.
    include_transpose: If set, the effective adjacency matrix is elementwise-or
      of input `directed_adjacency` and its transpose.

  Returns:
    Boolean tensor marking nodes reachable from `start_nodes` within `hops`
    according to `directed_adjacency`  (and optionally, its transpose).
  """
  adjacencies = [directed_adjacency]
  if include_transpose:
    adjacencies.append(directed_adjacency.T)
  engine = adjacencies[0].engine
  nodes = start_nodes
  for _ in range(hops):
    reachable = [one_hop_reachable(adj, nodes) for adj in adjacencies]
    reachable.append(nodes)
    nodes = engine.reduce_any(reachable, axis=0)
  return nodes

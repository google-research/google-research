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

"""Implements topological order for directed acyclic graphs."""

from sparse_deferred.implicit import matrix


def topological_order(directed_adjacency, max_depth = 20,
                      large = 9999, early_stop = False
                      ):
  """Finds topological of nodes given a binary (square) adjacency matrix.

  The adjacency must be square, binary, and form a DAG. Otherwise, the function
  will produce throwaway results.

  Args:
    directed_adjacency: Must have shape (n, n) where n is number of nodes. It
      must encode a directed acyclic graph (DAG).
    max_depth: The DAG's depth is assumed this much. All nodes beyond this depth
      will receive an order of `large`.
    large: Large number.
    early_stop: If set, the computation will stop **as soon as all nodes are
      ordered**. However, this produces instructions that are non-TPU friendly.
      Please use if you are processing data on host CPU.

  Returns:
    Vector of size n where entry `i` indicates the topological order of node `i`
  """
  adj = directed_adjacency  # For short
  in_degree = adj.engine.cast(adj.rowsums(), 'int32')  # Assuming binary adj.
  order = adj.engine.ones_like(in_degree) * large  # Vec of all large numbers.

  for i in range(max_depth):
    is_ready = in_degree == 0
    i_if_ready = adj.engine.cast(is_ready, 'int32') * i
    large_if_not_ready = adj.engine.cast(in_degree > 0, 'int32') * large
    total = i_if_ready + large_if_not_ready
    ready_just_now = (adj.engine.cast(total == i, 'float32') *
                      adj.engine.cast(order == large, 'float32'))
    processed_neighbors = adj @ ready_just_now
    in_degree -= adj.engine.cast(processed_neighbors, 'int32')
    order = adj.engine.minimum(order, total)
    if early_stop:
      if adj.engine.to_cpu(adj.engine.all(is_ready)):
        break

  return order


def argsort_topological_order(
    directed_adjacency, max_depth = 20,
    large = 9999, early_stop = False,
    direction = 'ASCENDING'):
  """Returns `argsort(topological_order(.))`."""
  return directed_adjacency.engine.argsort(topological_order(
      directed_adjacency, max_depth, large, early_stop), direction=direction)

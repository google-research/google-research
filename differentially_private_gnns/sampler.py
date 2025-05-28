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

"""Implements sampling with in-degree constraints."""

from typing import Dict, Sequence, Union

import chex
import jax
import numpy as np

from differentially_private_gnns import dataset_readers


Node = Union[int, str]
AdjacencyDict = Dict[Node, Sequence[Node]]


def reverse_edges(edges):
  """Reverses an edgelist to obtain incoming edges for each node."""
  reversed_edges = {u: [] for u in edges}
  for u, u_neighbors in edges.items():
    for v in u_neighbors:
      reversed_edges[v].append(u)
  return reversed_edges


def get_adjacency_lists(dataset):
  """Returns a dictionary of adjacency lists, with nodes as keys."""
  if len(dataset.senders) != len(dataset.receivers):
    raise ValueError('Senders and receivers should be of the same length.')

  edges = {u: [] for u in range(dataset.num_nodes())}
  for u, v in zip(dataset.senders, dataset.receivers):
    edges[u].append(v)
  return edges


def sample_adjacency_lists(edges, train_nodes,
                           max_degree, rng):
  """Statelessly samples the adjacency lists with in-degree constraints.

  This implementation performs Bernoulli sampling over edges.

  Note that the degree constraint only applies to training subgraphs.
  The validation and test subgraphs are sampled completely.

  Args:
    edges: The adjacency lists to sample.
    train_nodes: A sequence of train nodes.
    max_degree: The bound on in-degree for any node over training subgraphs.
    rng: The PRNGKey for reproducibility
  Returns:
    A sampled adjacency list, indexed by nodes.
  """
  train_nodes = set(train_nodes)
  all_nodes = edges.keys()

  reversed_edges = reverse_edges(edges)
  sampled_reversed_edges = {u: [] for u in all_nodes}

  # For every node, bound the number of incoming edges from training nodes.
  dropped_count = 0
  for u in all_nodes:
    u_rng = jax.random.fold_in(rng, u)
    incoming_edges = reversed_edges[u]
    incoming_train_edges = [v for v in incoming_edges if v in train_nodes]
    if not incoming_train_edges:
      continue

    in_degree = len(incoming_train_edges)
    sampling_prob = max_degree / (2 * in_degree)
    sampling_mask = (
        jax.random.uniform(u_rng, shape=(in_degree,)) <= sampling_prob)
    sampling_mask = np.asarray(sampling_mask)

    incoming_train_edges = np.asarray(incoming_train_edges)[sampling_mask]
    unique_incoming_train_edges = np.unique(incoming_train_edges)

    # Check that in-degree is bounded, otherwise drop this node.
    if len(unique_incoming_train_edges) <= max_degree:
      sampled_reversed_edges[u] = unique_incoming_train_edges.tolist()
    else:
      dropped_count += 1

  print('dropped count', dropped_count)
  sampled_edges = reverse_edges(sampled_reversed_edges)

  # For non-train nodes, we can sample the entire edgelist.
  for u in all_nodes:
    if u not in train_nodes:
      sampled_edges[u] = edges[u]
  return sampled_edges

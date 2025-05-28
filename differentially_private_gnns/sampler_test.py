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

"""Tests for sampler."""

from typing import Any, Callable, Dict, List, Union

from absl.testing import absltest
from absl.testing import parameterized
import jax
import networkx as nx
import numpy as np

from differentially_private_gnns import sampler

Subgraphs = Dict[int, Union[List[int], 'Subgraphs']]


def sample_subgraphs(edges, num_hops):
  """Samples subgraphs given an edgelist."""

  if num_hops not in [1, 2]:
    raise NotImplementedError

  if num_hops == 1:
    return edges

  subgraphs = {}
  for root_node, neighbors in edges.items():
    subgraphs[root_node] = {}
    for neighbor in neighbors:
      subgraphs[root_node][neighbor] = edges[neighbor]
  return subgraphs


def flatten_subgraphs(subgraphs):
  """Flattens sampled subgraphs."""

  def flatten_subgraph(
      node, node_subgraph):
    # Base case.
    if isinstance(node_subgraph, list):
      return [node, *node_subgraph]

    # Recurse on neighbours.
    flattened = []
    for neighbor, neighbor_subgraph in node_subgraph.items():
      flattened.extend(flatten_subgraph(neighbor, neighbor_subgraph))
    return flattened

  return {
      node: flatten_subgraph(node, node_subgraph)
      for node, node_subgraph in subgraphs.items()
  }


class SampleEdgelistTest(parameterized.TestCase):

  @parameterized.product(
      rng_key=[0, 1],
      sample_fn=[sampler.sample_adjacency_lists],
      num_nodes=[10, 20, 50],
      edge_probability=[0.1, 0.2, 0.5, 0.8, 1.],
      max_degree=[1, 2, 5, 10, 20])
  def test_occurrence_constraints_one_hop(
      self, rng_key, sample_fn,
      num_nodes, edge_probability, max_degree):

    graph = nx.erdos_renyi_graph(num_nodes, p=edge_probability)
    edges = {node: list(graph.neighbors(node)) for node in graph.nodes}
    train_nodes = set(np.arange(num_nodes, step=2).flat)
    rng = jax.random.PRNGKey(rng_key)
    sampled_edges = sample_fn(edges, train_nodes, max_degree, rng)
    sampled_subgraphs = sample_subgraphs(sampled_edges, num_hops=1)
    sampled_subgraphs = flatten_subgraphs(sampled_subgraphs)

    occurrence_counts = {node: 0 for node in sampled_edges}
    for root_node, subgraph in sampled_subgraphs.items():
      if root_node in train_nodes:
        for node in subgraph:
          occurrence_counts[node] += 1

    self.assertLen(sampled_edges, num_nodes)
    self.assertLen(sampled_subgraphs, num_nodes)
    for count in occurrence_counts.values():
      self.assertLessEqual(count, max_degree + 1)

  @parameterized.product(
      rng_key=[0, 1],
      sample_fn=[sampler.sample_adjacency_lists],
      num_nodes=[10, 20, 50],
      edge_probability=[0.1, 0.2, 0.5, 0.8, 1.],
      max_degree=[1, 2, 5, 10, 20])
  def test_occurrence_constraints_two_hop_disjoint(
      self, rng_key, sample_fn,
      num_nodes, edge_probability, max_degree):

    num_train_nodes = num_nodes // 2
    graph = nx.disjoint_union(
        nx.erdos_renyi_graph(num_train_nodes, p=edge_probability),
        nx.erdos_renyi_graph(num_nodes - num_train_nodes, p=edge_probability))
    edges = {node: list(graph.neighbors(node)) for node in graph.nodes}
    train_nodes = set(np.arange(num_train_nodes).flat)
    rng = jax.random.PRNGKey(rng_key)
    sampled_edges = sample_fn(edges, train_nodes, max_degree, rng)
    sampled_subgraphs = sample_subgraphs(sampled_edges, num_hops=2)
    sampled_subgraphs = flatten_subgraphs(sampled_subgraphs)

    occurrence_counts = {node: 0 for node in sampled_edges}
    for root_node, subgraph in sampled_subgraphs.items():
      if root_node in train_nodes:
        for node in subgraph:
          occurrence_counts[node] += 1

    self.assertLen(sampled_edges, num_nodes)
    self.assertLen(sampled_subgraphs, num_nodes)
    for count in occurrence_counts.values():
      self.assertLessEqual(count, max_degree * max_degree + max_degree + 1)


if __name__ == '__main__':
  absltest.main()

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

"""Input pipeline for DP-GNN training."""

from typing import Dict, Tuple

import chex
import jraph
import ml_collections
import numpy as np

from differentially_private_gnns import dataset_readers
from differentially_private_gnns import normalizations
from differentially_private_gnns import sampler


def add_reverse_edges(
    graph):
  """Add reverse edges to the graph."""
  senders = np.concatenate(
      (graph.senders, graph.receivers))
  receivers = np.concatenate(
      (graph.receivers, graph.senders))

  graph.senders = senders
  graph.receivers = receivers
  return graph


def subsample_graph(graph, max_degree,
                    rng):
  """Subsamples the undirected input graph."""
  edges = sampler.get_adjacency_lists(graph)
  edges = sampler.sample_adjacency_lists(edges, graph.train_nodes, max_degree,
                                         rng)
  senders = []
  receivers = []
  for u in edges:
    for v in edges[u]:
      senders.append(u)
      receivers.append(v)

  graph.senders = senders
  graph.receivers = receivers
  return graph


def compute_masks_for_splits(
    graph):
  """Compute boolean masks for the train, validation and test splits."""
  masks = {}
  num_nodes = graph.num_nodes()
  for split, split_nodes in zip(
      ['train', 'validation', 'test'],
      [graph.train_nodes, graph.validation_nodes, graph.test_nodes]):
    split_mask = np.zeros(num_nodes, dtype=bool)
    split_mask[split_nodes] = True
    masks[split] = split_mask
  return masks


def convert_to_graphstuple(
    graph):
  """Converts a dataset to one entire jraph.GraphsTuple, extracting labels."""
  return jraph.GraphsTuple(  # pytype: disable=wrong-arg-types  # jax-ndarray
      nodes=np.asarray(graph.node_features),
      edges=np.ones_like(graph.senders),
      senders=np.asarray(graph.senders),
      receivers=np.asarray(graph.receivers),
      globals=np.zeros(1),
      n_node=np.asarray([graph.num_nodes()]),
      n_edge=np.asarray([graph.num_edges()]),
  ), np.asarray(graph.node_labels)


def add_self_loops(graph):
  """Adds self-loops to the graph."""
  num_nodes = normalizations.compute_num_nodes(graph)
  senders = np.concatenate(
      (np.arange(num_nodes), np.asarray(graph.senders, dtype=np.int32)))
  receivers = np.concatenate(
      (np.arange(num_nodes), np.asarray(graph.receivers, dtype=np.int32)))

  return graph._replace(
      senders=senders,
      receivers=receivers,
      edges=np.ones_like(senders),
      n_edge=np.asarray([senders.shape[0]]))


def get_dataset(
    config,
    rng,
):
  """Load graph dataset."""
  graph = dataset_readers.get_dataset(config.dataset, config.dataset_path)
  graph = add_reverse_edges(graph)
  graph = subsample_graph(graph, config.max_degree, rng)
  masks = compute_masks_for_splits(graph)
  graph, labels = convert_to_graphstuple(graph)
  graph = add_self_loops(graph)
  graph = normalizations.normalize_edges_with_mask(
      graph, mask=None, adjacency_normalization=config.adjacency_normalization)
  return graph, labels, masks

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

"""Defines edge weight (adjacency) normalization methods."""

from typing import Optional
import chex
import jax
import jax.numpy as jnp
import jraph


def compute_num_nodes(graph):
  """Computes the number of nodes in the graph, statically if possible."""
  if graph.nodes is not None:
    return jax.tree.leaves(graph.nodes)[0].shape[0]

  return jnp.sum(graph.n_node)


def compute_num_edges(graph):
  """Computes the number of edges in the graph, statically if possible."""
  if graph.edges is not None:
    return jax.tree.leaves(graph.edges)[0].shape[0]

  return jnp.sum(graph.n_edge)


def zero_out_with_mask(arr, mask):
  """Zeros out an array where mask is False."""
  return jnp.where(mask, arr, 0.)


def masked_no_normalization(graph,
                            mask):
  """Returns a weight of 1 for each edge."""
  num_edges = compute_num_edges(graph)
  edges = jnp.ones((num_edges,), dtype=jnp.float32)
  edges = zero_out_with_mask(edges, mask)
  return edges[:, jnp.newaxis]


def masked_inverse_degree_normalization(graph,
                                        mask):
  """Returns weights for each edge corresponding to the normalization defined by D^-1 A."""
  num_nodes = compute_num_nodes(graph)
  senders = graph.senders
  sender_degree = jraph.segment_sum(  # pytype: disable=wrong-arg-types  # numpy-scalars
      mask.astype(jnp.int32), senders, num_nodes)
  sender_coeffs = 1 / jnp.maximum(sender_degree, 1.)
  edges = sender_coeffs[senders]
  edges = zero_out_with_mask(edges, mask)
  return edges[:, jnp.newaxis]


def masked_inverse_sqrt_degree_normalization(graph,
                                             mask):
  """Returns weights for each edge corresponding to the normalization defined by D^(-1/2) A D^(-1/2)."""
  num_nodes = compute_num_nodes(graph)
  senders = graph.senders
  receivers = graph.receivers
  sender_degree = jraph.segment_sum(  # pytype: disable=wrong-arg-types  # numpy-scalars
      mask.astype(jnp.int32), senders, num_nodes)
  receiver_degree = jraph.segment_sum(  # pytype: disable=wrong-arg-types  # numpy-scalars
      mask.astype(jnp.int32), receivers, num_nodes)
  sender_coeffs = 1 / jnp.sqrt(jnp.maximum(sender_degree, 1.))
  receiver_coeffs = 1 / jnp.sqrt(jnp.maximum(receiver_degree, 1.))
  edges = sender_coeffs[senders] * receiver_coeffs[receivers]
  edges = zero_out_with_mask(edges, mask)
  return edges[:, jnp.newaxis]


def normalize_edges_with_mask(
    graph,
    mask,
    adjacency_normalization):
  """Normalizes edge weights with a boolean mask indicating valid edges."""
  if mask is None:
    num_edges = compute_num_edges(graph)
    mask = jnp.ones(num_edges, dtype=bool)

  if adjacency_normalization is None:
    normalized_edges = masked_no_normalization(graph, mask)
  if adjacency_normalization == 'inverse-degree':
    normalized_edges = masked_inverse_degree_normalization(graph, mask)
  if adjacency_normalization == 'inverse-sqrt-degree':
    normalized_edges = masked_inverse_sqrt_degree_normalization(graph, mask)
  return graph._replace(edges=normalized_edges)

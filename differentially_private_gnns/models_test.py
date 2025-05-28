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

"""Tests for models."""

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jraph
import numpy as np

from differentially_private_gnns import models
from differentially_private_gnns import normalizations


def get_dummy_graph(
    add_self_loops,
    symmetrize_edges,
    adjacency_normalization):
  """Returns a small dummy GraphsTuple."""
  senders = np.array([0, 2])
  receivers = np.array([1, 1])
  num_edges = len(senders)
  num_nodes = 3
  node_features = np.array([[2.], [1.], [1.]], dtype=np.float32)

  if symmetrize_edges:
    new_senders = np.concatenate([senders, receivers], axis=0)
    new_receivers = np.concatenate([receivers, senders], axis=0)
    senders, receivers = new_senders, new_receivers
    num_edges *= 2

  if add_self_loops:
    senders = np.concatenate([senders, np.arange(num_nodes)], axis=0)
    receivers = np.concatenate([receivers, np.arange(num_nodes)], axis=0)
    num_edges += num_nodes

  dummy_graph = jraph.GraphsTuple(
      n_node=np.asarray([num_nodes]),
      n_edge=np.asarray([num_edges]),
      senders=senders,
      receivers=receivers,
      nodes=node_features,
      edges=np.ones((num_edges, 1)),
      globals=np.zeros((1, 1)),
  )

  return normalizations.normalize_edges_with_mask(
      dummy_graph, mask=None, adjacency_normalization=adjacency_normalization)


def get_adjacency_matrix(graph):
  """Returns a dense adjacency matrix for the given graph."""
  # Initialize the adjacency matrix as all zeros.
  num_nodes = graph.n_node[0]
  adj = np.zeros((num_nodes, num_nodes))

  # Add edges, indicated by a 1 in the corresponding row and column.
  for u, v in zip(graph.senders, graph.receivers):
    adj[u][v] = 1

  return adj


def normalize_adjacency(
    adj,
    adjacency_normalization):
  """Performs appropriate normalization of the given adjacency matrix."""
  if adjacency_normalization is None:
    return adj
  if adjacency_normalization == 'inverse-sqrt-degree':
    sender_degrees = np.sum(adj, axis=1)
    sender_degrees = np.maximum(sender_degrees, 1.)
    inv_sqrt_sender_degrees = np.diag(
        1 / np.sqrt(sender_degrees))
    receiver_degrees = np.sum(adj, axis=0)
    receiver_degrees = np.maximum(receiver_degrees, 1.)
    inv_sqrt_receiver_degrees = np.diag(
        1 / np.sqrt(receiver_degrees))
    return inv_sqrt_sender_degrees @ adj @ inv_sqrt_receiver_degrees
  if adjacency_normalization == 'inverse-degree':
    sender_degrees = np.sum(adj, axis=1)
    inv_sender_degrees = np.diag(1 / np.maximum(sender_degrees, 1.))
    return inv_sender_degrees @ adj
  raise ValueError(f'Unsupported normalization {adjacency_normalization}.')


class ModelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='inverse-degree-without-self-loops',
          add_self_loops=False,
          adjacency_normalization='inverse-degree'),
      dict(
          testcase_name='inverse-sqrt-degree-without-self-loops',
          add_self_loops=False,
          adjacency_normalization='inverse-sqrt-degree'),
      dict(
          testcase_name='no-normalization-symmetrize',
          adjacency_normalization=None,
          symmetrize_edges=True),
      dict(
          testcase_name='no-normalization-no-symmetrize',
          adjacency_normalization=None,
          symmetrize_edges=False),
      dict(
          testcase_name='inv-sqrt-degree-normalization-symmetrize',
          adjacency_normalization='inverse-sqrt-degree',
          symmetrize_edges=True),
      dict(
          testcase_name='inv-sqrt-degree-normalization-no-symmetrize',
          adjacency_normalization='inverse-sqrt-degree',
          symmetrize_edges=False),
      dict(
          testcase_name='inv-degree-normalization-symmetrize',
          adjacency_normalization='inverse-degree',
          symmetrize_edges=True),
      dict(
          testcase_name='inv-degree-normalization-no-symmetrize',
          adjacency_normalization='inverse-degree',
          symmetrize_edges=False),
  )
  def test_graph_convolution_one_hop(
      self,
      add_self_loops = True,
      symmetrize_edges = False,
      adjacency_normalization = None):

    # Create a dummy graph.
    dummy_graph = get_dummy_graph(
        add_self_loops=add_self_loops,
        symmetrize_edges=symmetrize_edges,
        adjacency_normalization=adjacency_normalization)

    # Build 1-hop GCN.
    model = models.OneHopGraphConvolution(update_fn=lambda nodes: nodes)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    processed_nodes = model.apply(params, dummy_graph).nodes

    # Compute expected node features.
    adj = get_adjacency_matrix(dummy_graph)
    normalized_adj = normalize_adjacency(
        adj, adjacency_normalization=adjacency_normalization)
    expected_nodes = normalized_adj @ dummy_graph.nodes

    # Check whether outputs match.
    self.assertTrue(np.allclose(processed_nodes, expected_nodes))


if __name__ == '__main__':
  absltest.main()

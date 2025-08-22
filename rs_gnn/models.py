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

"""GNN Models in jraph/flax."""

from typing import Sequence

from flax import linen as nn
import jax.numpy as jnp
import jraph

from rs_gnn import layers


class GCN(nn.Module):
  """A flax GCN."""
  features: Sequence[int]
  drop_rate: float
  activation: str

  @nn.compact
  def __call__(self, graph, train=True):
    for i, latent_size in enumerate(self.features):
      gc = jraph.GraphConvolution(nn.Dense(latent_size), add_self_edges=False)
      graph = gc(graph)
      act_fn = layers.Activation(self.activation)
      graph = graph._replace(nodes=act_fn(graph.nodes))
      if i == len(self.features) - 1:
        return graph.nodes
      dout = nn.Dropout(rate=self.drop_rate)
      graph = graph._replace(nodes=dout(graph.nodes, deterministic=not train))


class DGI(nn.Module):
  """A flax implementation of DGI."""
  hid_dim: int

  @nn.compact
  def __call__(self, graph, c_graph, train=True):
    gcn = GCN([self.hid_dim], 0.5, 'SeLU')
    bilinear = layers.Bilinear()
    nodes1 = gcn(graph)
    nodes2 = gcn(c_graph)
    summary = layers.dgi_readout(nodes1)
    nodes = jnp.concatenate([nodes1, nodes2], axis=0)
    logits = bilinear(nodes, summary)
    return (nodes1, nodes2, summary), logits


class RSGNN(nn.Module):
  """The RSGNN model."""
  hid_dim: int
  num_reps: int

  def setup(self):
    self.dgi = DGI(self.hid_dim)
    self.cluster = Cluster(self.num_reps)

  def __call__(self, graph, c_graph, train=True):
    (h, _, _), logits = self.dgi(graph, c_graph, train)
    h = layers.normalize(h)
    centers, rep_ids, cluster_loss = self.cluster(h)
    return h, centers, rep_ids, cluster_loss, logits


class Cluster(nn.Module):
  """Finds cluster centers given embeddings."""
  num_reps: int

  @nn.compact
  def __call__(self, embs):
    cluster = layers.EucCluster(self.num_reps)
    rep_ids, cluster_dists, centers = cluster(embs)
    loss = jnp.sum(cluster_dists)
    return centers, rep_ids, loss


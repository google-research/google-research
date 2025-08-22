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

"""FLAX model definitions."""

from flax import linen as nn
import jax
from jax import numpy as jnp

import sparse_deferred.jax as sdjnp
from sparse_deferred.structs import graph_struct


class GIN(nn.Module):
  """Graph information network: https://arxiv.org/pdf/1810.00826.pdf."""

  num_classes: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32
  epsilon: float = 0.1  # See GIN paper (link above)

  def setup(self):
    layer_dims = [self.hidden_dim] * self.num_hidden_layers
    layer_dims.append(self.num_classes)
    self.layers = [nn.Dense(dim, use_bias=False) for dim in layer_dims]

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    adj = graph.adj(sdjnp.engine, 'my_edges')
    adj = adj.add_eye(1 + self.epsilon)  # self connections with 1+eps weight.

    for i, layer in enumerate(self.layers):
      x = layer(adj @ x)
      if i < self.num_hidden_layers:
        x = nn.relu(x)
    return x


class GCN(nn.Module):
  """Graph convolutional network: https://arxiv.org/pdf/1609.02907.pdf."""

  num_classes: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32

  def setup(self):
    layer_dims = [self.hidden_dim] * self.num_hidden_layers
    layer_dims.append(self.num_classes)
    self.layers = [nn.Dense(dim, use_bias=False) for dim in layer_dims]

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    adj = graph.adj(sdjnp.engine, 'my_edges')
    adj_symnorm = (adj + adj.transpose()).add_eye().normalize_symmetric()

    for i, layer in enumerate(self.layers):
      x = layer(adj_symnorm @ x)
      if i < self.num_hidden_layers:
        x = nn.relu(x)
    return x


class GraphTransformer(nn.Module):
  """Graph Transformer: https://arxiv.org/pdf/2012.09699.pdf."""

  num_classes: int
  node_dim: int
  num_hidden_layers: int = 1
  out_dim: int = 32
  num_heads: int = 8
  use_bias: bool = False

  def setup(self):
    self.layers = [
        GTLayer(self.node_dim, self.out_dim, self.num_heads, self.use_bias)
        for _ in range(self.num_hidden_layers)
    ]
    self.output = nn.Dense(self.num_classes, use_bias=self.use_bias)

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    adj = graph.adj(sdjnp.engine, 'my_edges')

    for layer in self.layers:
      x = layer(adj @ x)

    return self.output(x)


class GTLayer(nn.Module):
  """Graph Transformer Layer.

  Adapted from
  https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_layer.py
  """

  node_dim: int
  out_dim: int = 32
  num_heads: int = 8
  use_bias: bool = False

  def setup(self):
    self.attention = MHALayer(
        self.out_dim // self.num_heads, self.num_heads, self.use_bias
    )
    self.o = nn.Dense(self.node_dim, use_bias=self.use_bias)
    self.layer_norm1 = nn.LayerNorm(use_bias=self.use_bias)
    self.layer_norm2 = nn.LayerNorm(use_bias=self.use_bias)
    self.layer1 = nn.Dense(self.node_dim * 2, use_bias=self.use_bias)
    self.layer2 = nn.Dense(self.node_dim, use_bias=self.use_bias)

  def __call__(self, x):
    h_0 = x

    attn_out = self.attention(x)
    h = jnp.reshape(attn_out, [-1, self.out_dim])

    h = self.o(h)

    h = self.layer_norm1(h + h_0)
    h_1 = h

    h = self.layer1(h)
    h = nn.relu(h)
    h = self.layer2(h)

    return self.layer_norm2(h + h_1)


class MHALayer(nn.Module):
  """Multi-Head Attention Layer.

  Adapted from
  https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_layer.py
  """

  out_dim: int = 32
  num_heads: int = 8
  use_bias: bool = False

  def setup(self):
    self.q = nn.Dense(self.out_dim * self.num_heads, use_bias=self.use_bias)
    self.k = nn.Dense(self.out_dim * self.num_heads, use_bias=self.use_bias)
    self.v = nn.Dense(self.out_dim * self.num_heads, use_bias=self.use_bias)

  def __call__(self, x):
    q_h = self.q(x)
    k_h = self.k(x)
    v_h = self.v(x)

    q_h = jnp.reshape(q_h, [-1, self.out_dim, self.num_heads])
    k_h = jnp.reshape(k_h, [-1, self.out_dim, self.num_heads])
    v_h = jnp.reshape(v_h, [-1, self.out_dim, self.num_heads])

    scores = jnp.multiply(q_h, k_h)
    w = nn.softmax(scores / jnp.sqrt(self.out_dim))

    return jnp.multiply(w, v_h)

# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Implementation of multi-head self attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


class SetTransformer(snt.AbstractModule):
  """Permutation-invariant Transformer."""

  def __init__(self,
               n_layers,
               n_heads,
               n_dims,
               n_output_dims,
               n_outputs,
               layer_norm=False,
               dropout_rate=0.,
               n_inducing_points=0):
    super(SetTransformer, self).__init__()
    self._n_layers = n_layers
    self._n_heads = n_heads
    self._n_dims = n_dims
    self._n_output_dims = n_output_dims
    self._n_outputs = n_outputs
    self._layer_norm = layer_norm
    self._dropout_rate = dropout_rate
    self._n_inducing_points = n_inducing_points

  def _build(self, x, presence=None):

    batch_size = int(x.shape[0])
    h = snt.BatchApply(snt.Linear(self._n_dims))(x)

    args = [self._n_heads, self._layer_norm, self._dropout_rate]
    klass = SelfAttention

    if self._n_inducing_points > 0:
      args = [self._n_inducing_points] + args
      klass = InducedSelfAttention

    for _ in range(self._n_layers):
      h = klass(*args)(h, presence)

    z = snt.BatchApply(snt.Linear(self._n_output_dims))(h)

    inducing_points = tf.get_variable(
        'inducing_points', shape=[1, self._n_outputs, self._n_output_dims])
    inducing_points = snt.TileByDim([0], [batch_size])(inducing_points)

    return MultiHeadQKVAttention(self._n_heads)(inducing_points, z, z, presence)


class QKVAttention(snt.AbstractModule):
  """Transformer-like self-attention."""

  def _build(self, queries, keys, values, presence=None):
    """Builds the module.

    Args:
      queries: Tensor of shape [B, N, d_k].
      keys: Tensor of shape [B, M, d_k].
      values: : Tensor of shape [B, M, d_v].
      presence: None or tensor of shape [B, M].

    Returns:
      Tensor of shape [B, N, d_v]
    """

    n_dim = int(queries.shape[-1])

    # [B, M, d] x [B, d, N] = [B, M, N]
    routing = tf.matmul(queries, keys, transpose_b=True)

    if presence is not None:
      presence = tf.cast(tf.expand_dims(presence, -2), tf.float32)
      routing -= (1. - presence) * 1e32

    routing = tf.nn.softmax(routing / np.sqrt(n_dim), -1)

    # every output is a linear combination of all inputs
    # [B, M, N] x [B, N, d_v] = [B, M, d_v]
    res = tf.matmul(routing, values)
    return res


class MultiHeadQKVAttention(snt.AbstractModule):
  """Multi-head version of Transformer-like attention."""

  def __init__(self, n_heads):
    super(MultiHeadQKVAttention, self).__init__()
    self._n_heads = n_heads

  def _build(self, queries, keys, values, presence=None):

    def transform(x, n=self._n_heads):
      n_dim = np.ceil(float(int(x.shape[-1])) / n)
      return snt.BatchApply(snt.Linear(int(n_dim)))(x)

    outputs = []
    for _ in range(self._n_heads):
      args = [transform(i) for i in [queries, keys, values]]
      if presence is not None:
        args.append(presence)
      outputs.append(QKVAttention()(*args))

    linear = snt.BatchApply(snt.Linear(values.shape[-1]))
    return linear(tf.concat(outputs, -1))


class SelfAttention(snt.AbstractModule):
  """Self-attention where keys, values and queries are the same."""

  def __init__(self, n_heads, layer_norm=False, dropout_rate=0.):
    super(SelfAttention, self).__init__()
    self._n_heads = n_heads
    self._layer_norm = layer_norm
    self._dropout_rate = dropout_rate

  def _build(self, x, presence=None):
    n_dims = int(x.shape[-1])

    y = self._self_attention(x, presence)

    if self._dropout_rate > 0.:
      x = tf.nn.dropout(x, rate=self._dropout_rate)

    y += x

    if presence is not None:
      y *= tf.expand_dims(tf.cast(presence, tf.float32), -1)

    if self._layer_norm:
      y = snt.LayerNorm(axis=-1)(y)

    h = snt.BatchApply(snt.nets.MLP([2*n_dims, n_dims]))(y)

    if self._dropout_rate > 0.:
      h = tf.nn.dropout(h, rate=self._dropout_rate)

    h += y

    if self._layer_norm:
      h = snt.LayerNorm(axis=-1)(h)

    return h

  def _self_attention(self, x, presence):
    heads = MultiHeadQKVAttention(self._n_heads)
    return heads(x, x, x, presence)


class InducedSelfAttention(SelfAttention):
  """Self-attention with inducing points a.k.a. ISAB from SetTransformer."""

  def __init__(self, n_inducing_points, n_heads, layer_norm=False,
               dropout_rate=0.):
    super(InducedSelfAttention, self).__init__(n_heads, layer_norm,
                                               dropout_rate)

    self._n_inducing_points = n_inducing_points

  def _self_attention(self, x, presence):

    head_before = MultiHeadQKVAttention(self._n_heads)
    # head_after = MultiHeadQKVAttention(self._n_heads)
    head_after = head_before

    inducing_points = tf.get_variable(
        'inducing_points', shape=[1, self._n_inducing_points,
                                  int(x.shape[-1])])

    inducing_points = snt.TileByDim([0], [int(x.shape[0])])(inducing_points)

    z = head_before(inducing_points, x, x)
    y = head_after(x, z, z)
    return y

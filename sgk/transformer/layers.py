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

"""Layers for Sparse Transformer models."""
import tensorflow.compat.v1 as tf

from sgk.sparse import ops


def preprocess_attention_component(x):
  shape = x.shape.as_list()
  assert len(shape) == 4
  return tf.reshape(x, [-1] + shape[2:])


def sparse_dot_product_attention(q, k, v, topology, **_):
  q_3d, k_3d, v_3d = [preprocess_attention_component(x) for x in [q, k, v]]
  logits = ops.replicated_sddmm(q_3d, k_3d, topology, transpose_rhs=True)
  weights = ops.replicated_sparse_softmax(logits, topology)
  out = ops.replicated_spmm(weights, topology, v_3d)
  return tf.reshape(out, tf.shape(q))


def dot_product_attention(q, k, v, bias, **_):
  """Dot product attention with our memory efficient softmax."""
  logits = tf.matmul(q, k, transpose_b=True)
  logits = tf.math.add(logits, bias)
  weights = ops.fused_softmax(logits)
  return tf.matmul(weights, v)

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

"""Tensorflow implementation of Fast Weight Layers.

A note on implementation: this code was originally implemented in TF1 (for
compatibility with Transformer-XL), but we have ported it to TF2 for easier
use. We ran into out-of-memory issues when computing second-order gradients on
TPUs with automatic differentiation, so this implementation computes the slow
weight backwards pass "by hand" rather than automatically.
"""

import tensorflow as tf


def causal_mask(x):
  seq_len = x.shape[-2]
  return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), 0, -1)


def _local_quadratic_attn(q, k, v):
  qk = tf.einsum("bgns,bgms->bgnm", q, k) * causal_mask(q)
  return tf.einsum("bgnm,bgme->bgne", qk, v)


def _global_linear_attn(q, k, v):
  kv = tf.einsum("bgcs,bgce->bgse", k, v)
  kv = tf.cumsum(kv, axis=1, exclusive=True)
  return tf.einsum("bgcs,bgse->bgce", q, kv)


def mixed_chunk_attn(q, k, v, n_chunks):
  """Mixed chunk linear attention from https://arxiv.org/pdf/2202.10447.pdf."""
  if n_chunks == 1:
    # if 1 chunk we do regular linear attention
    qk = tf.einsum("blh,bmh->blm", q, k) * causal_mask(q)
    return tf.einsum("blm,bmp->blp", qk, v)
  batch_size, seq_len = q.shape[:2]
  chunked_shape = [batch_size, n_chunks, seq_len // n_chunks, -1]
  q = tf.reshape(q, chunked_shape)
  v = tf.reshape(v, chunked_shape)
  k = tf.reshape(k, chunked_shape)
  attn = _global_linear_attn(q, k, v) + _local_quadratic_attn(q, k, v)
  return tf.reshape(attn, [batch_size, seq_len, -1])


class FWBlock(tf.keras.layers.Layer):
  """NN layers with fast-weight gradient updates for text generation."""

  def __init__(self, size, vocab_size, attn_chunks):
    self.size = size
    self.attn_chunks = attn_chunks
    self.layers = [
        FWDense(4 * size, attn_chunks, True),
        FWSquaredReLU(),
        FWDense(size, attn_chunks),
        FWLayerNorm(),
        FWUnembed(vocab_size)
    ]

  def fwd(self, x, use_fast_weights):
    output = x
    for layer in self.layers:
      output = layer(output, use_fast_weights)
    return output

  def bwd(self, x, labels, weights):
    grad = self.layers[-1].bwd(labels, weights)
    for layer in reversed(self.layers[:-1]):
      grad = layer.bwd(grad)

  def __call__(self, x, labels, weights):
    self.fwd(x, False)
    self.bwd(x, labels, weights)
    return self.fwd(x, True)


class FWUnembed(tf.keras.layers.Layer):
  """Unembed layer for use with fast weights."""

  def __init__(self, vocab_size):
    super().__init__()
    self.unembed = tf.keras.layers.Dense(vocab_size)
    self.probs = None

  def call(self, x, use_fast_weights=False):
    logits = self.unembed(x)
    self.probs = tf.nn.softmax(logits)
    return logits

  def bwd(self, labels, weights):
    grad = ((self.probs - labels) * tf.expand_dims(weights, -1) /
            tf.maximum(tf.reduce_sum(weights), 1e-8))
    return tf.matmul(grad, self.unembed.kernel, transpose_b=True)


class FWSquaredReLU(tf.keras.layers.Layer):
  """Squared ReLU activation for use with fast weights."""

  def __init__(self):
    super().__init__()
    self.sw_input = None
    self.sw_output = None

  def call(self, x, use_fast_weights=False):
    self.sw_input = x
    self.sw_output = tf.square(tf.nn.relu(x))
    return self.sw_output

  def bwd(self, upstream_grad):
    return upstream_grad * 2 * self.sw_input * tf.sign(self.sw_output)


class FWDense(tf.keras.layers.Layer):
  """Dense layer with fast weights."""

  def __init__(self, size, attn_chunks, is_first_layer=False):
    super().__init__()
    self.size = size
    self.attn_chunks = attn_chunks
    self.is_first_layer = is_first_layer
    self.sw_input = None
    self.sw_output = None
    self.upstream_grad = None

  def build(self, input_shape):
    self.kernel = tf.Variable(
        0.05 * tf.random.normal([input_shape[-1], self.size]), name="kernel")
    self.bias = tf.Variable(tf.zeros([self.size]), name="bias")
    self.kernel_step_size = tf.Variable(0.01, name="kernel_step_size")
    self.bias_step_size = tf.Variable(0.01, name="bias_step_size")

  def call(self, x, use_fast_weights=False):
    if use_fast_weights:
      output = (self.sw_output if self.is_first_layer else
                tf.matmul(x, self.kernel) + self.bias)
      kernel_update = mixed_chunk_attn(
          x, self.sw_input, self.upstream_grad, self.attn_chunks)
      bias_update = tf.cumsum(self.upstream_grad, exclusive=True, axis=1)
      return (output - (self.kernel_step_size * kernel_update) -
              (self.bias_step_size * bias_update))
    else:
      self.sw_input = x
      self.sw_output = tf.matmul(x, self.kernel) + self.bias
      return self.sw_output

  def bwd(self, upstream_grad):
    self.upstream_grad = upstream_grad
    if not self.is_first_layer:
      return tf.matmul(upstream_grad, self.kernel, transpose_b=True)


class FWLayerNorm(tf.keras.layers.Layer):
  """Layer norm with fast weights."""

  def __init__(self):
    super().__init__()
    self.sw_input = None
    self.u = None
    self.sigma = None
    self.shifted = None
    self.scaled = None
    self.upsteam_grad = None

  def build(self, input_shape):
    self.alpha = tf.Variable(tf.ones([input_shape[-1]]), name="alpha")
    self.beta = tf.Variable(tf.zeros([input_shape[-1]]), name="beta")
    self.alpha_step_size = tf.Variable(0.01, name="alpha_step_size")
    self.beta_step_size = tf.Variable(0.01, name="beta_step_size")

  def call(self, x, use_fast_weights=False):
    if use_fast_weights:
      u = tf.reduce_mean(x, -1, keepdims=True)
      sigma = tf.sqrt(tf.reduce_mean(tf.square(x - u), -1, keepdims=True))
      shifted = x - u
      alpha_update = tf.cumsum(self.upsteam_grad * self.shifted / self.sigma,
                               exclusive=True, axis=1)
      beta_update = tf.cumsum(self.upsteam_grad, exclusive=True, axis=1)
      alpha_fw = self.alpha - self.alpha_step_size * alpha_update
      beta_fw = self.beta - self.beta_step_size * beta_update
      return (alpha_fw * shifted / sigma) + beta_fw
    else:
      self.sw_input = x
      self.u = tf.reduce_mean(x, -1, keepdims=True)
      self.sigma = tf.sqrt(tf.reduce_mean(tf.square(x - self.u), -1,
                                          keepdims=True))
      self.shifted = x - self.u
      self.scaled = (self.alpha * self.shifted / self.sigma) + self.beta
      return self.scaled

  def bwd(self, upstream_grad):
    self.upsteam_grad = upstream_grad
    n_in = self.alpha.shape[0]

    dscaled = upstream_grad * self.alpha
    dshifted = dscaled / self.sigma
    du = -tf.reduce_sum(dshifted, -1, keepdims=True)
    dsigma = -tf.reduce_sum(dscaled * self.shifted / tf.square(self.sigma),
                            -1, keepdims=True)
    dsigmadz = (self.sw_input - self.u) / (self.sigma * n_in)
    dz_shifted = dshifted
    dz_u = du / n_in
    dz_sigma = dsigma * dsigmadz
    return dz_shifted + dz_u + dz_sigma


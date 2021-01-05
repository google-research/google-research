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

"""Utilities for Tensorflow 2.0.

Partially adapted from:
https://www.tensorflow.org/tutorials/text/image_captioning
"""
# Lint as: python3
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def film_params(sentence_embedding, n_layer_channel):
  """Generate FiLM parameters from a sentence embedding.

  Generate FiLM parameters from a sentence embedding. This method assumes a
  batch dimension exists.

  Args:
    sentence_embedding: a tensor containing batched sentenced embedding to be
      transformed
    n_layer_channel:    a list of integers specifying how many channels are at
      each hidden layer to be FiLM'ed

  Returns:
    a tuple of tensors the same length as n_layer_channel. Each element
    contains all gamma_i and beta_i for a single hidden layer.
  """
  n_total = sum(n_layer_channel) * 2
  all_params = tf.layers.dense(sentence_embedding, n_total)
  all_params = tf.keras.layers.Dense(
      2 * sum * (n_layer_channel), activation=tf.nn.relu)
  return tf.split(all_params, [c * 2 for c in n_layer_channel], 1)


def stack_conv_layer(layer_cfg, padding='same'):
  """Stack convolution layers per layer_cfg.

  Args:
    layer_cfg: list of integer tuples specifying the parameter each layer;
      each tuple should be (channel, kernel size, strides)
    padding: what kind of padding the conv layers use

  Returns:
    the keras model with stacked conv layers
  """
  layers = []
  for cfg in layer_cfg[:-1]:
    layers.append(
        tf.keras.layers.Conv2D(
            filters=cfg[0],
            kernel_size=cfg[1],
            strides=cfg[2],
            activation=tf.nn.relu,
            padding=padding))
  final_cfg = layer_cfg[-1]
  layers.append(
      tf.keras.layers.Conv2D(
          final_cfg[0], final_cfg[1], final_cfg[2], padding=padding))
  return tf.keras.Sequential(layers)


def stack_dense_layer(layer_cfg):
  """Stack Dense layers.

  Args:
    layer_cfg: list of integer specifying the number of units at each layer

  Returns:
    the keras model with stacked dense layers
  """
  layers = []
  for cfg in layer_cfg[:-1]:
    layers.append(tf.keras.layers.Dense(cfg, activation=tf.nn.relu))
  layers.append(tf.keras.layers.Dense(layer_cfg[-1]))
  return tf.keras.Sequential(layers)


def soft_variables_update(source_variables, target_variables, polyak_rate=1.0):
  """Update the target variables using exponential moving average.

  Specifically, v_s' = v_s * polyak_rate + (1-polyak_rate) * v_t

  Args:
    source_variables:  the moving average variables
    target_variables:  the new observations
    polyak_rate: rate of moving average

  Returns:
    Operation that does the update
  """
  updates = []
  for (v_s, v_t) in zip(source_variables, target_variables):
    v_t.shape.assert_is_compatible_with(v_s.shape)

    def update_fn(v1, v2):
      """Update variables."""
      # For not trainable variables do hard updates.
      return v1.assign(polyak_rate * v1 + (1 - polyak_rate) * v2)

    update = update_fn(v_t, v_s)
    updates.append(update)
  return updates


def vector_tensor_product(a, b):
  """"Returns keras layer that perfrom a outer product between a and b."""
  # a shape: [B, ?, d], b shape: [B, ?, d]
  shape_layer = tf.keras.layers.Lambda(tf.shape)
  shape = shape_layer(b)
  shape_numpy = b.get_shape()
  variable_length = shape[1]  # variable_len = ?
  expand_dims_layer_1 = tf.keras.layers.Reshape((-1, 1, shape_numpy[-1]))
  expand_dims_layer_2 = tf.keras.layers.Reshape((-1, 1, shape_numpy[-1]))
  a = expand_dims_layer_1(a)  # a shape: [B, ?, 1, d]
  b = expand_dims_layer_2(b)  # a shape: [B, ?, 1, d]
  tile_layer = tf.keras.layers.Lambda(
      lambda inputs: tf.tile(inputs[0], multiples=inputs[1]))
  a = tile_layer((a, [1, 1, variable_length, 1]))  # a shape: [B, ?, ?, d]
  b = tile_layer((b, [1, 1, variable_length, 1]))  # b shape: [B, ?, ?, d]
  b = tf.keras.layers.Permute((2, 1, 3))(b)  # b shape: [B, ?, ?, d]
  return tf.keras.layers.concatenate([a, b])  # shape: [B, ?, ?, 2*d]


class BahdanauAttention(tf.keras.Model):
  """Bahdanau Attention Layer.

  Attributes:
    w1: weights that process the feature
    w2: weights that process the memory state
    v: projection layer that project score vector to scalar
  """

  def __init__(self, units):
    """Initialize Bahdanau attention layer.

    Args:
      units: size of the dense layers
    """
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class GRUEnecoder(tf.keras.Model):
  """TF2.0 GRE encoder.

  Attributes:
    embedding: word embedding matrix
    gru: the GRU layer
  """

  def __init__(self, embedding_dim, units, vocab_size):
    """Initialize the GRU encoder.

    Args:
      embedding_dim: dimension of word emebdding
      units: number of units of the memory state
      vocab_size: total number of vocabulary
    """
    super(GRUEnecoder, self).__init__()
    self._units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(
        self.units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)
    # passing the concatenated vector to the GRU
    output, state = self.gru(x)
    return output, state

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self._units))

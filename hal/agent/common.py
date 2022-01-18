# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Common modules used by many agents."""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def get_vars(scope_name):
  """Returns variables in scope."""
  scope_var = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
  return scope_var


def encoder(inputs,
            embeddings,
            n_hidden_unit,
            trainable=True,
            reuse=False,
            name='encoder',
            time_major=False,
            cell_collection=None):
  """One layer GRU unit encoder.

  Args:
    inputs: a batch of sequences of integers (indices of tokens)
    embeddings:  word embedding matrix
    n_hidden_unit: number of hidden units the encoder has
    trainable: whether the weights are trainable
    reuse: whether to reuse the parameters
    name: optional name of the encoder
    time_major: whether the format is time major
    cell_collection: optional list to put the encoder cell in

  Returns:
    encoder_outputs: output of the encoder
    encoder_final_state: the final hidden state of the encoder
  """
  with tf.variable_scope(name, reuse=reuse):
    input_embedding = tf.nn.embedding_lookup(embeddings, inputs)
    encoder_cell = tf.contrib.rnn.GRUCell(n_hidden_unit, trainable=trainable)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell,
        input_embedding,
        dtype=tf.float32,
        time_major=time_major,
    )
    if cell_collection: cell_collection.append(encoder_cell)
    return encoder_outputs, encoder_final_state


def stack_dense_layer(inputs, layer_cfg):
  """Stack dense layers per layer_cfg.

  Args:
    inputs: input tensor
    layer_cfg: list of integer specifying the number of units at each layer

  Returns:
    output after all layers
  """
  for cfg in layer_cfg[:-1]:
    inputs = tf.layers.dense(inputs, cfg, activation=tf.nn.relu)
  return  tf.layers.dense(inputs, layer_cfg[-1])


def stack_conv_layer(inputs, layer_cfg, padding='same'):
  """Stack convolution layers per layer_cfg.

  Args:
    inputs: input tensor
    layer_cfg: list of integer tuples specifying the parameter each layer;
      each tuple should be (channel, kernel size, strides)
    padding: what kind of padding the conv layers use

  Returns:
    output after all layers
  """
  for cfg in layer_cfg[:-1]:
    inputs = tf.layers.conv2d(
        inputs,
        filters=cfg[0],
        kernel_size=cfg[1],
        strides=cfg[2],
        activation=tf.nn.relu,
        padding=padding
    )
  final_cfg = layer_cfg[-1]
  return  tf.layers.conv2d(
      inputs, final_cfg[0], final_cfg[1], final_cfg[2], padding=padding)


def film_params(sentence_embedding, n_layer_channel):
  """Generate FiLM parameters from a sentence embedding.

  Generate FiLM parameters from a sentence embedding. This method assumes a
  batch dimension exists.

  Args:
    sentence_embedding: a tensor containing batched sentenced embedding to be
                          transformed
    n_layer_channel:    a list of integers specifying how many channels are
                          at each hidden layer to be FiLM'ed

  Returns:
    A tuple of tensors the same length as n_layer_channel. Each element
    contains all gamma_i and beta_i for a single hidden layer.
  """
  n_total = sum(n_layer_channel) * 2
  all_params = tf.layers.dense(sentence_embedding, n_total)
  return tf.split(all_params, [c*2 for c in n_layer_channel], 1)


def vector_tensor_product(a, b):
  """"Conduct an outer product between two tensors.

  Instead of conducting scalar multiplication like regular outer product, this
  operation does 1-D vector concatenation. It also does it over entire batch.

  Args:
    a: a tensor of shape [B, ?, d_a]
    b: a tensor of shape [B, ?, d_b]

  Returns:
    a tensor of shape [B, ?, ?, d_a + d_b]
  """
  # a shape: [B, ?, d], b shape: [B, ?, d]
  variable_length = tf.shape(a)[1]  # variable_len = ?
  a = tf.expand_dims(a, axis=2)  # a shape: [B, ?, 1, d]
  b = tf.expand_dims(b, axis=2)  # b shape: [B, ?, 1, d]
  a = tf.tile(a, multiples=[1, 1, variable_length, 1])  # a shape: [B, ?, ?, d]
  b = tf.tile(b, multiples=[1, 1, variable_length, 1])  # b shape: [B, ?, ?, d]
  b = tf.transpose(b, perm=[0, 2, 1, 3])  # b shape: [B, ?, ?, d]
  return tf.concat([a, b], axis=-1)  # shape: [B, ?, ?, 2*d]


def tensor_concat(a, b, c):
  """Do tensor product between 3 vectors."""
  # a shape = [B, dc, de], b shape = [B, db, de], c shape = [B, dc, de]
  dim_a, dim_b, dim_c = tf.shape(a)[1], tf.shape(b)[1], tf.shape(c)[1]
  a = tf.expand_dims(a, axis=2)  # [B, da, 1, de]
  b = tf.expand_dims(b, axis=2)  # [B, db, 1, de]
  c = tf.expand_dims(c, axis=2)  # [B, dc, 1, de]
  c = tf.expand_dims(c, axis=3)  # [B, dc, 1, 1, de]
  a = tf.tile(a, multiples=[1, 1, dim_b, 1])  # [B, da, db, de]
  b = tf.tile(b, multiples=[1, 1, dim_a, 1])  # [B, db, da, de]
  c = tf.tile(c, multiples=[1, 1, dim_a, dim_b, 1])  # [B, dc, da, db, de]
  b = tf.transpose(b, perm=[0, 2, 1, 3])  # [B, da, db, de]
  ab = tf.concat([a, b], axis=-1)  # [B, da, db, de*2]
  ab = tf.expand_dims(ab, axis=3)  # [B, da, db, 1, de*2]
  ab = tf.tile(ab, multiples=[1, 1, 1, dim_c, 1])  # [B, da, db, dc, 2*de]
  c = tf.transpose(c, perm=[0, 2, 3, 1, 4])  # [B, da, db, dc, de]
  abc = tf.concat([ab, c], axis=-1)  # [B, da, db, dc, 3*de]
  return tf.identity(abc)


def factor_concat(factors):
  """Generalization of tensor_concat to any numbers of batched 2D tensors."""
  assert len(factors) >= 2
  primary_fac = factors[0]
  final_factor_shape = [-1, primary_fac.get_shape()[1]]
  for fac in factors[1:]:
    primary_fac = tf.expand_dims(primary_fac, axis=2)
    fac = tf.expand_dims(fac, axis=2)
    fac_shape = fac.get_shape()
    primary_shape = primary_fac.get_shape()
    # tiling primary to match the shape of fac
    primary_fac = tf.tile(primary_fac, multiples=[1, 1, fac_shape[1], 1])
    # tiling current fac to the shape of primary
    fac = tf.tile(fac, multiples=[1, 1, primary_shape[1], 1])
    # transpose the current fac
    fac = tf.transpose(fac, perm=[0, 2, 1, 3])
    primary_fac = tf.concat([primary_fac, fac], axis=-1)
    primary_fac = tf.reshape(
        primary_fac,
        shape=[
            -1,
            fac_shape[1]*primary_shape[1],
            primary_fac.get_shape()[-1]
        ]
    )
    final_factor_shape.append(fac_shape[1])
  return primary_fac

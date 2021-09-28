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

"""RNN models of BasisNet with LSTM cells.

It implements a RNN wrapper with specialized LSTM cell with bases
for the kernels.

"""

import functools
from typing import Optional

import tensorflow as tf

CLIENT_SZIE = 500000


class BasisRNNLayer(tf.keras.layers.Layer):
  """A RNN wrapper using LSTM cell with Basis kernels."""

  def __init__(self,
               cell,
               num_units,
               num_basis,
               recurrent_initializer,
               kernel_initializer,
               return_sequences=False):
    super().__init__()

    self.rnn_cell = cell(
        num_units=num_units,
        num_basis=num_basis,
        recurrent_initializer=recurrent_initializer,
        kernel_initializer=kernel_initializer)

    self.rnn = tf.keras.layers.RNN(
        self.rnn_cell, return_sequences=return_sequences)

  def call(self, input_tensor):
    return self.rnn(input_tensor)


class BasisLSTMCell(tf.keras.layers.Layer):
  """A LSTM cell with Basis kernels."""

  def __init__(self,
               num_units,
               num_basis,
               kernel_initializer,
               recurrent_initializer,
               word_emb_size=96,
               use_bias=True,
               activation=None,
               ):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      num_basis: The number of bases to learn.
      kernel_initializer: The initializer of the input/output kernels.
      recurrent_initializer: The initializer of the recurrent kernels.
      word_emb_size: The word embedding size.
      use_bias: Add bias or not.
      activation: Activation function of the inner states.  Default: `tanh`.
    """
    super().__init__()

    self._num_basis = num_basis
    self.kernel_initializer = kernel_initializer
    self.recurrent_initializer = recurrent_initializer

    self._num_units = num_units
    self.word_emb_size = word_emb_size

    self.activation = activation or tf.tanh
    self.recurrent_activation = tf.sigmoid

    self.use_bias = use_bias

  def build(self, input_shape):
    # the basis embedding is concatenated to the input embedding,
    # then split out in call().

    weight_shape = [self.word_emb_size, self._num_basis, 4 * self._num_units]
    self.basis_kernel = self.add_weight(
        shape=weight_shape,
        name='kernel',
        initializer=self.kernel_initializer,)

    self.basis_recurrent_kernel = self.add_weight(
        shape=(self._num_units, self._num_basis, self._num_units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
    )

    self.bias = tf.Variable([0.0]*weight_shape[-1], name='bias')

  @property
  def state_size(self):
    return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._num_units,
                                                   self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def compose_basis(self, c_prob):
    """Compose bases into a kernel."""

    composed_kernel = tf.keras.backend.sum(
        tf.expand_dims(self.basis_kernel, 0) * c_prob, axis=2)
    composed_recurrent_kernel = tf.keras.backend.sum(
        tf.expand_dims(self.basis_recurrent_kernel, 0) * c_prob, axis=2)
    return composed_kernel, composed_recurrent_kernel

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    inputs, c_prob = tf.split(inputs, [self.word_emb_size, self._num_basis], -1)
    c_prob = tf.reshape(c_prob, [-1, 1, self._num_basis, 1])
    composed_kernel, composed_recurrent_kernel = self.compose_basis(c_prob)

    # inputs:
    #   [batch_size, 1, self.word_emb_size]
    # composed_kernel:
    #   [batch_size, self.word_emb_size, self._num_units]
    # outputs (need to be squeezed):
    #   [batch_size, 1, self._num_units]
    z = tf.matmul(tf.expand_dims(inputs, 1), composed_kernel)
    z += tf.matmul(tf.expand_dims(h_tm1, 1), composed_recurrent_kernel)
    if self.use_bias:
      z = tf.keras.backend.bias_add(z, self.bias)
    z = tf.squeeze(z)

    z = tf.split(z, num_or_size_splits=4, axis=1)
    c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]


class TransposableEmbedding(tf.keras.layers.Layer):
  """A Keras layer implements a transposed projection for output."""

  def __init__(self, embedding_layer):
    super().__init__()
    self.embeddings = embedding_layer.embeddings

  # Placing `tf.matmul` under the `call` method is important for backpropagating
  # the gradients of `self.embeddings` in graph mode.
  def call(self, inputs):
    return tf.matmul(inputs, self.embeddings, transpose_b=True)


def create_basis_recurrent_model(vocab_size = 10000,
                                 num_oov_buckets = 1,
                                 embedding_size = 96,
                                 latent_size = 670,
                                 num_basis = 1,
                                 seqeunce_length = 20,
                                 name = 'rnn',
                                 shared_embedding = False,
                                 global_embedding_only = False,
                                 seed = 0):
  """Constructs zero-padded keras model with the given parameters and cell.

  Args:
    vocab_size: Size of vocabulary to use.
    num_oov_buckets: Number of out of vocabulary buckets.
    embedding_size: The size of the embedding.
    latent_size: The size of the recurrent state.
    num_basis: The number of bases to learn.
    seqeunce_length: The seqeunce length of an input.
    name: (Optional) string to name the returned `tf.keras.Model`.
    shared_embedding: (Optional) Whether to tie the input and output
      embeddings.
    global_embedding_only: use the global embedding only or not.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.

  Returns:
    `tf.keras.Model`.
  """
  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.

  input_x = tf.keras.layers.Input(shape=(None,), name='input_x')
  input_id = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='input_id')
  input_embedding = tf.keras.layers.Embedding(
      input_dim=extended_vocab_size,
      output_dim=embedding_size,
      mask_zero=True,
      embeddings_initializer=tf.keras.initializers.RandomUniform(seed=seed),
  )
  embedded = input_embedding(input_x)
  projected = embedded

  # Somehow if the vocabulary size is too small,
  # no out-of-range error will be reported and the model is still good
  basis_embeddings = tf.keras.layers.Embedding(
      CLIENT_SZIE, num_basis, name='client_embedding')

  if global_embedding_only:
    # using id = 0 for the global embedding
    basis_vec = basis_embeddings(tf.zeros_like(input_id))
  else:
    basis_vec = basis_embeddings(input_id)

  # [batch_size, 1, num_basis]
  basis_vec = tf.reshape(basis_vec, shape=[-1, 1, num_basis])
  basis_prob = tf.keras.layers.Softmax()(basis_vec)

  basis_tensor = tf.tile(
      basis_prob,
      tf.constant([1, seqeunce_length, 1], tf.int32))
  projected = tf.concat([projected, basis_tensor], -1)

  recurrent_initializer = tf.keras.initializers.Orthogonal(seed=seed)
  kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
  lstm_layer_builder = functools.partial(
      BasisRNNLayer,
      cell=BasisLSTMCell,
      num_units=latent_size,
      num_basis=num_basis,
      recurrent_initializer=recurrent_initializer,
      kernel_initializer=kernel_initializer,
      return_sequences=True,)

  dense_layer_builder = functools.partial(
      tf.keras.layers.Dense,
      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

  layer = lstm_layer_builder()
  processed = layer(projected)
  # A projection changes dimension from rnn_layer_size to input_embedding_size
  dense_layer = dense_layer_builder(units=embedding_size)
  projected = dense_layer(processed)
  projected = tf.concat([projected, basis_tensor], -1)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    logits = transposed_embedding(projected)
  else:
    final_dense_layer = dense_layer_builder(
        units=extended_vocab_size, activation=None)
    logits = final_dense_layer(projected)

  return tf.keras.Model(inputs=[input_x, input_id], outputs=logits, name=name)


def create_recurrent_model(vocab_size = 10000,
                           num_oov_buckets = 1,
                           embedding_size = 96,
                           latent_size = 670,
                           num_layers = 1,
                           name = 'rnn',
                           shared_embedding = False,
                           seed = 0):
  """Constructs zero-padded keras model with the given parameters and cell.

  Args:
    vocab_size: Size of vocabulary to use.
    num_oov_buckets: Number of out of vocabulary buckets.
    embedding_size: The size of the embedding.
    latent_size: The size of the recurrent state.
    num_layers: The number of layers.
    name: (Optional) string to name the returned `tf.keras.Model`.
    shared_embedding: (Optional) Whether to tie the input and output
      embeddings.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.

  Returns:
    `tf.keras.Model`.
  """
  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
  input_x = tf.keras.layers.Input(shape=(None,), name='input_x')
  # To be consistent with BasisNet pipeline, not using client id
  input_id = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='input_id')
  input_embedding = tf.keras.layers.Embedding(
      input_dim=extended_vocab_size,
      output_dim=embedding_size,
      mask_zero=True,
      embeddings_initializer=tf.keras.initializers.RandomUniform(seed=seed),
  )
  embedded = input_embedding(input_x)
  projected = embedded

  lstm_layer_builder = functools.partial(
      tf.keras.layers.LSTM,
      units=latent_size,
      return_sequences=True,
      recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed),
      kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))

  dense_layer_builder = functools.partial(
      tf.keras.layers.Dense,
      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

  for _ in range(num_layers):
    layer = lstm_layer_builder()
    processed = layer(projected)
    # A projection changes dimension from rnn_layer_size to input_embedding_size
    dense_layer = dense_layer_builder(units=embedding_size)
    projected = dense_layer(processed)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    logits = transposed_embedding(projected)
  else:
    final_dense_layer = dense_layer_builder(
        units=extended_vocab_size, activation=None)
    logits = final_dense_layer(projected)

  return tf.keras.Model(inputs=[input_x, input_id], outputs=logits, name=name)

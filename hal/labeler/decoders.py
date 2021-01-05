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

# Lint as: python3
"""Encoders that decode embeddings to instructions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


class RNNDecoder(tf.keras.Model):
  """Decoder that decodes a single row vector of embedded representation.

  Attributes:
    units: number of unites the GRU has
    embedding: word embedding matrix
    gru: the GRU layer this answerer uses
    fc1: first fully connected layer
    fc2: second fully connected layer
    dropout1: first dropout layer
  """

  def __init__(self,
               embedding_dim,
               hidden_units,
               vocab_size,
               name='RNNDecoder'):
    """Initializes the RNN decoder.

    Args:
      embedding_dim: size of the word embedding
      hidden_units: size of the memory state
      vocab_size: number of vocabulary
      name: optional name for the name scope
    """
    super(RNNDecoder, self).__init__(name=name)
    self.units = hidden_units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(
        self.units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.dropout1 = tf.keras.layers.Dropout(0.5)

  def call(self, x, features, hidden):
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(features, 1), x], axis=-1)
    # passing the concatenated vector to the GRU
    output, state = self.gru(x, initial_state=hidden)
    state = self.dropout1(state)
    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)
    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))
    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)
    return x, state, None

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


class FilmDecoder(tf.keras.Model):
  """Decoder for Feature Linear Modulation decoder.

  Attributes:
    text_encoder: keras model that enc
    embedding: word embedding matrix
    gru: the GRU layer this answerer uses
    fc1: first fully connected layer
    fc2: second fully connected layer
    dropout1: first dropout layer
  """

  def __init__(self, vocab_size, name='FiLM-Decoder'):
    super(FilmDecoder, self).__init__(name=name)
    self.text_encoder = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Dense(64, activation='relu'),
    ])
    self.fc1 = tf.keras.layers.Dense(256, activation='relu')
    self.fc2 = tf.keras.layers.Dense(128)
    self.fc_film = tf.keras.layers.Dense(128, activation='sigmoid')

  def call(self, text, features):
    text_embedding = self.text_encoder(text)
    film_mask = self.fc_film(text_embedding)
    x = self.fc1(features)
    x = self.fc2(x)
    return tf.multiply(x, film_mask)

################################################################################


class BahdanauAttention(tf.keras.Model):
  """Bahdanau Attention Layer.

  Attributes:
    w1: weights that process the feature
    w2: weights that process the memory state
    v: projection layer that project score vector to scalar
  """

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.w1 = tf.keras.layers.Dense(units)
    self.w2 = tf.keras.layers.Dense(units)
    self.v = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.w1(features) + self.w2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.v(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class RNNAttentionDecoder(tf.keras.Model):
  """Decoder that decodes a embedded representation.

  Attributes:
    units: size of the hidden units of GRU
    embedding: embedding matrix for text
    gru: the GRU layer
    fc1: first dense layer that process tha hidden state
    fc2: second dense layer that process tha hidden state
    dropout: dropout layer applied to the hidden state
    attention: Bahdanau attention layer
  """

  def __init__(self, embedding_dim, units, vocab_size, name='RNN Decoder'):
    super(RNNAttentionDecoder, self).__init__(name=name)
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(
        self.units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden, training=None):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size,1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    state = self.dropout(state, training=training)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

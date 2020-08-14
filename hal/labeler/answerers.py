# coding=utf-8
# Copyright 2020 The Google Research Authors.
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


class RNNAnswerer(tf.keras.Model):
  """Decoder that is used for VQA.

  Attributes:
    units: number of unites the GRU has
    embedding: word embedding matrix
    gru: the GRU layer this answerer uses
    fc1: first fully connected layer
    fc2: second fully connected layer
  """

  def __init__(self,
               embedding_dim,
               hidden_units,
               vocab_size,
               name='RNNDecoder'):
    """Initializes the RNN answerer.

    Args:
      embedding_dim: size of the word embedding
      hidden_units: size of the memory state
      vocab_size: number of vocabulary
      name: optional name for the name scope
    """
    super(RNNAnswerer, self).__init__(name=name)
    self.units = hidden_units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(
        self.units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

  def call(self, x, features, hidden):
    """Process one time step of the label.

    Args:
      x: input token
      features: features of the state
      hidden: hidden state of the GRU

    Returns:
      next token and updated state
    """
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(features, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x, initial_state=hidden)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, None

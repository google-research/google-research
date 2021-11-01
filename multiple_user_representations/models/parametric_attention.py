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

"""Defines a simple parametric attention model that is used for user tower."""

import tensorflow as tf

from multiple_user_representations.models import model_utils


class SimpleParametricAttention(tf.keras.Model):
  """A parametric attention model that can be used as a user tower."""

  def __init__(self,
               output_dimension,
               max_sequence_size,
               vocab_size,
               input_embedding_dimension,
               num_representations = 1,
               use_positional_encoding = False,
               use_projection_layer = False,
               mask_zero = False):
    """Initializes the parameteric attention model.

    Args:
      output_dimension: The output dimension of the user representation.
      max_sequence_size: The maximum size of the input sequence.
      vocab_size: The vocabulary size for input tokens/items.
      input_embedding_dimension: The embedding dimension for input tokens/items.
      num_representations: Number of output representations.
      use_positional_encoding: Whether positional encoding is applied or not.
      use_projection_layer: Whether to apply projection before using parametric
        attention. Used for the SUR model to increase the number of paramaters.
      mask_zero: If true, uses zero in sequence as the mask.
    """

    super(SimpleParametricAttention, self).__init__()
    self._input_embedding_dimension = input_embedding_dimension
    self._use_positional_encoding = use_positional_encoding
    self._positional_encoding = model_utils.positional_encoding(
        max_sequence_size, input_embedding_dimension)
    self._output_dimension = output_dimension
    self._attention = tf.keras.layers.Attention(use_scale=True)
    self._num_heads = num_representations
    self._mask_zero = mask_zero

    self._reset_query_head()

    self.embedding = tf.keras.layers.Embedding(
        vocab_size,
        input_embedding_dimension,
        mask_zero=mask_zero,
        embeddings_initializer='normal')

    if use_projection_layer and self._num_heads == 1:
      # Linear layer for SUR model (to increase num_parameters).
      self._projection = tf.keras.layers.Dense(
          self._embedding_dimension, use_bias=False)
    else:
      # Identity projection
      self._projection = tf.keras.layers.Layer()

  def _reset_query_head(self):
    query_init = tf.random_uniform_initializer()
    self.query_head = tf.Variable(
        query_init([1, self._num_heads, self._output_dimension]),
        trainable=True)

  def call(self, inputs):
    """Implements the forward pass of the keras model.

    Args:
      inputs: Batch of input sequences.

    Returns:
      output: The output after applying parametric attention.
    """

    x = self.embedding(inputs)
    x_mask = self.embedding.compute_mask(inputs)
    x *= tf.math.sqrt(tf.cast(self._input_embedding_dimension, tf.float32))

    if self._use_positional_encoding:
      x += self._positional_encoding

    # TODO(nikhilmehta): consider using multi-headed self-attention layer
    output = self._attention([x, x], mask=[x_mask, x_mask])
    output = self._projection(output)
    output = self._attention([self.query_head, output], mask=[None, x_mask])
    return output

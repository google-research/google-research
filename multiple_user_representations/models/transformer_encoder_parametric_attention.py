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

"""Defines a deep parametric attention encoder with multi-head attention layers that can be used as the user tower."""

import numpy as np
import tensorflow as tf

from multiple_user_representations.models import model_utils


class ParametricAttentionEncoder(tf.keras.Model):
  """A deep parametric attention encoder that can be used as a user tower."""

  def __init__(self,
               num_mha_layers,
               num_mha_heads,
               mha_dropout,
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
      num_mha_layers: The number of layers of Multi-Headed Attention(MHA).
      num_mha_heads: The number of heads to use for MHA.
      mha_dropout: Dropout for MHA.
      output_dimension: The output dimension of the user representation.
      max_sequence_size: The maximum size of the input sequence.
      vocab_size: The vocabulary size for input tokens/items.
      input_embedding_dimension: The embedding dimension for input tokens/items.
      num_representations: Number of output representations.
      use_positional_encoding: Whether positional encoding is applied or not.
      use_projection_layer: Whether to apply projection before using parametric
        attention.
      mask_zero: If true, uses zero in sequence as the mask.
    """

    super(ParametricAttentionEncoder, self).__init__()
    self._num_mha_layers = num_mha_layers
    self._num_mha_heads = num_mha_heads
    self._mha_dropout = mha_dropout
    self._input_embedding_dimension = input_embedding_dimension
    self._use_positional_encoding = use_positional_encoding
    self._positional_encoding = model_utils.positional_encoding(
        max_sequence_size, input_embedding_dimension)
    self._output_dimension = output_dimension
    self._attention = tf.keras.layers.Attention(use_scale=True)
    self._num_heads = num_representations
    self._mask_zero = mask_zero

    self.embedding = tf.keras.layers.Embedding(
        vocab_size,
        input_embedding_dimension,
        mask_zero=mask_zero,
        name="user_tower_item_input_embedding",
        embeddings_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.1, maxval=0.1))

    self._input_projection = tf.keras.layers.Dense(
        output_dimension, use_bias=True)

    self.reset_query_head()

    self._mha_layers = []
    for _ in range(num_mha_layers):
      self._mha_layers.append(
          tf.keras.layers.MultiHeadAttention(
              num_mha_heads,
              output_dimension,
              output_shape=output_dimension,
              dropout=mha_dropout))

    if use_projection_layer:
      self._output_projection = tf.keras.layers.Dense(
          output_dimension,
          use_bias=True,
          kernel_regularizer=tf.keras.regularizers.L2(0.001))
    else:
      # Identity projection
      self._output_projection = tf.keras.layers.Layer()

  def reset_query_head(self, mean=None, stdev=None):

    head_shape = [1, self._num_heads, self._output_dimension]
    if mean is not None and stdev is not None:
      mean = np.reshape(mean, (1, 1, -1))
      stdev = np.reshape(stdev, (1, 1, -1))
      initialized_value = np.random.uniform(low=0.0, high=1.0, size=head_shape)
      query_init = (mean - stdev) + (2 * stdev) * initialized_value
      query_init = query_init.astype(np.float32)
      print(query_init.shape)
    else:
      query_init = tf.random_uniform_initializer()(head_shape)

    self.query_head = tf.Variable(query_init, trainable=True)

  def call(self, inputs, training = True):
    """Implements the forward pass of the keras model.

    Args:
      inputs: Batch of input sequences.
      training: If True, model is in training mode.

    Returns:
      output: The output after applying parametric attention.
    """

    x = self.embedding(inputs)
    x_mask = self.embedding.compute_mask(inputs)  # B x T
    x *= tf.math.sqrt(tf.cast(self._input_embedding_dimension, tf.float32))
    x = self._input_projection(x)

    if self._use_positional_encoding:
      x += self._positional_encoding

    mha_mask = tf.expand_dims(x_mask, axis=1) if x_mask is not None else None
    for mha_layer in self._mha_layers:
      mha_output = mha_layer(x, x, attention_mask=mha_mask)
      mha_output = tf.keras.activations.elu(mha_output)
      x = x + mha_output

    # Consider using output projection layer.
    output = self._output_projection(x)
    output = self._attention([self.query_head, output], mask=[None, x_mask])
    return output

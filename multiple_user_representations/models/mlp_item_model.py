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

"""Defines the MLP model that is used for item tower."""

import tensorflow as tf


class ItemModelMLP(tf.keras.Model):
  """An MLP model that can be used as an item tower."""

  def __init__(self,
               output_dimension,
               vocab_size,
               input_embedding_dimension,
               num_layers,
               dropout = 0.0):
    """Initializes the parameteric attention model.

    Args:
      output_dimension: The output dimension of the user representation.
      vocab_size: The vocabulary size for input tokens/items.
      input_embedding_dimension: The embedding dimension for input tokens/items.
      num_layers: Number of layers in the MLP.
      dropout: The dropout to apply in the hidden layers.
    """

    super(ItemModelMLP, self).__init__()
    self._input_embedding_dimension = input_embedding_dimension

    self.item_input_embedding = tf.keras.layers.Embedding(
        vocab_size,
        input_embedding_dimension,
        name="item_embedding",
        embeddings_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.1, maxval=0.1))
    self.item_model = tf.keras.Sequential()
    for layer in range(num_layers):
      activation = "elu" if layer != (num_layers - 1) else None

      self.item_model.add(
          tf.keras.layers.Dense(
              output_dimension, use_bias=True, activation=activation))

      if layer != (num_layers - 1) and dropout > 0:
        self.item_model.add(tf.keras.layers.Dropout(rate=dropout))

  def call(self, inputs):
    """Implements the forward pass of the keras model.

    Args:
      inputs: Batch of input item ids.

    Returns:
      output: The output of the item tower.
    """

    x = self.item_input_embedding(inputs)
    x *= tf.math.sqrt(tf.cast(self._input_embedding_dimension, tf.float32))
    output = self.item_model(x)

    return output

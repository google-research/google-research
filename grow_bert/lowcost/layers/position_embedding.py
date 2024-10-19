# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Keras-based positional embedding layer."""

import tensorflow as tf

from official.modeling import tf_utils


class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  This layer creates a positional embedding as described in "BERT: Pre-training
  of Deep Bidirectional Transformers for Language Understanding"
  (https://arxiv.org/abs/1810.04805).

  This layer can be set up to either create a statically shaped slice or a
  dynamically shaped slice. If `use_dynamic_slicing` is True, the input tensor
  can have a dynamic 1st dimension, while if `use_dynamic_slicing` is False the
  input size must be fixed.
  """

  def __init__(self,
               embed_dim,
               initializer="glorot_uniform",
               use_dynamic_slicing=False,
               max_sequence_length=None,
               **kwargs):
    """Initialize.

    Arguments:
      embed_dim: The dimension of input Embedding.
      initializer: The initializer to use for the embedding weights. Defaults to
        "glorot_uniform".
      use_dynamic_slicing: Whether to use the dynamic slicing path.
      max_sequence_length: The maximum size of the dynamic sequence. Only
        applicable if `use_dynamic_slicing` is True.
      **kwargs: **kwargs.
    """
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    if "dtype" not in kwargs:
      kwargs["dtype"] = "float32"

    super(PositionEmbedding, self).__init__(**kwargs)
    if use_dynamic_slicing and max_sequence_length is None:
      raise ValueError(
          "If `use_dynamic_slicing` is True, `max_sequence_length` must be set."
      )
    self.embed_dim = embed_dim
    self._max_sequence_length = max_sequence_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._use_dynamic_slicing = use_dynamic_slicing

  def get_config(self):
    config = {
        "embed_dim": self.embed_dim,
        "max_sequence_length": self._max_sequence_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
        "use_dynamic_slicing": self._use_dynamic_slicing,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    """Implements build() for the layer."""
    dimension_list = input_shape.as_list()

    if len(dimension_list) != 2:
      raise ValueError("PositionEmbedding expects a 2-dimensional input tensor "
                       "of shape [batch, sequence]")
    seq_length = dimension_list[1]

    # If we are not using dynamic slicing, we must assume that the sequence
    # length is fixed and max_sequence_length should not be specified.
    if not self._use_dynamic_slicing:
      if seq_length is None:
        raise ValueError(
            "PositionEmbedding must have `use_dynamic_slicing` set "
            "to True (and max_sequence_length set) when the "
            "sequence (1st) dimension of the input is None.")
      if self._max_sequence_length is not None:
        raise ValueError(
            "When `use_dynamic_slicing` is False, max_sequence_length should "
            "not be specified and we ought to use seq_length to get the "
            "variable shape.")

    if self._max_sequence_length is not None:
      weight_sequence_length = self._max_sequence_length
    else:
      weight_sequence_length = seq_length

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, self.embed_dim],
        initializer=self._initializer)

    super(PositionEmbedding, self).build(input_shape)

  def call(self, input_positions):
    """Implements call() for the layer."""
    batch_size, seq_len = tf_utils.get_shape_list(
        input_positions, expected_rank=2)
    flat_positions = tf.reshape(input_positions, [-1])
    position_embeddings = tf.gather(self._position_embeddings, flat_positions)
    position_embeddings = tf.reshape(position_embeddings,
                                     [batch_size, seq_len, self.embed_dim])

    if self._use_dynamic_slicing:
      position_embeddings = position_embeddings[:, :seq_len, :]

    return position_embeddings

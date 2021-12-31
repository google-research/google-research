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

"""Embedding layers for ReadTwice."""

import math
from typing import Optional, Text

import tensorflow as tf

from readtwice.layers import tensor_utils


class EmbeddingLookup(tf.keras.layers.Layer):
  """Embedding lookup layer for id tensor input.

  This layer functions very similarly to `tf.keras.layers.Embedding` except:
  1. The `use_one_hot_lookup` option can enable potentially faster TPU lookup
    via one-hot multiplication.
  2. The optional `input_mask` argument will ensure that all masked embedding
    vectors are 0.
  3. The optional `projection_size` argument makes it easy to project the
    embedding size to a different output size as done by ALBERT.
  """

  def __init__(self,
               vocab_size,
               embedding_size,
               projection_size = 0,
               initializer_range = 0.02,
               use_one_hot_lookup = False,
               name = 'embedding_lookup',
               **kwargs):
    """Init.

    Args:
      vocab_size: Size of the embedding vocabulary. Must be positive and larger
        than the maximum input id.
      embedding_size: Width of the embedding table. Must be positive.
      projection_size: If positive and different from embedding_size, the output
        from the embedding table lookup will be projected via a dense layer to
        this size.
      initializer_range: The standard deviation of the truncated normal
        initializer for initializing the embedding table.
      use_one_hot_lookup: Whether to use tf.one_hot for embedding lookup instead
        of tf.gather. Default is False, but setting to True may be more
        efficient on TPUs for vocab sizes that aren't too large.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(EmbeddingLookup, self).__init__(name=name, **kwargs)

    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.projection_size = projection_size
    self.initializer_range = initializer_range
    self.use_one_hot_lookup = use_one_hot_lookup

  def build(self, input_shape):
    """Keras build function.

    Args:
      input_shape: TensorShape of the input; unused.
    """
    self.embedding_table = self.add_weight(
        name='embedding_table',
        shape=[self.vocab_size, self.embedding_size],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.initializer_range),
        trainable=True)

    if self.projection_size > 0 and self.embedding_size != self.projection_size:
      self.embedding_projection = tf.keras.layers.Dense(
          units=self.projection_size,
          activation=None,
          use_bias=True,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=1.0 / math.sqrt(self.embedding_size)),
          bias_initializer='zeros',
          name='projection')

    super(EmbeddingLookup, self).build(input_shape)

  def call(self,
           input_ids,
           input_mask = None):
    """Calls the layer.

    Args:
      input_ids: <int>[batch_size, ...] Tensor of ids to look up. All ids must
        be between 0 (inclusive) and `self.vocab_size` (exclusive).
      input_mask: <int>[batch_size, ...] Tensor of the same shape as
        `input_ids`. Should have only 0 and 1 values, with 0 for ids to mask and
        1 otherwise. The returned embeddings for all masked ids will be 0, so
        the corresponding ids in `input_ids` are ignored.

    Returns:
      <float32>[input_ids.shape, embedding_size] Tensor of embeddings.
    """
    if input_mask is not None:
      # Make all masked ids 0 since their embeddings will be set to 0 later.
      input_ids *= input_mask

    output = (
        tensor_utils.gather_by_one_hot(self.embedding_table, input_ids) if
        self.use_one_hot_lookup else tf.gather(self.embedding_table, input_ids))

    if self.projection_size > 0 and self.embedding_size != self.projection_size:
      output = self.embedding_projection(output)

    # Zero out embeddings for masked ids if any. Generally, should be
    # the last step as some of the previous ops (e.g. projection that ends up
    # adding a bias) could potentially have altered the tensors for masked ids.
    if input_mask is not None:
      output *= tf.expand_dims(tf.cast(input_mask, dtype=output.dtype), axis=-1)

    return output

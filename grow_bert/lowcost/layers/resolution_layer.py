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

"""Pooling layer to reduce input sequence length."""

import tensorflow as tf
from tensorflow_models.official.modeling import tf_utils


class MaskPoolLayer(tf.keras.layers.Layer):
  """Mask pooling layer."""

  def __init__(self, pool_size, nocls=True, **kwargs):
    super(MaskPoolLayer, self).__init__(**kwargs)
    self.nocls = nocls
    self.pool_size = pool_size
    assert self.pool_size > 0

  def call(self, input_tensor, unpooled_len=0):
    if self.pool_size == 1:
      return input_tensor

    batch_size, seq_len = tf_utils.get_shape_list(input_tensor, expected_rank=2)
    # reshape tensor in order to use tf.nn.pool
    reshaped_tensor = tf.reshape(input_tensor, [batch_size, seq_len, 1])
    if self.nocls:
      tensor_to_pool = reshaped_tensor[:, 1:, :]
    else:
      tensor_to_pool = reshaped_tensor

    if unpooled_len > 0:
      tensor_to_pool = tensor_to_pool[:, :-unpooled_len, :]

    pooled_tensor = tf.nn.max_pool(
        tensor_to_pool,
        ksize=self.pool_size,
        strides=self.pool_size,
        padding='SAME')

    if self.nocls:
      pooled_tensor = tf.concat([reshaped_tensor[:, 0:1, :], pooled_tensor],
                                axis=1)
    if unpooled_len > 0:
      pooled_tensor = tf.concat(
          [pooled_tensor, reshaped_tensor[:, -unpooled_len:, :]], axis=1)

    pooled_tensor = tf.reshape(pooled_tensor, [batch_size, -1])
    return pooled_tensor


class EmbedPoolLayer(tf.keras.layers.Layer):
  """Embedding pooling layer."""

  def __init__(self, hidden_size, pool_size, pool_name=None, **kwargs):
    super(EmbedPoolLayer, self).__init__(**kwargs)
    self.pool_name = pool_name
    self.pool_size = pool_size
    self.hidden_size = hidden_size
    if self.pool_name == 'concat':
      self.embedding_projection_dense = tf.keras.layers.Dense(
          self.hidden_size, name='resolution/projection_dense')

  def call(self, input_tensor, unpooled_len=0):
    if self.pool_size <= 1 or self.pool_name is None:
      return input_tensor

    if self.pool_name == 'concat':
      if unpooled_len == 0:
        tensor_to_pool = input_tensor
      else:
        tensor_to_pool = input_tensor[:, :-unpooled_len, :]
    else:
      if unpooled_len == 0:
        tensor_to_pool = input_tensor[:, 1:, :]
      else:
        tensor_to_pool = input_tensor[:, 1:-unpooled_len, :]

    if self.pool_name == 'mean':
      pooled_tensor = tf.nn.avg_pool(
          tensor_to_pool,
          ksize=self.pool_size,
          strides=self.pool_size,
          padding='SAME')
      pooled_tensor = tf.concat([input_tensor[:, 0:1, :], pooled_tensor],
                                axis=1)
    elif self.pool_name == 'max':
      pooled_tensor = tf.nn.max_pool(
          tensor_to_pool,
          ksize=self.pool_size,
          strides=self.pool_size,
          padding='SAME')
      pooled_tensor = tf.concat([input_tensor[:, 0:1, :], pooled_tensor],
                                axis=1)
    elif self.pool_name == 'concat':
      batch_size, seq_len, embed_dim = tensor_to_pool.shape
      assert seq_len % self.pool_size == 0, (f'seqlen: {seq_len}, poolsize: '
                                             f'{self.pool_size}')
      pooled_len = seq_len // self.pool_size
      pooled_tensor = tf.reshape(
          tensor_to_pool, [batch_size, pooled_len, self.pool_size * embed_dim])
      pooled_tensor = self.embedding_projection_dense(pooled_tensor)
    elif self.pool_name is not None:
      raise NotImplementedError
    if unpooled_len > 0:
      pooled_tensor = tf.concat(
          [pooled_tensor, input_tensor[:, -unpooled_len:, :]], axis=1)
    return pooled_tensor

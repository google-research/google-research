# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""AutoInt architecture utils."""
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class MultiHeadLayer(Layer):
  """Multi head attention layer."""

  def __init__(self, att_embed_size=32,
               head_num=2, use_res=True,
               scaling=False, seed=1024,
               **kwargs):

    super(MultiHeadLayer, self).__init__()
    if head_num <= 0:
      raise ValueError('head_num must be a int > 0')
    self.att_embed_size = att_embed_size
    self.head_num = head_num
    self.use_res = use_res
    self.seed = seed
    self.scaling = scaling

  def build(self, input_shape):
    if len(input_shape) != 3:
      raise ValueError(
          'Unexpected inputs dimensions %d, expect to be 3 dimensions' %
          (len(input_shape)))
    embed_size = int(input_shape[-1])
    self.w_query = self.add_weight(
        name='query',
        shape=[embed_size, self.att_embed_size * self.head_num],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
    self.w_key = self.add_weight(
        name='key',
        shape=[embed_size, self.att_embed_size * self.head_num],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
    self.w_value = self.add_weight(
        name='value',
        shape=[embed_size, self.att_embed_size * self.head_num],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
    if self.use_res:
      self.w_res = self.add_weight(
          name='res',
          shape=[embed_size, self.att_embed_size * self.head_num],
          dtype=tf.float32,
          initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))

  def call(self, inputs, **kwargs):
    if K.ndim(inputs) != 3:
      raise ValueError(
          'Unexpected inputs dimensions %d, expect to be 3 dimensions' %
          (K.ndim(inputs)))
    querys = tf.tensordot(
        inputs, self.w_query, axes=(-1, 0))
    keys = tf.tensordot(inputs, self.w_key, axes=(-1, 0))
    values = tf.tensordot(inputs, self.w_value, axes=(-1, 0))
    # head_num None F D
    querys = tf.stack(tf.split(querys, self.head_num, axis=2))
    keys = tf.stack(tf.split(keys, self.head_num, axis=2))
    values = tf.stack(tf.split(values, self.head_num, axis=2))
    inner_product = tf.matmul(
        querys, keys, transpose_b=True)
    if self.scaling:
      inner_product /= self.att_embed_size ** 0.5
    self.normalized_att_scores = tf.nn.softmax(inner_product)
    result = tf.matmul(self.normalized_att_scores, values)
    result = tf.concat(
        tf.split(
            result,
            self.head_num,
        ), axis=-1)
    result = tf.squeeze(result, axis=0)
    if self.use_res:
      result += tf.tensordot(inputs, self.w_res, axes=(-1, 0))
    result = tf.nn.relu(result)
    return result

  def compute_output_shape(self, input_shape):
    return (None, input_shape[1], self.att_embed_size * self.head_num)

  def get_config(self,):
    config = {
        'att_embed_size': self.att_embed_size,
        'head_num': self.head_num,
        'use_res': self.use_res,
        'seed': self.seed
    }
    base_config = super(MultiHeadLayer, self).get_config()
    base_config.update(config)
    return base_config


class AutoInt:
  """AutoInt Architecture Main."""

  def __init__(self,
               data_path,
               embed_size=16,
               nb_multihead=3):
    self.pre_process(data_path)
    self.embed_size = embed_size
    self.nb_class = 2
    self.nb_multihead = nb_multihead

  def pre_process(self, data_path):
    data = pd.read_csv(data_path)
    cols = data.columns.values
    self.dense_feats = [f for f in cols if f[0] == 'I']
    self.sparse_feats = [f for f in cols if f[0] == 'C']

  def input_layers(self):
    self.dense_inputs = []
    for f in self.dense_feats:
      my_input = Input([1], name=f)
      self.dense_inputs.append(my_input)
    self.sparse_inputs = []
    for f in self.sparse_feats:
      my_input = Input([1], name=f.split(' ')[0])
      self.sparse_inputs.append(my_input)

  def embedding_layers(self):
    """Embedding layers for sparse and dense fratures."""
    self.dense_kd_embed = []
    for i, my_input in enumerate(self.dense_inputs):
      f = self.dense_feats[i]
      embed = tf.Variable(
          tf.random.truncated_normal(shape=(1, self.embed_size), stddev=0.01),
          name=f)
      scaled_embed = tf.expand_dims(my_input * embed, axis=1)
      self.dense_kd_embed.append(scaled_embed)

    self.sparse_kd_embed = []
    for i, my_input in enumerate(self.sparse_inputs):
      f = self.sparse_feats[i]
      voc_size = int(f.split(' ')[1])
      kd_embed = Embedding(
          voc_size + 1,
          self.embed_size,
          embeddings_regularizer=tf.keras.regularizers.l2(0.5))(
              my_input)
      self.sparse_kd_embed.append(kd_embed)

  def construct(self, cluster_per_class=500, model_type='selfclr_llp'):
    """Construct auto-int architecture."""
    self.input_layers()
    self.embedding_layers()
    embed_layer = Concatenate(axis=1)(
        self.dense_kd_embed + self.sparse_kd_embed)
    x = embed_layer
    for _ in range(self.nb_multihead):
      x = MultiHeadLayer()(x)
    multihead_layer = Flatten()(x)
    x = Flatten()(embed_layer)
    x = Concatenate(axis=1)([multihead_layer, x])
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if model_type == 'selfclr_llp':
      output_layer = Dense(
          cluster_per_class * self.nb_class,
          kernel_regularizer=tf.keras.regularizers.l2(0.0005),
          activation='relu')(
              x)
      return Model(self.dense_inputs + self.sparse_inputs, output_layer)
    if model_type == 'dllp' or model_type == 'supervised':
      out_layer = Dense(1, activation='sigmoid')(x)
      return Model(self.dense_inputs+ self.sparse_inputs, out_layer)
    return

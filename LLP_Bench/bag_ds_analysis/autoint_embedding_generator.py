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

"""Generates AutoInt Embeddings to compute bag distances."""
import functools
import random
from typing import Sequence

from absl import app
from absl import logging
import analysis_constants
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf


tfk = tf.keras
tfkl = tf.keras.layers
lapl = stats.laplace


class DenseEmbedding(tfkl.Layer):
  """Dense Embedding layer."""

  def __init__(self, embed_size, stddev, op_name):
    """Constructor."""
    super().__init__()
    self.embed_size = embed_size
    self.stddev = stddev
    self.op_name = op_name

  def build(self, input_shape):
    """Builds the layer."""
    self.embed = tf.Variable(
        tf.random.truncated_normal(
            shape=(1, self.embed_size), stddev=self.stddev
        ),
        name=self.op_name,
    )

  # pylint: disable=invalid-name
  def call(self, _input, **kwargs):
    """Calls the layer."""
    scaled_embed = tf.math.multiply(_input, self.embed)
    return scaled_embed


class MultiHeadLayer(tfkl.Layer):
  """MultiHeadLayer."""

  def __init__(
      self,
      att_embedding_size=32,
      head_num=2,
      use_res=True,
      scaling=False,
      seed=1024,
      **kwargs
  ):
    """Constructor."""
    super().__init__()
    if head_num <= 0:
      raise ValueError('head_num must be a int > 0')
    self.att_embedding_size = att_embedding_size
    self.head_num = head_num
    self.use_res = use_res
    self.seed = seed
    self.scaling = scaling

  def build(self, input_shape):
    """Builds the layer."""
    if len(input_shape) != 3:
      raise ValueError(
          'Unexpected input dimensions %d, expect 3 dimensions'
          % (len(input_shape))
      )
    embedding_size = int(input_shape[-1])
    # pylint: disable=invalid-name
    self.W_Query = self.add_weight(
        name='query',
        shape=[embedding_size, self.att_embedding_size * self.head_num],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed),
    )
    self.W_key = self.add_weight(
        name='key',
        shape=[embedding_size, self.att_embedding_size * self.head_num],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1),
    )
    self.W_Value = self.add_weight(
        name='value',
        shape=[embedding_size, self.att_embedding_size * self.head_num],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2),
    )
    if self.use_res:
      self.W_Res = self.add_weight(
          name='res',
          shape=[embedding_size, self.att_embedding_size * self.head_num],
          dtype=tf.float32,
          initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed),
      )

  def call(self, inputs, **kwargs):
    """Calls the layer."""
    if tfk.backend.ndim(inputs) != 3:
      raise ValueError(
          'Unexpected input dimensions %d, expect 3 dimensions'
          % (tfk.backend.ndim(inputs))
      )

    querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
    keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
    values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

    querys = tf.stack(tf.split(querys, self.head_num, axis=2))
    keys = tf.stack(tf.split(keys, self.head_num, axis=2))
    values = tf.stack(tf.split(values, self.head_num, axis=2))

    inner_product = tf.matmul(querys, keys, transpose_b=True)

    if self.scaling:
      inner_product /= self.att_embedding_size**0.5

    self.normalized_att_scores = tf.nn.softmax(inner_product)

    result = tf.matmul(self.normalized_att_scores, values)
    result = tf.concat(
        tf.split(
            result,
            self.head_num,
        ),
        axis=-1,
    )
    result = tf.squeeze(result, axis=0)

    if self.use_res:
      result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))

    return result

  def compute_output_shape(self, input_shape):
    """Computes the output shape."""
    return (None, input_shape[1], self.att_embedding_size * self.head_num)

  def get_config(
      self,
  ):
    """Returns the config."""
    config = {
        'att_embedding_size': self.att_embedding_size,
        'head_num': self.head_num,
        'use_res': self.use_res,
        'seed': self.seed,
    }
    base_config = super().get_config()
    base_config.update(config)
    return base_config


class AutoInt:
  """AutoInt."""

  def __init__(self, embed_size=16, nb_multihead=3):
    """Constructor."""
    self.dense_feats = [analysis_constants.I + str(i) for i in range(1, 14)]
    self.sparse_feats = [analysis_constants.C + str(i) for i in range(1, 27)]
    self.embed_size = embed_size
    self.nb_multihead = nb_multihead
    self.sparse_vocab_sizes = analysis_constants.VOCAB_SIZE_DICT

  def input_layers(self):
    """Constructs Input layers."""
    self.all_inputs = tfkl.Input(shape=(39,))

  def embedding_layers(self):
    """Constructs Embedding layers."""
    self.dense_kd_embed = []
    self.dense_layers = []
    for i, f in enumerate(self.dense_feats):
      embed_layer = DenseEmbedding(
          embed_size=self.embed_size, stddev=0.01, op_name=f
      )
      embed_layer.build(input_shape=(1, self.embed_size))
      # pylint: disable=invalid-name
      _input = self.all_inputs[:, i]
      scaled_embed = embed_layer(tf.reshape(_input, [-1, 1]))
      self.dense_kd_embed.append(tf.expand_dims(scaled_embed, axis=1))
      self.dense_layers.append(embed_layer)
    self.sparse_kd_embed = []
    self.sparse_layers = []
    for i, f in enumerate(self.sparse_feats):
      voc_size = self.sparse_vocab_sizes[f]
      # pylint: disable=invalid-name
      _input = self.all_inputs[:, i + 13]
      embed_layer = tfkl.Embedding(
          voc_size,
          self.embed_size,
          embeddings_regularizer=tf.keras.regularizers.l2(0.5),
      )
      kd_embed = embed_layer(_input)
      self.sparse_kd_embed.append(tf.expand_dims(kd_embed, axis=1))
      self.sparse_layers.append(embed_layer)

  def construct(self):
    """Constructs the model."""
    self.input_layers()
    self.embedding_layers()
    embed_layer = tfkl.Concatenate(axis=1)(
        self.dense_kd_embed + self.sparse_kd_embed
    )

    x = embed_layer

    for _ in range(self.nb_multihead):
      x = MultiHeadLayer()(x)
    multihead_layer = tfkl.Flatten()(x)

    x = tfkl.Flatten()(embed_layer)
    x = tfkl.Concatenate(axis=1)([multihead_layer, x])

    x = tfkl.Dense(256, activation='relu')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(256, activation='relu')(x)
    x = tfkl.BatchNormalization()(x)

    output_layer = tfkl.Dense(1, activation='sigmoid')(x)
    model = tfk.models.Model(self.all_inputs, output_layer)

    return model


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  list_of_cols_test = analysis_constants.LIST_OF_COLS_TEST
  feature_cols = analysis_constants.FEATURE_COLS
  criteo_df = pd.read_csv(
      '../data/preprocessed_dataset/preprocessed_criteo.csv',
      usecols=list_of_cols_test,
  )
  train_df, test_df = train_test_split(criteo_df, test_size=0.2, random_state=1)
  # pylint: disable=invalid-name
  X_test = test_df.drop(['label'], axis=1)
  Y_test = test_df['label']

  BATCH_SIZE = 1024
  num_epochs = 100

  def generate_indices_seq_randorder(steps_per_epoch, BATCH_SIZE, len_df):
    """Generates random order indices."""
    list_to_choose_from = np.arange(len_df).tolist()
    random.shuffle(list_to_choose_from)
    list_of_indices = list_to_choose_from[0 : BATCH_SIZE * steps_per_epoch]
    return list_of_indices

  def instance_batch_xy_gen_seq_randorder(BATCH_SIZE, df):
    """Generator function to construct the dataset."""
    length_of_df = len(df)
    steps_per_epoch = int(length_of_df / BATCH_SIZE)
    list_of_indices = generate_indices_seq_randorder(
        steps_per_epoch, BATCH_SIZE, length_of_df
    )
    shuffled_df_bags = df.iloc[list_of_indices].reset_index(drop=True)
    total_num_steps = int(length_of_df / BATCH_SIZE)
    batch_no = 0
    while batch_no < total_num_steps:
      rows = shuffled_df_bags.iloc[
          np.arange(batch_no * BATCH_SIZE, (batch_no + 1) * BATCH_SIZE)
      ]

      x = rows[feature_cols].to_numpy(dtype='int32')
      y = rows[['label']].to_numpy(dtype='float32')

      yield x, y
      batch_no = batch_no + 1

  # pylint: disable=unused-argument
  def lr_scheduler(x):
    """Learning rate scheduler."""
    return 1e-5

  opt = tfk.optimizers.SGD(learning_rate=1e-5)
  bce = tf.keras.losses.BinaryCrossentropy(
      reduction=tf.keras.losses.Reduction.SUM
  )

  autoint = AutoInt()
  model = autoint.construct()

  model.compile(
      optimizer=opt, loss=bce, metrics=[tf.keras.metrics.AUC(name='auc')]
  )

  reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

  logging.info('Training Started')

  f = functools.partial(
      instance_batch_xy_gen_seq_randorder, BATCH_SIZE, train_df
  )
  bag_batch_xy_generator_seq_randorder = tf.data.Dataset.from_generator(
      f,
      output_signature=(
          tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.int32),
          tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
      ),
  )

  model.fit(
      bag_batch_xy_generator_seq_randorder,
      batch_size=BATCH_SIZE,
      epochs=num_epochs,
      validation_data=(X_test, Y_test),
      validation_batch_size=1024,
      callbacks=[reduce_lr],
  )

  result_dir = '../results/autoint_embeddings/'

  # pylint: disable=logging-not-lazy
  for col_name, embed_layer in zip(autoint.dense_feats, autoint.dense_layers):
    filename = result_dir + col_name + '_embeddings.npy'
    logging.info(col_name + ' ' + str(embed_layer.get_weights()[0].shape))
    np.save(filename, embed_layer.get_weights()[0])

  for col_name, embed_layer in zip(autoint.sparse_feats, autoint.sparse_layers):
    filename = result_dir + col_name + '_embeddings.npy'
    logging.info(
        col_name
        + ' '
        + str(autoint.sparse_vocab_sizes[col_name])
        + ' '
        + str(embed_layer.get_weights()[0].shape)
    )
    np.save(filename, embed_layer.get_weights()[0])


if __name__ == '__main__':
  app.run(main)

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
"""Encoders that embed observations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


class MLPEncoder(tf.keras.Model):
  """Encoder that uses a 1 layer MLP."""

  def __init__(self,
               embedding_dim,
               name='MLP_Encoder',
               subtract_neighboring_observation=False):
    """Initializes the MLD encoder.

    Args:
      embedding_dim: size of the final observation embedding
      name: optional name for the name scope
      subtract_neighboring_observation: subtract second observation from the
        first observation
    """
    super(MLPEncoder, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._subtract_neighboring_observation = subtract_neighboring_observation
    self._fc1 = tf.keras.layers.Dense(256)
    self._fc2 = tf.keras.layers.Dense(embedding_dim)
    self._dropout1 = tf.keras.layers.Dropout(0.5)

  def call(self, x):
    x = tf.nn.relu(self._fc1(x))
    x = self._fc2(x)
    x = self._dropout1(x)
    return x

  def preprocess(self, x):
    if self._subtract_neighboring_observation:
      x_1, x_2 = tf.split(x, 2, axis=1)
      return tf.squeeze(x_1 - x_2, axis=1)
    else:
      return x


class CNNEncoder(tf.keras.Model):
  """Encoder that uses Convnets."""

  def __init__(self, embedding_dim, width=64, name='CNN_Encoder'):
    super(CNNEncoder, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._conv2 = tf.keras.layers.Conv2D(
        width, 3, 1, padding='same', activation='relu')
    self._bn2 = tf.keras.layers.BatchNormalization()
    self._conv3 = tf.keras.layers.Conv2D(
        width * 2, 3, 2, padding='same', activation='relu')
    self._bn3 = tf.keras.layers.BatchNormalization()
    self._conv4 = tf.keras.layers.Conv2D(
        width * 4, 3, 2, padding='same', activation='relu')
    self._bn4 = tf.keras.layers.BatchNormalization()
    self._conv5 = tf.keras.layers.Conv2D(embedding_dim, 1, 1, padding='same')
    self._reshape = tf.keras.layers.Reshape((-1, embedding_dim))

  def call(self, x, training=None):
    x_1, x_2 = x[:, 0, :, :, :], x[:, 1, :, :, :]
    feature = tf.concat([x_1, x_2 - x_1], axis=-1)
    x = self._conv2(feature)
    x = self._conv3(x)
    x = self._conv4(x)

    posx, posy = tf.meshgrid(
        tf.linspace(-1., 1., num=16), tf.linspace(-1., 1., num=16))
    stacked_pos = tf.expand_dims(tf.stack([posx, posy], axis=-1), axis=0)
    stacked_pos = tf.tile(stacked_pos, [tf.shape(x)[0], 1, 1, 1])
    x = tf.concat([x, stacked_pos], axis=-1)
    x = self._conv5(x)
    return self._reshape(x)

  def preprocess(self, x):
    return x


class CNNEncoderSingleFrame(tf.keras.Model):
  """Encoder that uses Convnets for a single frame of observation."""

  def __init__(self, embedding_dim, width=64, name='CNN_Encoder'):
    super(CNNEncoderSingleFrame, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._conv2 = tf.keras.layers.Conv2D(
        width, 3, 1, padding='same', activation='relu')
    self._bn2 = tf.keras.layers.BatchNormalization()
    self._conv3 = tf.keras.layers.Conv2D(
        width * 2, 3, 2, padding='same', activation='relu')
    self._bn3 = tf.keras.layers.BatchNormalization()
    self._conv4 = tf.keras.layers.Conv2D(
        width * 4, 3, 2, padding='same', activation='relu')
    self._bn4 = tf.keras.layers.BatchNormalization()
    self._conv5 = tf.keras.layers.Conv2D(embedding_dim, 1, 1, padding='same')
    self._reshape = tf.keras.layers.Reshape((-1, embedding_dim))

  def call(self, x, training=None):
    x = self._conv2(x)
    x = self._conv3(x)
    x = self._conv4(x)
    posx, posy = tf.meshgrid(
        tf.linspace(-1., 1., num=16), tf.linspace(-1., 1., num=16))
    stacked_pos = tf.expand_dims(tf.stack([posx, posy], axis=-1), axis=0)
    stacked_pos = tf.tile(stacked_pos, [tf.shape(x)[0], 1, 1, 1])
    x = tf.concat([x, stacked_pos], axis=-1)
    x = self._conv5(x)
    return self._reshape(x)

  def preprocess(self, x):
    return x

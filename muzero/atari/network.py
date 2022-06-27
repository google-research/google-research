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

# pylint: disable=g-complex-comprehension
# pylint: disable=missing-docstring

import tensorflow as tf
from tensorflow.keras import layers

from muzero import network

LARGE_NUM = 1e9


class MLPandLSTM(network.AbstractEncoderandLSTM):
  """Conv+LSTM network for use with MuZero."""

  def __init__(self,
               trivial_encoding,
               observation_space,
               *args,
               encoder_size=3,
               pretrain_temperature=1.,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.trivial_encoding = trivial_encoding
    self.pretrain_temperature = 1.

    if encoder_size == 0:
      encoding_layers = [
          layers.Conv2D(
              filters=32,
              kernel_size=8,
              strides=(4, 4),
              padding='valid',
              activation='relu',
              batch_input_shape=(None, *observation_space)),
          layers.Conv2D(
              filters=64,
              kernel_size=4,
              strides=(2, 2),
              padding='valid',
              activation=None,
              use_bias=False,
          ),
          tf.keras.layers.LayerNormalization(),
          tf.keras.layers.ReLU(),
          layers.Conv2D(
              filters=128,
              kernel_size=4,
              strides=(2, 2),
              padding='valid',
              activation='relu',
          ),
          layers.Conv2D(
              filters=256,
              kernel_size=3,
              strides=(1, 1),
              padding='valid',
              activation=None,
              use_bias=False,
          ),
          tf.keras.layers.LayerNormalization(),
          tf.keras.layers.ReLU(),
      ]
    else:
      encoding_layers = [
          layers.Conv2D(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding='same',
              activation='relu',
              batch_input_shape=(None, *observation_space)),  # add activation?
      ]
      if encoder_size > 0:
        encoding_layers.append(ResidualBlock(64),)
        if encoder_size > 1:
          encoding_layers.append(ResidualBlock(64),)
      encoding_layers.append(
          layers.Conv2D(
              filters=128,
              kernel_size=3,
              strides=(2, 2),
              activation='relu',
              padding='same'),  # add activation?
      )
      if encoder_size > 0:
        encoding_layers.append(ResidualBlock(128),)
        if encoder_size > 1:
          encoding_layers.append(ResidualBlock(128),)
          if encoder_size > 2:
            encoding_layers.append(ResidualBlock(128),)
      encoding_layers.append(
          layers.AveragePooling2D(
              pool_size=(3, 3), strides=(2, 2), padding='same'),)
      if encoder_size > 0:
        encoding_layers.append(ResidualBlock(128),)
        if encoder_size > 1:
          encoding_layers.append(ResidualBlock(128),)
          if encoder_size > 2:
            encoding_layers.append(ResidualBlock(128),)
      encoding_layers.append(
          layers.AveragePooling2D(
              pool_size=(3, 3), strides=(2, 2), padding='same'))

    self._observation_encoder = tf.keras.Sequential(
        encoding_layers, name='observation_encoder')

    pretrain_hidden_layers = self._head_hidden_layers()
    pretrain_output_size = self.head_hidden_sizes[
        -1] if self.head_hidden_sizes else self.hidden_state_size
    self._pretrain_head = tf.keras.Sequential(
        pretrain_hidden_layers + [
            layers.Dense(pretrain_output_size, name='pretrain_output'),
        ],
        name='pretrain_head')
    self._pretrain_predictor = tf.keras.Sequential([
        tf.keras.layers.Dense(pretrain_output_size // 4, use_bias=False),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(pretrain_output_size),
    ],
                                                   name='pretrain_predictor')

  def _encode_observation(self, observation, training=True):
    observation = observation * 2 - 1.
    if self.trivial_encoding:
      # use the trivial observation encoding from
      # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5.
      # Simply take the difference between the last two observations.
      return observation[:, :, :, -1] - observation[:, :, :, -2]
    return self._observation_encoder(observation, training=training)

  # The loss is according to SimCLR(https://arxiv.org/abs/2002.05709).
  def pretraining_loss(self, sample, training=True):
    obs1, obs2 = sample
    out1 = self._pretrain_head(
        self.initial_inference(obs1, training=training).hidden_state)
    out2 = self._pretrain_head(
        self.initial_inference(obs2, training=training).hidden_state)
    pred1 = self._pretrain_predictor(out1)
    pred2 = self._pretrain_predictor(out2)
    loss = self.add_contrastive_loss(
        pred1, out2) / 2. + self.add_contrastive_loss(pred2, out1) / 2.

    return loss, None

  def add_contrastive_loss(self,
                           hidden1,
                           hidden2,
                           hidden_norm=True,
                           weights=1.0):
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
      hidden1 = tf.math.l2_normalize(hidden1, -1)
      hidden2 = tf.math.l2_normalize(hidden2, -1)
    batch_size = tf.shape(hidden1)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(
        hidden1, hidden1, transpose_b=True) / self.pretrain_temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(
        hidden2, hidden2, transpose_b=True) / self.pretrain_temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(
        hidden1, hidden2, transpose_b=True) / self.pretrain_temperature
    logits_ba = tf.matmul(
        hidden2, hidden1, transpose_b=True) / self.pretrain_temperature
    logits_a = tf.concat([logits_ab, logits_aa], 1)
    logits_b = tf.concat([logits_ba, logits_bb], 1)

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits_a)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits_b)
    loss = loss_a + loss_b

    return loss

  def get_pretraining_trainable_variables(self):
    return (self._observation_encoder.trainable_variables +
            self._to_hidden.trainable_variables +
            self._pretrain_head.trainable_variables +
            self._pretrain_predictor.trainable_variables)


class ResidualBlock(layers.Layer):
  """Residualblock.

  Implementation adapted from:
  https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a
  .

  """

  def __init__(self, planes):
    super(ResidualBlock, self).__init__(name='')
    self.planes = planes

    self.conv2a = layers.Conv2D(
        filters=self.planes,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        use_bias=False)
    self.bn2a = layers.LayerNormalization()

    self.conv2b = layers.Conv2D(
        filters=self.planes,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        use_bias=False)
    self.bn2b = layers.LayerNormalization()
    self.relu = layers.ReLU()

  def __call__(self, input_tensor, training=True, **kwargs):

    x = self.conv2a(input_tensor, training=training)
    x = self.bn2a(x, training=training)
    x = self.relu(x)

    x = self.conv2b(x, training=training)
    x = self.bn2b(x, training=training)

    x += input_tensor
    return self.relu(x)

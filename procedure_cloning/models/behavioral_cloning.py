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

"""Vanilla, Aux, and Aug BC."""

import numpy as np
import tensorflow as tf
from procedure_cloning import utils


class BehavioralCloning(tf.keras.Model):
  """BC with neural nets."""

  def __init__(self,
               maze_size,
               num_actions=4,
               encoder_size='default',
               augment=True,
               encode_dim=256,
               aux_weight=0.,
               learning_rate=1e-3):
    super().__init__()

    self._maze_size = maze_size
    self._num_actions = num_actions
    self._encode_dim = encode_dim
    self._augment = augment
    self._aux_weight = aux_weight

    if encoder_size == 'default':
      kernel_sizes = (3,) * 5
      stride_sizes = (1,) * 5
      pool_sizes = (2, None, 2, None, None)
      num_filters = (encode_dim / 2,) * 2 + (encode_dim,) * 3
    else:
      raise NotImplementedError

    self._encoder = utils.create_conv(
        [self._maze_size, self._maze_size, 3],  # (wall; goal; loc)
        kernel_sizes=kernel_sizes,
        stride_sizes=stride_sizes,
        pool_sizes=pool_sizes,
        num_filters=num_filters,
        output_dim=encode_dim)

    self._policy = utils.create_mlp(encode_dim, self._num_actions)
    self._action_network = utils.create_mlp(encode_dim, maze_size * maze_size * num_actions)

    if self._augment:
      self._augment_layers = tf.keras.Sequential([
          tf.keras.layers.RandomCrop(maze_size, maze_size),
          tf.keras.layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1), fill_mode='constant'),
          tf.keras.layers.RandomZoom((-0.1, 0.1), (-0.1, 0.1), fill_mode='constant'),
      ])

    self._optimizer = tf.keras.optimizers.Adam(learning_rate)

  def process_states(self, observations, maze_maps, training=True):
    """Returns [B, W, W, 3] binary values. Channels are (wall; goal; obs)"""
    loc = tf.one_hot(
        tf.cast(observations[:, 0] * self._maze_size + observations[:, 1], tf.int32),
        self._maze_size * self._maze_size)
    loc = tf.reshape(loc, [tf.shape(observations)[0], self._maze_size, self._maze_size])
    maze_maps = tf.cast(maze_maps, tf.float32)
    states = tf.concat([maze_maps, loc[Ellipsis, None]], axis=-1)
    if self._augment and training:
      states = self._augment_layers(states)
    return states

  def embed_states(self, states):
    return self._encoder(states)

  @tf.function
  def call(self, dataset_iter, training=True, generate=False):
    observations, actions, maze_maps, value_maps = next(dataset_iter)

    states = self.process_states(observations, maze_maps, training=training)

    with tf.GradientTape() as tape:
      embed = self.embed_states(states)
      logit = self._policy(embed)
      pred_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=actions, logits=logit)
      pred = tf.math.argmax(logit, axis=-1, output_type=tf.dtypes.int32)
      acc = tf.reduce_sum(tf.cast(pred == actions, tf.float32)
                          ) / tf.cast(tf.shape(pred)[0], tf.float32)

      value_maps = tf.reshape(value_maps, [-1, self._num_actions])
      valid_indices = tf.where(tf.reduce_sum(value_maps, axis=-1) == 1)
      value_maps = tf.gather(value_maps, valid_indices)
      logit = self._action_network(embed)
      logit = tf.reshape(logit, [-1, self._num_actions])
      logit = tf.gather(logit, valid_indices)
      aux_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=value_maps, logits=logit)

      loss = tf.reduce_mean(pred_loss) + self._aux_weight * tf.reduce_mean(aux_loss)
    if training:
      grads = tape.gradient(loss, self.trainable_variables)
      self._optimizer.apply_gradients(zip(grads, self.trainable_variables))

    return {'loss': loss,
            'pred_loss': tf.reduce_mean(pred_loss),
            'aux_loss': tf.reduce_mean(aux_loss),
            'acc': acc
            }

  @tf.function
  def act(self, observation, maze_map):
    observations = tf.convert_to_tensor([observation])
    maze_maps = tf.convert_to_tensor([maze_map])
    states = self.process_states(observations, maze_maps)
    embed = self.embed_states(states)
    logit = self._policy(embed)
    return tf.random.categorical(logit, 1, dtype=tf.int32)[:, 0]

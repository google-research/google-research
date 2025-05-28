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

"""Procedure cloning with BFS."""

import numpy as np
import tensorflow as tf
from procedure_cloning import utils


class BehavioralCloningBFS(tf.keras.Model):
  """BC with neural nets."""

  def __init__(self,
               maze_size,
               num_actions=4,
               encoder_size='default',
               augment=False,
               encode_dim=256,
               learning_rate=1e-3):
    super().__init__()

    self._maze_size = maze_size
    self._num_actions = num_actions
    self._encode_dim = encode_dim
    self._augment = augment

    if encoder_size == 'default':
      kernel_sizes = (3,) * 5
      stride_sizes = (1,) * 5
      pool_sizes = None
      num_filters = (encode_dim / 2,) * 2 + (encode_dim,) * 2 + (
          self._num_actions + 1,)
    else:
      raise NotImplementedError

    self._encoder = utils.create_conv(
        [self._maze_size, self._maze_size, 8],  # (wall; goal; loc; a)
        kernel_sizes=kernel_sizes,
        stride_sizes=stride_sizes,
        pool_sizes=pool_sizes,
        num_filters=num_filters,
        activation_last_layer=False,
    )

    if self._augment:
      self._augment_layers = tf.keras.Sequential([
          tf.keras.layers.RandomCrop(maze_size, maze_size),
          tf.keras.layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1),
                                            fill_mode='constant'),
          tf.keras.layers.RandomZoom((-0.1, 0.1), (-0.1, 0.1),
                                     fill_mode='constant'),
      ])

    self._optimizer = tf.keras.optimizers.Adam(learning_rate)

  def process_states(self, observations, maze_maps, training=True):
    """Returns [B, W, W, 3] binary values. Channels are (wall; goal; obs)"""
    loc = tf.one_hot(
        tf.cast(observations[:, 0] * self._maze_size + observations[:, 1],
                tf.int32),
        self._maze_size * self._maze_size,
        dtype=tf.int32)
    loc = tf.reshape(
        loc, [tf.shape(observations)[0], self._maze_size, self._maze_size])
    states = tf.cast(
        tf.concat([maze_maps, loc[Ellipsis, None]], axis=-1), tf.float32)
    if self._augment and training:
      states = self._augment_layers(states)
    return states

  @tf.function
  def call(self, dataset_iter, training=True, generate=False):
    observations, maze_maps, bfs_input_maps, bfs_output_maps = next(
        dataset_iter)
    maze_maps = tf.cast(maze_maps, tf.int32)
    with tf.GradientTape() as tape:
      # Potential data augmentation for bfs maps
      bfs_maze_maps = tf.concat(
          [bfs_input_maps[Ellipsis, None], bfs_output_maps[Ellipsis, None], maze_maps],
          axis=-1)
      bfs_input_maps, bfs_output_maps, states = tf.split(
          self.process_states(observations, bfs_maze_maps, training=training),
          [1, 1, -1],
          axis=-1)
      bfs_input_maps = tf.cast(bfs_input_maps[Ellipsis, 0], tf.int32)
      bfs_output_maps = tf.cast(bfs_output_maps[Ellipsis, 0], tf.int32)
      bfs_input_onehot = tf.one_hot(
          bfs_input_maps, self._num_actions + 1, dtype=tf.float32)
      bfs_states = tf.concat([states, bfs_input_onehot], axis=-1)
      logits = self._encoder(bfs_states)
      logits = tf.reshape(logits, [-1, self._num_actions + 1])
      labels = tf.reshape(bfs_output_maps, [-1])

      pred_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      preds = tf.math.argmax(logits, axis=-1, output_type=tf.dtypes.int32)
      acc = tf.reduce_sum(tf.cast(preds == labels, tf.float32)) / tf.cast(
          tf.shape(preds)[0], tf.float32)
      loss = tf.reduce_mean(pred_loss)
    if training:
      grads = tape.gradient(loss, self.trainable_variables)
      self._optimizer.apply_gradients(zip(grads, self.trainable_variables))

    info_dict = {
        'loss': loss,
        'acc': acc,
    }
    return info_dict

  @tf.function
  def act(self, observation, maze_map, max_len=100):
    maze_map = tf.cast(maze_map, tf.int32)
    observations = tf.convert_to_tensor([observation])
    maze_maps = tf.convert_to_tensor([maze_map])
    observation = tf.cast(observation, tf.int32)

    states = self.process_states(observations, maze_maps)
    bfs_input_maps = self._num_actions * tf.ones(
        [1, self._maze_size, self._maze_size], dtype=tf.int32)

    i = 0
    while tf.gather_nd(bfs_input_maps[0],
                       observation) == self._num_actions and i < max_len:
      bfs_input_onehot = tf.one_hot(
          bfs_input_maps, self._num_actions + 1, dtype=tf.float32)
      bfs_states = tf.concat([states, bfs_input_onehot], axis=-1)
      logits = self._encoder(bfs_states)
      logits = tf.reshape(
          logits, [-1, self._maze_size, self._maze_size, self._num_actions + 1])
      bfs_input_maps = tf.math.argmax(
          logits, axis=-1, output_type=tf.dtypes.int32)
      i += 1
    action = tf.gather_nd(bfs_input_maps[0], observation)
    if action == self._num_actions:
      action = tf.random.uniform([1], 0, self._num_actions, dtype=tf.int32)[0]
    return action

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

# Lint as: python3
"""Neural network models and checkpointing helpers for the agent."""

import random
from absl import logging
import tensorflow.compat.v2 as tf

EPS = 1e-9


class RandomizingConv2d(tf.keras.layers.Conv2D):

  def call(self, x):
    if self.built:
      self.kernel.assign(self.kernel_initializer(self.kernel.shape))
    return super().call(x)


class RandConv(tf.keras.layers.Layer):
  """RandConv layer from Network Randomization Paper."""

  def __init__(self, alpha=0.1, kernel_size=2, num_channels=1, **kwargs):
    super().__init__()
    self._alpha = alpha
    self._kernel_size = kernel_size
    self._num_channels = num_channels
    self.rand_conv = RandomizingConv2d(
        filters=num_channels, kernel_size=kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer=tf.initializers.glorot_normal(),
        trainable=False,
        name='randcnn', **kwargs)

  def rand_output(self, x):
    num_splits = x.shape[-1] // self._num_channels
    x = tf.concat(tf.split(x, num_splits, axis=-1), axis=0)
    x = self.rand_conv(x)
    x = tf.concat(tf.split(x, num_splits, axis=0), axis=-1)
    return x

  def call(self, x):
    if random.random() < self._alpha:
      return x
    else:
      return self.rand_output(x)


def create_checkpoint_manager(model,
                              ckpt_dir,
                              step,
                              optimizer=None,
                              restore=False,
                              max_to_keep=1):
  """Helper function for checkpointing."""
  objects_to_save = dict(model=model, step=step)
  if optimizer:
    objects_to_save.update(optimizer=optimizer)
  checkpoint = tf.train.Checkpoint(**objects_to_save)
  manager = tf.train.CheckpointManager(
      checkpoint, directory=ckpt_dir, max_to_keep=max_to_keep)
  if restore:
    if manager.latest_checkpoint is not None:
      status = checkpoint.restore(manager.latest_checkpoint)
      logging.info('Loaded checkpoint %s', manager.latest_checkpoint)
      status.assert_existing_objects_matched()
  return manager


class JumpyWorldNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's policy."""

  def __init__(self,
               num_actions,
               name = None,
               dropout = 0.0,
               rand_conv = False,
               projection = True,
               **kwargs):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: number of actions.
      name: used to create scope for network parameters.
      dropout: Dropout rate.
      rand_conv: Whether to use rand_conv or not.
      projection: Whether to use projected representation or not.
      **kwargs: Arbitrary keyword arguments.
    """
    super(JumpyWorldNetwork, self).__init__(**kwargs)
    self.kwargs = kwargs
    self._dropout = dropout
    self._num_actions = num_actions
    if rand_conv:
      self.rand_conv = RandConv()
    else:
      self.rand_conv = None
    self._projection = projection
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    self.conv0 = tf.keras.layers.Conv2D(
        32, [8, 8],
        strides=4,
        padding='same',
        activation=activation_fn,
        name='Conv')
    self.conv1 = tf.keras.layers.Conv2D(
        64, [4, 4],
        strides=2,
        padding='same',
        activation=activation_fn,
        name='Conv1')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [3, 3],
        strides=1,
        padding='same',
        activation=activation_fn,
        name='Conv2')
    self.flatten = tf.keras.layers.Flatten()
    self.dense0 = tf.keras.layers.Dense(256, activation=activation_fn)
    self.dense1 = tf.keras.layers.Dense(64, activation=activation_fn)
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

  @tf.function
  def call(self, state, training=True):
    """Creates the output tensor/op given the state tensor as input.

    Args:
      state: tf.Tensor, Input tensor.
      training: boolean, indicating whether to use dropout or not.

    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = self.representation(state, projection=False)
    if training:
      x = tf.nn.dropout(x, rate=self._dropout)
    x = self.dense2(x)
    if self._num_actions == 1:
      x = tf.squeeze(x, axis=-1)
    return x

  def representation(self, state, projection=True):
    x = tf.cast(state, tf.float32)
    if self.rand_conv is not None:
      x = self.rand_conv(x)
    x = self.conv0(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.dense0(x)
    if projection and self._projection:
      x = self.dense1(x)
    return x

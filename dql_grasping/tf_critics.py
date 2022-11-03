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

"""Critic function approximators in TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import gin
from six.moves import range
import tensorflow.compat.v1 as tf
from dql_grasping import tf_modules
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


@gin.configurable
def cnn_ia_v1(state,
              action,
              scope,
              channels=32,
              num_convs=3,
              is_training=True,
              reuse=False,
              use_timestep=True):
  """CNN architecture for DQN that takes action as vector-encoded input.

  Args:
    state: 2-Tuple of image and timestep tensors: (image, timestep).
    action: Tensor of proposed actions.
    scope: String name of the TF variable scope.
    channels: Number of channels in each layer.
    num_convs: Number of convolution channels to apply to the state.
    is_training: Whether this graph is for training or inference.
    reuse: Whether or not to reuse variables from this variable scope.
    use_timestep: If True, incorporate timestep into model prediction.

  Returns:
    Tensor of size (batch_size,) representing Q(s, a).
  """
  net, timestep = state
  end_points = {}
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    with slim.arg_scope(tf_modules.argscope(is_training=is_training)):
      for layer_index in range(num_convs):
        net = slim.conv2d(net, channels, kernel_size=3)
        logging.info('conv%d %s', layer_index, net.get_shape())
      if use_timestep:
        _, height, width, _ = net.get_shape().as_list()
        timestep = tf.cast(timestep, tf.float32)
        timestep = tf.tile(tf.reshape(timestep, [-1, 1, 1, 1]),
                           [1, height, width, 1])
        net = tf.concat([net, timestep], axis=3)
        # Process Action
        context = slim.fully_connected(action, channels + 1)
      else:
        context = slim.fully_connected(action, channels)
      net = tf_modules.add_context(net, context)
      net = tf.layers.flatten(net)
      net = slim.stack(net, slim.fully_connected, [channels, channels])
      net = slim.fully_connected(net,
                                 num_outputs=1,
                                 normalizer_fn=None,
                                 weights_regularizer=None,
                                 activation_fn=None)
      return tf.squeeze(net, 1), end_points


@gin.configurable
def cnn_v0(state,
           num_actions,
           scope,
           channels=32,
           activation_fn=None,
           is_training=True,
           reuse=False,
           use_timestep=True):
  """CNN architecture for discrete-output DQN.

  Args:
    state: 2-Tuple of image and timestep tensors: (image, timestep).
    num_actions: (int) Number of discrete actions.
    scope: String name of the TF variable scope.
    channels: Number of channels in each layer.
    activation_fn: Python function specifying activation of final layer. Can be
      used to implement action clipping for DDPG Actors.
    is_training: Whether this graph is for training or inference.
    reuse: Whether or not to reuse variables from this variable scope.
    use_timestep: If True, incorporate timestep into model prediction.

  Returns:
    Tensor of size (batch_size, num_actions) representing Q(s, a) for each
    action.
  """
  if use_timestep:
    net, timestep = state
  else:
    net = state
  end_points = {}
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    with slim.arg_scope(tf_modules.argscope(is_training=is_training)):
      for layer_index in range(3):
        net = slim.conv2d(net, channels, kernel_size=3)
        logging.info('conv%d %s', layer_index, net.get_shape())
      if use_timestep:
        _, height, width, _ = net.get_shape().as_list()
        timestep = tf.cast(timestep, tf.float32)
        timestep = tf.tile(tf.reshape(timestep, [-1, 1, 1, 1]),
                           [1, height, width, 1])
        net = tf.concat([net, timestep], axis=3)
      net = tf.layers.flatten(net)
      net = slim.stack(net, slim.fully_connected, [channels, channels])
      net = slim.fully_connected(net,
                                 num_outputs=num_actions,
                                 normalizer_fn=None,
                                 weights_regularizer=None,
                                 activation_fn=activation_fn)
      return net, end_points

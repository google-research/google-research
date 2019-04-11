# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Networks for the PPO algorithm defined as recurrent cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_MEAN_WEIGHTS_INITIALIZER = tf.contrib.layers.variance_scaling_initializer(
    factor=0.1)
_LOGSTD_INITIALIZER = tf.random_normal_initializer(-1, 1e-10)

class LinearGaussianPolicy(tf.contrib.rnn.RNNCell):
  """Indepent linear network with a tanh at the end for policy and feedforward network for the value.

  The policy network outputs the mean action and the log standard deviation
  is learned as indepent parameter vector.
  """

  def __init__(self,
               policy_layers,
               value_layers,
               action_size,
               mean_weights_initializer=_MEAN_WEIGHTS_INITIALIZER,
               logstd_initializer=_LOGSTD_INITIALIZER):
    self._policy_layers = policy_layers
    self._value_layers = value_layers
    self._action_size = action_size
    self._mean_weights_initializer = mean_weights_initializer
    self._logstd_initializer = logstd_initializer

  @property
  def state_size(self):
    unused_state_size = 1
    return unused_state_size

  @property
  def output_size(self):
    return (self._action_size, self._action_size, tf.TensorShape([]))

  def __call__(self, observation, state):
    with tf.variable_scope('policy'):
      x = tf.contrib.layers.flatten(observation)
      mean = tf.contrib.layers.fully_connected(
          x,
          self._action_size,
          tf.tanh,
          weights_initializer=self._mean_weights_initializer)
      logstd = tf.get_variable('logstd', mean.shape[1:], tf.float32,
                               self._logstd_initializer)
      logstd = tf.tile(logstd[None, ...],
                       [tf.shape(mean)[0]] + [1] * logstd.shape.ndims)
    with tf.variable_scope('value'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[:, 0]
    return (mean, logstd, value), state


class ForwardGaussianPolicy(tf.contrib.rnn.RNNCell):
  """Independent feed forward networks for policy and value.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector.
  """

  def __init__(
      self, policy_layers, value_layers, action_size,
      mean_weights_initializer=_MEAN_WEIGHTS_INITIALIZER,
      logstd_initializer=_LOGSTD_INITIALIZER):
    self._policy_layers = policy_layers
    self._value_layers = value_layers
    self._action_size = action_size
    self._mean_weights_initializer = mean_weights_initializer
    self._logstd_initializer = logstd_initializer

  @property
  def state_size(self):
    unused_state_size = 1
    return unused_state_size

  @property
  def output_size(self):
    return (self._action_size, self._action_size, tf.TensorShape([]))

  def __call__(self, observation, state):
    with tf.variable_scope('policy'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._policy_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      mean = tf.contrib.layers.fully_connected(
          x, self._action_size, tf.tanh,
          weights_initializer=self._mean_weights_initializer)
      logstd = tf.get_variable(
          'logstd', mean.shape[1:], tf.float32, self._logstd_initializer)
      logstd = tf.tile(
          logstd[None, ...], [tf.shape(mean)[0]] + [1] * logstd.shape.ndims)
    with tf.variable_scope('value'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[:, 0]
    return (mean, logstd, value), state


class RecurrentGaussianPolicy(tf.contrib.rnn.RNNCell):
  """Independent recurrent policy and feed forward value networks.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector. The last policy layer is recurrent
  and uses a GRU cell.
  """

  def __init__(
      self, policy_layers, value_layers, action_size,
      mean_weights_initializer=_MEAN_WEIGHTS_INITIALIZER,
      logstd_initializer=_LOGSTD_INITIALIZER):
    self._policy_layers = policy_layers
    self._value_layers = value_layers
    self._action_size = action_size
    self._mean_weights_initializer = mean_weights_initializer
    self._logstd_initializer = logstd_initializer
    self._cell = tf.contrib.rnn.GRUBlockCell(100)

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return (self._action_size, self._action_size, tf.TensorShape([]))

  def __call__(self, observation, state):
    with tf.variable_scope('policy'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._policy_layers[:-1]:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      x, state = self._cell(x, state)
      mean = tf.contrib.layers.fully_connected(
          x, self._action_size, tf.tanh,
          weights_initializer=self._mean_weights_initializer)
      logstd = tf.get_variable(
          'logstd', mean.shape[1:], tf.float32, self._logstd_initializer)
      logstd = tf.tile(
          logstd[None, ...], [tf.shape(mean)[0]] + [1] * logstd.shape.ndims)
    with tf.variable_scope('value'):
      x = tf.contrib.layers.flatten(observation)
      for size in self._value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[:, 0]
    return (mean, logstd, value), state

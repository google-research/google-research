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

"""Implementation of noisy networks DQN https://arxiv.org/abs/1706.10295.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from dopamine.agents.dqn import dqn_agent as base_dqn_agent
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim
linearly_decaying_epsilon = base_dqn_agent.linearly_decaying_epsilon


@gin.constants_from_enum
class NoiseDistribution(enum.Enum):
  INDEPENDENT = 0
  FACTORISED = 1


def signed_sqrt(tensor):
  return tf.sign(tensor) * tf.sqrt(tf.abs(tensor))


def fully_connected(inputs, num_outputs,
                    activation_fn=tf.nn.relu,
                    scope=None,
                    collection=None,
                    distribution=NoiseDistribution.INDEPENDENT,
                    summary_writer=None):
  """Creates a fully connected layer with noise."""
  num_inputs = int(inputs.get_shape()[-1])
  weight_shape = (num_inputs, num_outputs)
  biases_shape = [num_outputs]

  # Parameters for each noise distribution, see Section 3.2 in original paper.
  if distribution == NoiseDistribution.INDEPENDENT:
    stddev = np.sqrt(3./num_inputs)
    constant = 0.017
    epsilon_w = tf.truncated_normal(weight_shape)
    epsilon_b = tf.truncated_normal(biases_shape)
  elif distribution == NoiseDistribution.FACTORISED:
    stddev = np.sqrt(1./num_inputs)
    constant = 0.5*np.sqrt(1/num_inputs)
    noise_input = tf.truncated_normal(weight_shape)
    noise_output = tf.truncated_normal(biases_shape)
    epsilon_w = tf.matmul(
        signed_sqrt(noise_output)[:, None], signed_sqrt(noise_input)[None, :])
    epsilon_b = signed_sqrt(noise_output)
  else:
    raise ValueError('Unknown noise distribution')

  mu_initializer = tf.initializers.random_uniform(
      minval=-stddev,
      maxval=stddev)
  sigma_initializer = tf.constant_initializer(value=constant)

  with tf.variable_scope(scope):
    mu_w = tf.get_variable('mu_w', weight_shape, trainable=True,
                           initializer=mu_initializer)
    sigma_w = tf.get_variable('sigma_w', weight_shape, trainable=True,
                              initializer=sigma_initializer)
    mu_b = tf.get_variable('mu_b', biases_shape, trainable=True,
                           initializer=mu_initializer)
    sigma_b = tf.get_variable('sigma_b', biases_shape, trainable=True,
                              initializer=sigma_initializer)
    if collection is not None:
      tf.add_to_collection(collection, mu_w)
      tf.add_to_collection(collection, mu_b)
      tf.add_to_collection(collection, sigma_w)
      tf.add_to_collection(collection, sigma_b)

    w = mu_w + sigma_w * epsilon_w
    b = mu_b + sigma_b * epsilon_b
    layer = tf.matmul(inputs, w)
    layer_bias = tf.nn.bias_add(layer, b)

    if summary_writer is not None:
      with tf.variable_scope('Noisy'):
        tf.summary.scalar('Sigma', tf.reduce_mean(sigma_w))

    if activation_fn is not None:
      layer_bias = activation_fn(layer_bias)
  return layer_bias


@gin.configurable
class NoisyDQNAgent(base_dqn_agent.DQNAgent):
  """Base class for a DQN agent with noisy layers."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=lambda w, x, y, z: 0,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=3,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500,
               noise_distribution=NoiseDistribution.INDEPENDENT):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      noise_distribution: string, distribution used to sample noise, must be
        `factorised` or `independent`.
    """
    self.noise_distribution = noise_distribution
    super(NoisyDQNAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _network_template(self, state):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(net, 32, [8, 8], stride=4)
    net = slim.conv2d(net, 64, [4, 4], stride=2)
    net = slim.conv2d(net, 64, [3, 3], stride=1)
    net = slim.flatten(net)
    net = fully_connected(net, 512, distribution=self.noise_distribution,
                          scope='fully_connected')
    q_values = fully_connected(net, self.num_actions,
                               activation_fn=None,
                               distribution=self.noise_distribution,
                               scope='fully_connected_1')
    return self._get_network_type()(q_values)

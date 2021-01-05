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

"""Implementation of a Rainbow agent with bootstrap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bonus_based_exploration.noisy_networks import noisy_dqn_agent

from dopamine.agents.rainbow import rainbow_agent as base_rainbow_agent
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


@gin.configurable
class NoisyRainbowAgent(base_rainbow_agent.RainbowAgent):
  """A Rainbow agent with noisy networks."""

  def __init__(self,
               sess,
               num_actions,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=lambda w, x, y, z: 0,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500,
               noise_distribution='independent'):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
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
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      noise_distribution: string, distribution used to sample noise, must be
        `factorised` or `independent`.
    """
    self.noise_distribution = noise_distribution
    super(NoisyRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
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
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(
        net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    net = slim.flatten(net)
    net = noisy_dqn_agent.fully_connected(
        net, 512, scope='fully_connected')
    net = noisy_dqn_agent.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        scope='fully_connected_1')

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = contrib_layers.softmax(logits)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    return self._get_network_type()(q_values, logits, probabilities)

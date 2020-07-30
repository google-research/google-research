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
"""An implementation of Advantage Learning in Dopamine.

Paper: "Increasing the Action Gap: New Operators for Reinforcement Learning",
Bellmare et al., AAAI 2016.
https://arxiv.org/abs/1512.04860.

The class ALDQNAgent inherits from Dopamine's DQNAgent.
"""

from dopamine.agents.dqn import dqn_agent
import gin.tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


@gin.configurable
class ALDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the AL-DQN agent."""

  def __init__(self, sess, num_actions, alpha=0.9,
               **kwargs):
    """Creates AL-DQN agent.

    Args:
     sess: tf.Session
     num_actions: int, number of actions in the environment.
     alpha: float in [0, 1]. Adcatage scaling factor.
     **kwargs: see dqn_agent.DQNAgent dcoumentation.
    """
    self.alpha = alpha
    super(ALDQNAgent, self).__init__(sess, num_actions, **kwargs)

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_target_net_outputs: The replayed states' target Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_net_outputs = self.online_convnet(
        self._replay.next_states)

    self._replay_target_net_outputs = self.target_convnet(
        self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

  def _build_target_op(self):
    raise NotImplementedError()

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_next_target_value = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)
    replay_target_value = tf.reduce_max(
        self._replay_target_net_outputs.q_values, 1)

    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')
    replay_target_chosen_q = tf.reduce_sum(
        self._replay_target_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    augmented_rewards = self._replay.rewards -  self.alpha * (
        replay_target_value - replay_target_chosen_q)

    target = (
        augmented_rewards + self.cumulative_gamma * replay_next_target_value *
        (1. - tf.cast(self._replay.terminals, tf.float32)))
    target = tf.stop_gradient(target)

    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))

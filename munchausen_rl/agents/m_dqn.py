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
"""An implementation of Munchausen DQN in Dopamine style.

The class MunchausenDQNAgent inherits from Dopamine's DQNAgent.
"""

import random
from dopamine.agents.dqn import dqn_agent
import gin
import tensorflow.compat.v1 as tf

from munchausen_rl.common import utils

tf.disable_v2_behavior()


@gin.configurable
class MunchausenDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the Munchausen-DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               tau,
               alpha=1,
               clip_value_min=-10,
               interact='greedy',
               optimizer_type='adam',
               optimizer_lr=0.00005,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    About tau and alpha coefficients:
    tau and alpha balance the entropy and KL regularizations. tau is used as the
    'explicit' entropy temperature, and alpha as a scaling of the log-policy.
    Implicitly, it defines an entropy regularization of coefficient
    (1-alpha) * tau and a KL one of coeff alpha * tau.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      tau: float (>0.), tau regularization factor in M-DQN.
      alpha: float in [0, 1], entropy scaling factor.
      clip_value_min: float (<0), minimum value to clip the log-policy.
      interact: string, 'stochastic' or 'greedy'. Which policy to use.
      optimizer_type: string, 'adam' or 'rms'.
      optimizer_lr: float, optimizer learning rate.
      **kwargs: see dqn_agent.DQNAgent doc.
    """
    self.tau = tau
    self.alpha = alpha
    self.clip_value_min = clip_value_min
    self._interact = interact
    self.optimizer_type = optimizer_type
    self.optimizer_lr = optimizer_lr
    self.optimizer = self._build_optimizer()

    super(MunchausenDQNAgent, self).__init__(sess, num_actions, **kwargs)

  def _build_optimizer(self):
    """Creates the optimizer for the Q-networks."""
    if self.optimizer_type == 'adam':
      return tf.train.AdamOptimizer(
          learning_rate=self.optimizer_lr, epsilon=0.0003125)
    if self.optimizer_type == 'rms':
      return tf.train.RMSPropOptimizer(
          learning_rate=self.optimizer_lr,
          decay=0.95,
          momentum=0.0,
          epsilon=0.00001,
          centered=True)
    raise ValueError('Undefined optimizer')

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_net_outputs: The replayed next states' Q-values.
      self._replay_target_net_outputs: The replayed states' target
        Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values.
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')

    self._net_outputs = self.online_convnet(self.state_ph)

    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(
        self._replay.states)
    self._replay_next_net_outputs = self.online_convnet(
        self._replay.next_states)

    self._replay_target_net_outputs = self.target_convnet(
        self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    self._policy_logits = utils.stable_scaled_log_softmax(
        self._net_outputs.q_values, self.tau, axis=1) / self.tau

    self._stochastic_action = tf.random.categorical(
        self._policy_logits,
        num_samples=1,
        dtype=tf.int32)[0][0]

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    # tau * ln pi_k+1 (s')
    replay_next_log_policy = utils.stable_scaled_log_softmax(
        self._replay_next_target_net_outputs.q_values, self.tau, axis=1)
    # tau * ln pi_k+1(s)
    replay_log_policy = utils.stable_scaled_log_softmax(
        self._replay_target_net_outputs.q_values, self.tau, axis=1)
    replay_next_policy = utils.stable_softmax(  # pi_k+1(s')
        self._replay_next_target_net_outputs.q_values, self.tau, axis=1)

    replay_next_qt_softmax = tf.reduce_sum(
        (self._replay_next_target_net_outputs.q_values -
         replay_next_log_policy) * replay_next_policy, 1)

    tau_log_pi_a = tf.reduce_sum(  # tau * ln pi_k+1(a|s)
        replay_log_policy * replay_action_one_hot, axis=1)

    tau_log_pi_a = tf.clip_by_value(
        tau_log_pi_a,
        clip_value_min=self.clip_value_min,
        clip_value_max=1)

    munchausen_term = self.alpha * tau_log_pi_a

    modified_bellman = (
        self._replay.rewards + munchausen_term +
        self.cumulative_gamma * replay_next_qt_softmax *
        (1. - tf.cast(self._replay.terminals, tf.float32)))

    if self.summary_writer is not None:
      with tf.variable_scope('policy'):
        entropy = -tf.reduce_sum(
            replay_next_policy * replay_next_log_policy / self.tau, axis=1)
        tf.summary.scalar('entropy', tf.reduce_mean(entropy))

    return modified_bellman

  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # Choose the action with highest Q-value at the current state.
      if self._interact == 'stochastic':
        selected_action = self._stochastic_action
      elif self._interact == 'greedy':
        selected_action = self._q_argmax
      else:
        raise ValueError('Undefined interaction')
      return self._sess.run(selected_action, {self.state_ph: self.state})

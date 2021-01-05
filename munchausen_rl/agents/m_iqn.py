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

"""An implementation of Munchausen IQN in Dopamine style.

The class MunchausenIQNAgent inherits from Dopamine's RainbowAgent.
"""

import math
import random
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
import gin
import numpy as np
import tensorflow.compat.v1 as tf

from munchausen_rl.common import utils

tf.disable_v2_behavior()


@gin.configurable
class MunchausenIQNAgent(rainbow_agent.RainbowAgent):
  """An implementation of the Munchausen-IQN agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=atari_lib.ImplicitQuantileNetwork,
               kappa=1.0,
               alpha=0.9,
               tau=0.03,
               clip_value_min=-1,
               interact='stochastic',
               replay_scheme='uniform',
               num_tau_samples=32,
               num_tau_prime_samples=32,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               eval_mode=False,
               use_staging=True,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.train.AdamOptimizer(),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes the agent and constructs the Graph.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      sess: `tf.Session` object for running associated ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expects three parameters:
        (num_actions, quantile_embedding_dim, network_type). This class is used
        to generate network instances that are used by the agent. Each
        instantiation would have different set of variables. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      kappa: float, Huber loss cutoff.
      alpha: float in [0, 1], entropy scaling factor.
      tau: float (>0.), tau regularization factor in M-DQN.
      clip_value_min: float (<0), minimum value to clip the log-policy.
      interact: string, 'stochastic' or 'greedy'. Which policy to use.
      replay_scheme: string, 'uniform' or 'prioritized'.
      num_tau_samples: int, number of online quantile samples for loss
        estimation.
      num_tau_prime_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
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
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      eval_mode: bool, True for evaluation and False for training.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: tf.train.Optimizer, the optimizer to use for training.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    self.kappa = kappa
    # num_tau_samples = N below equation (3) in the paper.
    self.num_tau_samples = num_tau_samples
    # num_tau_prime_samples = N' below equation (3) in the paper.
    self.num_tau_prime_samples = num_tau_prime_samples
    # num_quantile_samples = k below equation (3) in the paper.
    self.num_quantile_samples = num_quantile_samples
    # quantile_embedding_dim = n above equation (4) in the paper.
    self.quantile_embedding_dim = quantile_embedding_dim
    # option to perform double dqn.
    self.alpha = alpha
    self.tau = tau
    self.clip_value_min = clip_value_min
    self._interact = interact

    self._replay_scheme = replay_scheme
    self.num_actions = num_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    with tf.device(tf_device):
      # Create a placeholder for the state input to the DQN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.placeholder(
          self.observation_dtype, state_shape, name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)

      self._build_networks()

      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess

    var_map = atari_lib.maybe_transform_variable_names(
        tf.global_variables())
    self._saver = tf.train.Saver(
        var_list=var_map, max_to_keep=max_tf_checkpoints_to_keep)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

  def _create_network(self, name):
    r"""Builds an Implicit Quantile ConvNet.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_actions, self.quantile_embedding_dim,
                           name=name)
    return network

  def _build_networks(self):
    """Builds the IQN computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's quantile values.
      self.target_convnet: For computing the next state's target quantile
        values.
      self._net_outputs: The actual quantile values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' quantile values.
      self._replay_next_target_net_outputs: The replayed next states' target
        quantile values.
    """
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')

    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(self.state_ph,
                                            self.num_quantile_samples)
    # Shape of self._net_outputs.quantile_values:
    # num_quantile_samples x num_actions.
    # e.g. if num_actions is 2, it might look something like this:
    # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
    #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
    # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
    self._q_values = tf.reduce_mean(self._net_outputs.quantile_values, axis=0)
    self._q_argmax = tf.argmax(self._q_values, axis=0)
    self._policy_logits = tf.nn.softmax(self._q_values / self.tau, axis=0)
    self._stochastic_action = tf.random.categorical(
        self._policy_logits[None, Ellipsis],
        num_samples=1,
        dtype=tf.int32)[0][0]

    self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                   self.num_tau_samples)
    # Shape: (num_tau_samples x batch_size) x num_actions.
    self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
    self._replay_net_quantiles = self._replay_net_outputs.quantiles

    # Do the same for next states in the replay buffer.
    self._replay_net_target_outputs = self.target_convnet(
        self._replay.next_states, self.num_tau_prime_samples)
    # Shape: (num_tau_prime_samples x batch_size) x num_actions.
    vals = self._replay_net_target_outputs.quantile_values
    self._replay_net_target_quantile_values = vals

    # Compute Q-values which are used for action selection for the states and
    # next states in the replay buffer.
    target_next_action = self.target_convnet(self._replay.next_states,
                                             self.num_quantile_samples)
    target_action = self.target_convnet(self._replay.states,
                                        self.num_quantile_samples)

    # Shape: (num_quantile_samples x batch_size) x num_actions.
    target_next_quantile_values_action = target_next_action.quantile_values
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_next_quantile_values_action = tf.reshape(
        target_next_quantile_values_action,
        [self.num_quantile_samples, self._replay.batch_size, self.num_actions])

    # Shape: (num_quantile_samples x batch_size) x num_actions.
    target_quantile_values_action = target_action.quantile_values
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                               [self.num_quantile_samples,
                                                self._replay.batch_size,
                                                self.num_actions])
    # Shape: batch_size x num_actions.
    self._replay_next_target_q_values = tf.squeeze(tf.reduce_mean(
        target_next_quantile_values_action, axis=0))
    self._replay_target_q_values = tf.squeeze(tf.reduce_mean(
        target_quantile_values_action, axis=0))

    self._replay_next_qt_argmax = tf.argmax(
        self._replay_next_target_q_values, axis=1)

  def _build_target_quantile_values_op(self):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    ###### Munchausen-specific
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    # tau * ln pi_k+1 (s')
    replay_next_log_policy = utils.stable_scaled_log_softmax(
        self._replay_next_target_q_values, self.tau, axis=1)
    # tau * ln pi_k+1(s)
    replay_log_policy = utils.stable_scaled_log_softmax(
        self._replay_target_q_values, self.tau, axis=1)
    replay_next_policy = utils.stable_softmax(  # pi_k+1(s')
        self._replay_next_target_q_values, self.tau, axis=1)

    tau_log_pi_a = tf.reduce_sum(  # ln pi_k+1(a|s)
        replay_log_policy * replay_action_one_hot, axis=1)

    tau_log_pi_a = tf.clip_by_value(
        tau_log_pi_a, clip_value_min=self.clip_value_min, clip_value_max=0)

    munchuasen_term = self.alpha * tau_log_pi_a
    #########

    # Shape of rewards: (num_tau_prime_samples x batch_size) x 1.
    rewards = self._replay.rewards[:, None] + munchuasen_term[Ellipsis, None]
    rewards = tf.tile(rewards, [self.num_tau_prime_samples, 1])

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                  [self.num_tau_prime_samples, 1])

    # shape: (batch_size * num_tau_prime_samples) x num_actions
    replay_next_policy_ = tf.tile(replay_next_policy,
                                  [self.num_tau_prime_samples, 1])
    replay_next_log_policy_ = tf.tile(replay_next_log_policy,
                                      [self.num_tau_prime_samples, 1])

    # shape: (batch_size * num_tau_prime_samples) x 1
    replay_quantile_values = tf.reshape(
        self._replay_net_target_quantile_values,
        [batch_size * self.num_tau_prime_samples, self.num_actions])

    # shape: (batch_size * num_tau_prime_samples) x num_actions
    weighted_logits = (
        replay_next_policy_ * (replay_quantile_values
                               - replay_next_log_policy_))

    # shape: (batch_size * num_tau_prime_samples) x 1
    target_quantile_values = tf.reduce_sum(weighted_logits, axis=1,
                                           keepdims=True)

    return rewards + gamma_with_terminal * target_quantile_values

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    batch_size = tf.shape(self._replay.rewards)[0]

    target_quantile_values = tf.stop_gradient(
        self._build_target_quantile_values_op())
    # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    target_quantile_values = tf.reshape(target_quantile_values,
                                        [self.num_tau_prime_samples,
                                         batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_tau_prime_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_tau_prime_samples x 1.
    target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])

    # Shape of indices: (num_tau_samples x batch_size) x 1.
    # Expand dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    indices = tf.range(self.num_tau_samples * batch_size)[:, None]

    # Expand the dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    reshaped_actions = self._replay.actions[:, None]
    reshaped_actions = tf.tile(reshaped_actions, [self.num_tau_samples, 1])
    # Shape of reshaped_actions: (num_tau_samples x batch_size) x 2.
    reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

    chosen_action_quantile_values = tf.gather_nd(
        self._replay_net_quantile_values, reshaped_actions)
    # Reshape to self.num_tau_samples x batch_size x 1 since this is the manner
    # in which the quantile values are tiled.
    chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                               [self.num_tau_samples,
                                                batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_tau_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of chosen_action_quantile_values:
    # batch_size x num_tau_samples x 1.
    chosen_action_quantile_values = tf.transpose(
        chosen_action_quantile_values, [1, 0, 2])

    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = target_quantile_values[
        :, :, None, :] - chosen_action_quantile_values[:, None, :, :]
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = (
        tf.cast(tf.abs(bellman_errors) <= self.kappa, tf.float32) *
        0.5 * bellman_errors ** 2)
    huber_loss_case_two = (
        tf.cast(tf.abs(bellman_errors) > self.kappa, tf.float32) *
        self.kappa * (tf.abs(bellman_errors) - 0.5 * self.kappa))
    huber_loss = huber_loss_case_one + huber_loss_case_two

    # Reshape replay_quantiles to batch_size x num_tau_samples x 1
    replay_quantiles = tf.reshape(
        self._replay_net_quantiles, [self.num_tau_samples, batch_size, 1])
    replay_quantiles = tf.transpose(replay_quantiles, [1, 0, 2])

    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    replay_quantiles = tf.cast(
        tf.tile(replay_quantiles[:, None, :, :],
                [1, self.num_tau_prime_samples, 1, 1]), tf.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (tf.abs(replay_quantiles - tf.stop_gradient(
        tf.cast(bellman_errors < 0, tf.float32))) * huber_loss) / self.kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = tf.reduce_sum(quantile_huber_loss, axis=2)
    # Shape: batch_size x 1.
    loss = tf.reduce_mean(loss, axis=1)

    update_priorities_op = tf.no_op()
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('QuantileLoss', tf.reduce_mean(loss))
      return self.optimizer.minimize(tf.reduce_mean(loss)), tf.reduce_mean(loss)

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
      return self._sess.run(selected_action,
                            {self.state_ph: self.state})

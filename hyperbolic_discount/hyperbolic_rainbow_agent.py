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

"""Compact implementation of a Hyperbolic Rainbow agent.

Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.

Plus auxiliary tasks for hyperbolic action.

These three components were found to significantly impact the performance of
the Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

  * keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
  * remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""
import collections
import math

from dopamine.agents.dqn import dqn_agent
import gin
import numpy as np
import tensorflow.compat.v1 as tf

from hyperbolic_discount import agent_utils
from hyperbolic_discount.replay_memory import prioritized_replay_buffer
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim


@gin.configurable
class HyperRainbowAgent(dqn_agent.DQNAgent):
  """A compact implementation of a Hyperbolic Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               number_of_gammas=8,
               gamma_max=0.99,
               acting_policy='hyperbolic',
               hyp_exponent=1.0,
               integral_estimate='lower',
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               gradient_clipping_norm=None,
               network_size_expansion=1.0,
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=50000):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      number_of_gammas: int, the number of gammas to estimate in parallel.
      gamma_max: int, the maximum gammas we will learn via Bellman updates.
      acting_policy: str, the policy with which the agent will act.  One of
        ['hyperbolic', 'largest_gamma']
      hyp_exponent:  float, the parameter k in the equation 1. / (1. + k * t)
        for hyperbolic discounting.  Smaller parameter will lead to a longer
        horizon.
      integral_estimate:  str, how to estimate the integral of the hyperbolic
        discount.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon). This function should return the epsilon value
        used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      gradient_clipping_norm: str, if not None, this will set the gradient
        clipping value.
      network_size_expansion: float, the multiplier on the default layer size.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = tf.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    self.optimizer = optimizer
    self.number_of_gammas = number_of_gammas
    self.gamma_max = gamma_max
    self.acting_policy = acting_policy
    self.hyp_exponent = hyp_exponent
    self.integral_estimate = integral_estimate
    self.gradient_clipping_norm = gradient_clipping_norm
    self.network_size_expansion = network_size_expansion

    # These are the discount factors (gammas) used to estimate the integral.
    self.eval_gammas = agent_utils.compute_eval_gamma_interval(
        self.gamma_max, self.hyp_exponent, self.number_of_gammas)
    # However, if we wish to estimate hyperbolic discounting with the form,
    #
    #      \Gamma_t =  1. / (1. + k * t)
    #
    # where we now have a coefficient k <= 1.0
    # we need consider the value functions for \gamma ^ k.  We refer to
    # these below as self.gammas, since these are the gammas actually being
    # learned via Bellman updates.
    self.gammas = [
        math.pow(gamma, self.hyp_exponent) for gamma in self.eval_gammas
    ]

    assert max(self.gammas) <= self.gamma_max

    super(HyperRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        gamma=0,  # TODO(liamfedus): better way to deal with self.gamma
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _get_network_type(self):
    """Returns the type of the outputs of a value distribution network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple(
        'hyper_rainbow_network',
        ['hyp_q_values', 'q_values', 'logits', 'probabilities'])

  def _network_template(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    with tf.variable_scope('body'):
      net = tf.cast(state, tf.float32)
      net = tf.div(net, 255.)
      net = slim.conv2d(
          net,
          int(32 * self.network_size_expansion), [8, 8],
          stride=4,
          weights_initializer=weights_initializer)
      net = slim.conv2d(
          net,
          int(64 * self.network_size_expansion), [4, 4],
          stride=2,
          weights_initializer=weights_initializer)
      net = slim.conv2d(
          net,
          int(64 * self.network_size_expansion), [3, 3],
          stride=1,
          weights_initializer=weights_initializer)
      net = slim.flatten(net)
      body_net = slim.fully_connected(
          net,
          int(512 * self.network_size_expansion),
          weights_initializer=weights_initializer)

    logits = []
    probabilities = []
    q_values = []
    with tf.variable_scope('head'):
      for _ in range(self.number_of_gammas):
        net = slim.fully_connected(
            body_net,
            self.num_actions * self._num_atoms,
            activation_fn=None,
            weights_initializer=weights_initializer)

        gamma_logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
        gamma_probabilities = contrib_layers.softmax(gamma_logits)
        gamma_q_values = tf.reduce_sum(
            self._support * gamma_probabilities, axis=2)

        # Add one for each gamma being used.
        logits.append(gamma_logits)
        probabilities.append(gamma_probabilities)
        q_values.append(gamma_q_values)

    # Estimate the hyperbolic discounted q-values.
    hyp_q_values = agent_utils.integrate_q_values(q_values,
                                                  self.integral_estimate,
                                                  self.eval_gammas,
                                                  self.number_of_gammas,
                                                  self.gammas)

    return self._get_network_type()(hyp_q_values, q_values, logits,
                                    probabilities)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedPrioritizedReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon)

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs = self.online_convnet(self.state_ph)

    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    if self.acting_policy == 'hyperbolic':
      self._q_argmax = tf.argmax(self._net_outputs.hyp_q_values, axis=1)[0]
    elif self.acting_policy == 'largest_gamma':
      self._q_argmax = tf.argmax(self._net_outputs.q_values[-1], axis=1)[0]
    else:
      raise NotImplementedError

    vars_to_sum = [
        v for v in tf.trainable_variables() if v.name.startswith('Online')
    ]
    for var in vars_to_sum:
      agent_utils.variable_summaries(var, var.name)

  def _build_discounted_n_step_rewards(self, gamma):
    """Compute the discounted n-step return."""
    discounts = [gamma**i for i in range(self.update_horizon)]
    discount_tensor = tf.constant(discounts)
    return tf.reduce_sum(self._replay.rewards * discount_tensor, axis=1)

  def _build_target_distribution(self):
    """Builds the C51 target distribution as per Bellemare et al. (2017).

    First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
    is the support of the next state distribution:

      * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
      * 0 otherwise (duplicated num_atoms times).

    Second, we compute the next-state probabilities, corresponding to the action
    with highest expected value.

    Finally we project the Bellman target (support + probabilities) onto the
    original support.

    Returns:
      target_distributions: A list of length self.number_of_gammas of tf.tensor,
      the target distribution from the replay.
    """
    target_distributions = []
    for gamma, target_q_values, target_probabilities in zip(
        self.gammas, self._replay_next_target_net_outputs.q_values,
        self._replay_next_target_net_outputs.probabilities):
      batch_size = self._replay.batch_size

      # size of rewards: batch_size x 1
      rewards = self._build_discounted_n_step_rewards(gamma)[:, None]

      # size of tiled_support: batch_size x num_atoms
      tiled_support = tf.tile(self._support, [batch_size])
      tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

      # size of target_support: batch_size x num_atoms

      is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
      # Incorporate terminal state to discount factor.
      # size of gamma_with_terminal: batch_size x 1
      cumulative_gamma = math.pow(gamma, self.update_horizon)
      gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
      gamma_with_terminal = gamma_with_terminal[:, None]

      target_support = rewards + gamma_with_terminal * tiled_support

      # size of next_qt_argmax: 1 x batch_size
      next_qt_argmax = tf.argmax(target_q_values, axis=1)[:, None]
      batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
      # size of next_qt_argmax: batch_size x 2
      batch_indexed_next_qt_argmax = tf.concat([batch_indices, next_qt_argmax],
                                               axis=1)

      # size of next_probabilities: batch_size x num_atoms
      next_probabilities = tf.gather_nd(target_probabilities,
                                        batch_indexed_next_qt_argmax)

      distribution = project_distribution(target_support, next_probabilities,
                                          self._support)
      target_distributions.append(distribution)
    return target_distributions

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    target_distributions = self._build_target_distribution()
    target_distributions = [tf.stop_gradient(t) for t in target_distributions]
    loss = 0.
    gamma_loss_summaries = []
    # Maintain priorities across all the gammas.
    priorities = tf.zeros([self._replay.batch_size], dtype=tf.float32)

    for i, target_distribution in enumerate(target_distributions):
      # size of indices: batch_size x 1.
      indices = tf.range(tf.shape(
          self._replay_net_outputs.logits[i])[0])[:, None]
      # size of reshaped_actions: batch_size x 2.
      reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
      # For each element of the batch, fetch the logits for its selected action.
      chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits[i],
                                          reshaped_actions)

      gamma_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=target_distribution, logits=chosen_action_logits)

      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      # Weight the gamma_loss by the inverse priorities.
      weighted_gamma_loss = loss_weights * gamma_loss

      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      loss += weighted_gamma_loss

      if self.summary_writer is not None:
        loss_summary = tf.summary.scalar('Losses/CrossEntropyLoss_{}'.format(i),
                                         tf.reduce_mean(weighted_gamma_loss))
        gamma_loss_summaries.append(loss_summary)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.

      # Accrue the priorities for each gamma.
      # This was the approach used for the paper Fedus et al. (2019), but other
      # prioritization schemes exist.
      # Alternatively, we could have prioritized by only the TD-errors from
      # the largest gamma.
      priorities += tf.sqrt(gamma_loss + 1e-10) / self.number_of_gammas

    # Commented version for illustration on how to use the priority only
    # for the largest gamma.
    # priorities += tf.sqrt(gamma_loss + 1e-10)

    # We have an update_priorities_op for all the gammas.
    if self._replay_scheme == 'prioritized':
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, priorities)
    else:
      update_priorities_op = tf.no_op()

    # Divide by the number of gammas to preserve scale.
    loss = loss / self.number_of_gammas

    with tf.control_dependencies([update_priorities_op] + gamma_loss_summaries):

      def clip_if_not_none(grad, clip_norm=5.):
        """Clip the gradient only if not None."""
        if grad is None:
          return grad
        return tf.clip_by_norm(grad, clip_norm)

      if self.gradient_clipping_norm is not None:
        # Clip gradients to test stability.
        grads_and_vars = self.optimizer.compute_gradients(tf.reduce_mean(loss))
        clipped_gradients = [
            (clip_if_not_none(grad, clip_norm=self.gradient_clipping_norm), var)
            for grad, var in grads_and_vars
        ]

        return self.optimizer.apply_gradients(clipped_gradients), loss
      else:
        return self.optimizer.minimize(tf.reduce_mean(loss)), loss

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer (last_observation, action, reward,
    is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      priority = (1. if self._replay_scheme == 'uniform' else
                  self._replay.memory.sum_tree.max_recorded_priority)

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)


def project_distribution(supports, weights, target_support,
                         validate_args=False):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:

    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]

  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.
    validate_args: Whether we will verify the contents of the target_support
      parameter.

  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas = target_support[1:] - target_support[:-1]
  # delta_z = `\Delta z` in Eq7.
  delta_z = target_support_deltas[0]
  validate_deps = []
  supports.shape.assert_is_compatible_with(weights.shape)
  supports[0].shape.assert_is_compatible_with(target_support.shape)
  target_support.shape.assert_has_rank(1)
  if validate_args:
    # Assert that supports and weights have the same shapes.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
            [supports, weights]))
    # Assert that elements of supports and target_support have the same shape.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(
                tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
            [supports, target_support]))
    # Assert that target_support has a single dimension.
    validate_deps.append(
        tf.Assert(
            tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
    # Assert that the target_support is monotonically increasing.
    validate_deps.append(
        tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
    # Assert that the values in target_support are equally spaced.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
            [target_support]))

  with tf.control_dependencies(validate_deps):
    # Ex: `v_min, v_max = 4, 8`.
    v_min, v_max = target_support[0], target_support[-1]
    # Ex: `batch_size = 2`.
    batch_size = tf.shape(supports)[0]
    # `N` in Eq7.
    # Ex: `num_dims = 5`.
    num_dims = tf.shape(target_support)[0]
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
    #                         [[ 4.  4.  4.  5.  6.]]]`.
    clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
    # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]]
    #                        [[ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]]]]`.
    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
    # Ex: `reshaped_target_support = [[[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]
    #                                 [[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]]`.
    reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])
    # numerator = `|clipped_support - z_i|` in Eq7.
    # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
    #                     [ 1.  1.  1.  1.  3.]
    #                     [ 2.  2.  2.  0.  2.]
    #                     [ 3.  3.  3.  1.  1.]
    #                     [ 4.  4.  4.  2.  0.]]
    #                    [[ 0.  0.  0.  1.  2.]
    #                     [ 1.  1.  1.  0.  1.]
    #                     [ 2.  2.  2.  1.  0.]
    #                     [ 3.  3.  3.  2.  1.]
    #                     [ 4.  4.  4.  3.  2.]]]]`.
    numerator = tf.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  1.]]
    #                           [[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  1.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]]]]`.
    clipped_quotient = tf.clip_by_value(quotient, 0, 1)
    # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
    #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
    weights = weights[:, None, :]
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
    # in Eq7.
    # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0.1]]
    #                     [[ 0.1  0.2  0.5  0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0.1]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]]]]`.
    inner_prod = clipped_quotient * weights
    # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
    #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection

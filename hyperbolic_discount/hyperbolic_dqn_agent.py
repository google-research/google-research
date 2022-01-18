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

"""Hyperbolic DQN agent."""
import collections
import math


from dopamine.agents.dqn import dqn_agent
import gin
import tensorflow.compat.v1 as tf

from hyperbolic_discount import agent_utils
from hyperbolic_discount.replay_memory import circular_replay_buffer
from tensorflow.contrib import slim


@gin.configurable
class HyperDQNAgent(dqn_agent.DQNAgent):
  """A compact implementation of a Hyperbolic DQN agent."""

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
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='uniform',
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

    super(HyperDQNAgent, self).__init__(
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

    The hyperbolic Q-values are estimated via a Riemann sum to approximate the
    integral.  This builds a lower or an upper estimate of the integral via a
    set of rectangles.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('hyper_dqn_network',
                                  ['hyp_q_value', 'q_values'])

  def _network_template(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(
        net, int(32 * self.network_size_expansion), [8, 8], stride=4)
    net = slim.conv2d(
        net, int(64 * self.network_size_expansion), [4, 4], stride=2)
    net = slim.conv2d(
        net, int(64 * self.network_size_expansion), [3, 3], stride=1)
    net = slim.flatten(net)
    net = slim.fully_connected(net, int(512 * self.network_size_expansion))

    q_values = []
    for _ in range(self.number_of_gammas):
      gamma_q_value = slim.fully_connected(
          net, self.num_actions, activation_fn=None)
      q_values.append(gamma_q_value)

    # Estimate the hyperbolic discounted q-values
    hyp_q_value = agent_utils.integrate_q_values(q_values,
                                                 self.integral_estimate,
                                                 self.eval_gammas,
                                                 self.number_of_gammas,
                                                 self.gammas)

    return self._get_network_type()(hyp_q_value, q_values)

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
      self._q_argmax = tf.argmax(self._net_outputs.hyp_q_value, axis=1)[0]
    elif self.acting_policy == 'largest_gamma':
      self._q_argmax = tf.argmax(self._net_outputs.q_values[-1], axis=1)[0]
    else:
      raise NotImplementedError

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer object.
    """
    return circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_discounted_n_step_rewards(self, gamma):
    """Compute the discounted n-step return."""
    discounts = [gamma**i for i in range(self.update_horizon)]
    discount_tensor = tf.constant(discounts)
    return tf.reduce_sum(self._replay.rewards * discount_tensor, axis=1)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    targets = []
    for gamma, target_q in zip(self.gammas,
                               self._replay_next_target_net_outputs.q_values):
      # Get the maximum Q-value across the actions dimension.
      replay_next_qt_max = tf.reduce_max(target_q, 1)

      # Calculate the Bellman target value.
      #   Q_t = R_t + \gamma^N * Q'_t+1
      # where,
      #   Q'_t+1 = \argmax_a Q(S_t+1, a)
      #          (or) 0 if S_t is a terminal state,
      # and
      #   N is the update horizon (by default, N=1).
      cumulative_gamma = math.pow(gamma, self.update_horizon)
      n_step_reward = self._build_discounted_n_step_rewards(gamma)
      targets.append(n_step_reward + cumulative_gamma * replay_next_qt_max *
                     (1. - tf.cast(self._replay.terminals, tf.float32)))
    return targets

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')

    replay_chosen_qs = []
    for i in range(len(self.gammas)):
      replay_chosen_q = tf.reduce_sum(
          self._replay_net_outputs.q_values[i] * replay_action_one_hot,
          reduction_indices=1,
          name='replay_chosen_q_{}'.format(i))
      replay_chosen_qs.append(replay_chosen_q)

    targets = self._build_target_q_op()
    loss = 0.

    for i, (target,
            replay_chosen_q) in enumerate(zip(targets, replay_chosen_qs)):
      gamma_loss = tf.losses.huber_loss(
          tf.stop_gradient(target),
          replay_chosen_q,
          reduction=tf.losses.Reduction.NONE)

      loss += gamma_loss
      if self.summary_writer is not None:
        tf.summary.scalar('Losses/GammaLoss_{}'.format(i),
                          tf.reduce_mean(gamma_loss))

    # Divide by the number of gammas to preserve scale.
    loss = loss / self.number_of_gammas

    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))

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

      return self.optimizer.apply_gradients(clipped_gradients)
    else:
      return self.optimizer.minimize(tf.reduce_mean(loss))

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

"""A DQN Agent.

Implements the DQN algorithm from

"Human level control through deep reinforcement learning"
  Mnih et al., 2015
  https://deepmind.com/research/dqn/
"""

import collections
from abps import tf_agent
import gin
import tensorflow.compat.v2 as tf

from tf_agents.policies import boltzmann_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops


class DqnLossInfo(
    collections.namedtuple('DqnLossInfo', ('td_loss', 'td_error'))):
  """DqnLossInfo is stored in the `extras` field of the LossInfo instance.

  Both `td_loss` and `td_error` have a validity mask applied to ensure that
  no loss or error is calculated for episode boundaries.

  td_loss: The **weighted** TD loss (depends on choice of loss metric and
    any weights passed to the DQN loss function.
  td_error: The **unweighted** TD errors, which are just calculated as:

    ```
    td_error = td_targets - q_values
    ```

    These can be used to update Prioritized Replay Buffer priorities.

    Note that, unlike `td_loss`, `td_error` may contain a time dimension when
    training with RNN mode.  For `td_loss`, this axis is averaged out.
  """
  pass


# this file. Move them to utils/common or utils/losses.
def element_wise_squared_loss(x, y):
  return tf.compat.v1.losses.mean_squared_error(
      x, y, reduction=tf.compat.v1.losses.Reduction.NONE)


def element_wise_huber_loss(x, y):
  return tf.compat.v1.losses.huber_loss(
      x, y, reduction=tf.compat.v1.losses.Reduction.NONE)


def compute_td_targets(next_q_values, rewards, discounts):
  return tf.stop_gradient(rewards + discounts * next_q_values)


@gin.configurable
class DqnAgent(tf_agent.TFAgent):
  """A DQN Agent.

  Implements the DQN algorithm from

  "Human level control through deep reinforcement learning"
    Mnih et al., 2015
    https://deepmind.com/research/dqn/

  This agent also implements n-step updates. See "Rainbow: Combining
  Improvements in Deep Reinforcement Learning" by Hessel et al., 2017, for a
  discussion on its benefits: https://arxiv.org/abs/1710.02298
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      q_network,
      optimizer,
      epsilon_greedy=0.1,
      n_step_update=1,
      boltzmann_temperature=None,
      emit_log_probability=False,
      update_period=None,
      # Params for target network updates
      target_update_tau=1.0,
      target_update_period=1,
      # Params for training.
      td_errors_loss_fn=None,
      gamma=1.0,
      reward_scale_factor=1.0,
      gradient_clipping=None,
      # Params for debugging
      debug_summaries=False,
      enable_functions=True,
      summarize_grads_and_vars=False,
      train_step_counter=None,
      name=None):
    """Creates a DQN Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type).
      optimizer: The optimizer to use for training.
      epsilon_greedy: probability of choosing a random action in the default
        epsilon-greedy collect policy (used only if a wrapper is not provided to
        the collect_policy method).
      n_step_update: The number of steps to consider when computing TD error and
        TD loss. Defaults to single-step updates. Note that this requires the
        user to call train on Trajectory objects with a time dimension of
        `n_step_update + 1`. However, note that we do not yet support
        `n_step_update > 1` in the case of RNNs (i.e., non-empty
        `q_network.state_spec`).
      boltzmann_temperature: Temperature value to use for Boltzmann sampling of
        the actions during data collection. The closer to 0.0, the higher the
        probability of choosing the best action.
      emit_log_probability: Whether policies emit log probabilities or not.
      update_period: Update period.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      td_errors_loss_fn: A function for computing the TD errors loss. If None, a
        default value of element_wise_huber_loss is used. This function takes as
        input the target and the estimated Q values and returns the loss for
        each element of the batch.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      enable_functions: A bool to decide whether or not to enable tf function
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or action
        spec minimum is not equal to 0.
      NotImplementedError: If `q_network` has non-empty `state_spec` (i.e., an
        RNN is provided) and `n_step_update > 1`.
    """
    tf.Module.__init__(self, name=name)

    flat_action_spec = tf.nest.flatten(action_spec)
    self._num_actions = [
        spec.maximum - spec.minimum + 1 for spec in flat_action_spec
    ]

    if len(flat_action_spec) > 1 or flat_action_spec[0].shape.ndims > 1:
      raise ValueError('Only one dimensional actions are supported now.')

    if not all(spec.minimum == 0 for spec in flat_action_spec):
      raise ValueError(
          'Action specs should have minimum of 0, but saw: {0}'.format(
              [spec.minimum for spec in flat_action_spec]))

    if epsilon_greedy is not None and boltzmann_temperature is not None:
      raise ValueError(
          'Configured both epsilon_greedy value {} and temperature {}, '
          'however only one of them can be used for exploration.'.format(
              epsilon_greedy, boltzmann_temperature))

    self._q_network = q_network
    self._target_q_network = self._q_network.copy(name='TargetQNetwork')
    self._epsilon_greedy = epsilon_greedy
    self._n_step_update = n_step_update
    self._boltzmann_temperature = boltzmann_temperature
    self._optimizer = optimizer
    self._td_errors_loss_fn = td_errors_loss_fn or element_wise_huber_loss
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._gradient_clipping = gradient_clipping
    self._update_target = self._get_target_updater(target_update_tau,
                                                   target_update_period)

    policy = q_policy.QPolicy(
        time_step_spec,
        action_spec,
        q_network=self._q_network,
        emit_log_probability=emit_log_probability)

    if boltzmann_temperature is not None:
      collect_policy = boltzmann_policy.BoltzmannPolicy(
          policy, temperature=self._boltzmann_temperature)
    else:
      collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
          policy, epsilon=self._epsilon_greedy)
    policy = greedy_policy.GreedyPolicy(policy)

    if q_network.state_spec and n_step_update != 1:
      raise NotImplementedError(
          'DqnAgent does not currently support n-step updates with stateful '
          'networks (i.e., RNNs), but n_step_update = {}'.format(n_step_update))

    train_sequence_length = (
        n_step_update + 1 if not q_network.state_spec else None)

    super(DqnAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=train_sequence_length,
        update_period=update_period,
        debug_summaries=debug_summaries,
        enable_functions=enable_functions,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

    tf.compat.v1.summary.scalar(
        'epsilon/' + self.name,
        self._epsilon_greedy,
        collections=['train_' + self.name])

  def _initialize_v1(self):
    self._q_network.create_variables()
    if self._target_q_network:
      self._target_q_network.create_variables()
    return common.soft_variables_update(
        self._q_network.variables, self._target_q_network.variables, tau=1.0)

  def _initialize(self):
    common.soft_variables_update(
        self._q_network.variables, self._target_q_network.variables, tau=1.0)

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the q network, and its corresponding
    weight w_t in the target_q_network, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets'):

      def update():
        return common.soft_variables_update(self._q_network.variables,
                                            self._target_q_network.variables,
                                            tau)

      return common.Periodically(update, period, 'periodic_update_targets')

  def _experience_to_transitions(self, experience):
    transitions = trajectory.to_transition(experience)

    # Remove time dim if we are not using a recurrent network.
    if not self._q_network.state_spec:
      transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]),
                                          transitions)

    time_steps, policy_steps, next_time_steps = transitions
    actions = policy_steps.action
    return time_steps, actions, next_time_steps

  # Use @common.function in graph mode or for speeding up.
  def _train(self, experience, weights):
    with tf.GradientTape() as tape:
      loss_info = self._loss(
          experience,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights)
    tf.debugging.check_numerics(loss_info[0], 'Loss is inf or nan')
    variables_to_train = self._q_network.trainable_weights
    assert list(variables_to_train), "No variables in the agent's q_network."
    grads = tape.gradient(loss_info.loss, variables_to_train)
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(grads, variables_to_train))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    self._optimizer.apply_gradients(
        grads_and_vars, global_step=self.train_step_counter)

    self._update_target()

    return loss_info

  def _train_v1(self, experience, weights):
    with tf.GradientTape() as tape:
      loss_info = self._loss(
          experience,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights)
    tf.debugging.check_numerics(loss_info[0], 'Loss is inf or nan')
    variables_to_train = self._q_network.trainable_weights
    assert list(variables_to_train), "No variables in the agent's q_network."
    grads = tape.gradient(loss_info.loss, variables_to_train)
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(grads, variables_to_train))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    train_op = self._optimizer.apply_gradients(
        grads_and_vars, global_step=self.train_step_counter)

    update_op = self._update_target()
    train_op = tf.group(train_op, update_op)

    return train_op, loss_info

  def _loss(self,
            experience,
            td_errors_loss_fn=element_wise_huber_loss,
            gamma=1.0,
            reward_scale_factor=1.0,
            weights=None):
    """Computes loss for DQN training.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.policy.step_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.train_sequence_length` if that property is
        not `None`.
      td_errors_loss_fn: A function(td_targets, predictions) to compute the
        element wise loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output td_loss will be scaled by these weights, and the
        final scalar loss is the mean of these values.

    Returns:
      loss: An instance of `DqnLossInfo`.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    # Check that `experience` includes two outer dimensions [B, T, ...]. This
    # method requires `experience` to include the time dimension.
    self._check_trajectory_dimensions(experience)

    if self._n_step_update == 1:
      time_steps, actions, next_time_steps = self._experience_to_transitions(
          experience)
    else:
      # To compute n-step returns, we need the first time steps, the first
      # actions, and the last time steps. Therefore we extract the first and
      # last transitions from our Trajectory.
      first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
      last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
      time_steps, actions, _ = self._experience_to_transitions(first_two_steps)
      _, _, next_time_steps = self._experience_to_transitions(last_two_steps)

    with tf.name_scope('loss'):
      actions = tf.nest.flatten(actions)[0]
      q_values, _ = self._q_network(time_steps.observation,
                                    time_steps.step_type)

      # Handle action_spec.shape=(), and shape=(1,) by using the
      # multi_dim_actions param.
      multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.ndims > 0
      q_values = common.index_with_actions(
          q_values,
          tf.cast(actions, dtype=tf.int32),
          multi_dim_actions=multi_dim_actions)

      next_q_values = self._compute_next_q_values(next_time_steps)

      if self._n_step_update == 1:
        # Special case for n = 1 to avoid a loss of performance.
        td_targets = compute_td_targets(
            next_q_values,
            rewards=reward_scale_factor * next_time_steps.reward,
            discounts=gamma * next_time_steps.discount)
      else:
        # When computing discounted return, we need to throw out the last time
        # index of both reward and discount, which are filled with dummy values
        # to match the dimensions of the observation.
        # TODO(b/131557265): Replace value_ops.discounted_return with a method
        # that only computes the single value needed.
        n_step_return = value_ops.discounted_return(
            rewards=reward_scale_factor * experience.reward[:, :-1],
            discounts=gamma * experience.discount[:, :-1],
            final_value=next_q_values,
            time_major=False)

        # We only need the first value within the time dimension which
        # corresponds to the full final return. The remaining values are only
        # partial returns.
        td_targets = n_step_return[:, 0]

      valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
      td_error = valid_mask * (td_targets - q_values)

      td_loss = valid_mask * td_errors_loss_fn(td_targets, q_values)

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Do a sum over the time dimension.
        td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

      if weights is not None:
        td_loss *= weights

      # Average across the elements of the batch.
      # Note: We use an element wise loss above to ensure each element is always
      #   weighted by 1/N where N is the batch size, even when some of the
      #   weights are zero due to boundary transitions. Weighting by 1/K where K
      #   is the actual number of non-zero weight would artificially increase
      #   their contribution in the loss. Think about what would happen as
      #   the number of boundary samples increases.
      loss = tf.reduce_mean(input_tensor=td_loss)

      with tf.name_scope('Losses/'):
        tf.compat.v1.summary.scalar(
            'loss_' + self.name, loss, collections=['train_' + self.name])
        # family=self.name)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._q_network.trainable_weights:
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)

      if self._debug_summaries:
        diff_q_values = q_values - next_q_values
        common.generate_tensor_summaries('td_error', td_error,
                                         self.train_step_counter)
        common.generate_tensor_summaries('td_loss', td_loss,
                                         self.train_step_counter)
        common.generate_tensor_summaries('q_values', q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('next_q_values', next_q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                         self.train_step_counter)

      return tf_agent.LossInfo(loss,
                               DqnLossInfo(td_loss=td_loss, td_error=td_error))

  def _compute_next_q_values(self, next_time_steps):
    """Compute the q value of the next state for TD error computation.

    Args:
      next_time_steps: A batch of next timesteps

    Returns:
      A tensor of Q values for the given next state.
    """
    next_target_q_values, _ = self._target_q_network(
        next_time_steps.observation, next_time_steps.step_type)
    # Reduce_max below assumes q_values are [BxF] or [BxTxF]
    assert next_target_q_values.shape.ndims in [2, 3]
    return tf.reduce_max(input_tensor=next_target_q_values, axis=-1)


@gin.configurable
class DdqnAgent(DqnAgent):
  """A Double DQN Agent.

  Implements the Double-DQN algorithm from

  "Deep Reinforcement Learning with Double Q-learning"
   Hasselt et al., 2015
   https://arxiv.org/abs/1509.06461

  """

  def _compute_next_q_values(self, next_time_steps):
    """Compute the q value of the next state for TD error computation.

    Args:
      next_time_steps: A batch of next timesteps

    Returns:
      A tensor of Q values for the given next state.
    """
    # TODO(b/117175589): Add binary tests for DDQN.
    next_q_values, _ = self._q_network(next_time_steps.observation,
                                       next_time_steps.step_type)
    best_next_actions = tf.cast(
        tf.argmax(input=next_q_values, axis=-1), dtype=tf.int32)
    next_target_q_values, _ = self._target_q_network(
        next_time_steps.observation, next_time_steps.step_type)
    multi_dim_actions = best_next_actions.shape.ndims > 1
    return common.index_with_actions(
        next_target_q_values,
        best_next_actions,
        multi_dim_actions=multi_dim_actions)

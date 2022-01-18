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

# Lint as: python2, python3
"""A Soft Actor-Critic Agent.

Implements the Soft Actor-Critic (SAC) algorithm from
"Soft Actor-Critic Algorithms and Applications" by Haarnoja et al (2019).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import numpy as np
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity

SacLossInfo = collections.namedtuple(
    'SacLossInfo',
    ('critic_loss', 'actor_loss', 'alpha_loss', 'critic_no_entropy_loss'))


# TODO(b/148889463): deprecate std_clip_transform
@gin.configurable
def std_clip_transform(stddevs):
  stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 2),
                                  stddevs)
  return tf.exp(stddevs)


@gin.configurable
class SacAgent(tf_agent.TFAgent):
  """A SAC Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               critic_network,
               actor_network,
               actor_optimizer,
               critic_optimizer,
               alpha_optimizer,
               critic_network_no_entropy=None,
               critic_no_entropy_optimizer=None,
               actor_loss_weight=1.0,
               critic_loss_weight=0.5,
               alpha_loss_weight=1.0,
               actor_policy_ctor=actor_policy.ActorPolicy,
               critic_network_2=None,
               target_critic_network=None,
               target_critic_network_2=None,
               target_update_tau=1.0,
               target_update_period=1,
               td_errors_loss_fn=tf.math.squared_difference,
               gamma=1.0,
               reward_scale_factor=1.0,
               initial_log_alpha=0.0,
               use_log_alpha_in_alpha_loss=True,
               target_entropy=None,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name=None):
    """Creates a SAC Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      critic_network: A function critic_network((observations, actions)) that
        returns the q_values for each observation and action.
      actor_network: A function actor_network(observation, action_spec) that
        returns action distribution.
      actor_optimizer: The optimizer to use for the actor network.
      critic_optimizer: The default optimizer to use for the critic network.
      alpha_optimizer: The default optimizer to use for the alpha variable.
      actor_loss_weight: The weight on actor loss.
      critic_loss_weight: The weight on critic loss.
      alpha_loss_weight: The weight on alpha loss.
      actor_policy_ctor: The policy class to use.
      critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
        the second critic network during Q learning.  The weights from
        `critic_network` are copied if this is not provided.
      target_critic_network: (Optional.)  A `tf_agents.network.Network` to be
        used as the target critic network during Q learning. Every
        `target_update_period` train steps, the weights from `critic_network`
        are copied (possibly withsmoothing via `target_update_tau`) to `
        target_critic_network`.  If `target_critic_network` is not provided, it
        is created by making a copy of `critic_network`, which initializes a new
        network with the same structure and its own layers and weights.
        Performing a `Network.copy` does not work when the network instance
        already has trainable parameters (e.g., has already been built, or when
        the network is sharing layers with another).  In these cases, it is up
        to you to build a copy having weights that are not shared with the
        original `critic_network`, so that this can be used as a target network.
        If you provide a `target_critic_network` that shares any weights with
        `critic_network`, a warning will be logged but no exception is thrown.
      target_critic_network_2: (Optional.) Similar network as
        target_critic_network but for the critic_network_2. See documentation
        for target_critic_network. Will only be used if 'critic_network_2' is
        also specified.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      td_errors_loss_fn:  A function for computing the elementwise TD errors
        loss.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      initial_log_alpha: Initial value for log_alpha.
      use_log_alpha_in_alpha_loss: A boolean, whether using log_alpha or alpha
        in alpha loss. Certain implementations of SAC use log_alpha as log
        values are generally nicer to work with.
      target_entropy: The target average policy entropy, for updating alpha. The
        default value is negative of the total number of actions.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)

    self._check_action_spec(action_spec)

    self._critic_network_1 = critic_network
    self._critic_network_1.create_variables()
    if target_critic_network:
      target_critic_network.create_variables()
    self._target_critic_network_1 = (
        common.maybe_copy_target_network_with_checks(self._critic_network_1,
                                                     target_critic_network,
                                                     'TargetCriticNetwork1'))

    # for estimating critics without entropy included
    self._critic_network_no_entropy_1 = critic_network_no_entropy
    if critic_network_no_entropy is not None:
      self._critic_network_no_entropy_1.create_variables()
      self._target_critic_network_no_entropy_1 = (
          common.maybe_copy_target_network_with_checks(
              self._critic_network_no_entropy_1, None,
              'TargetCriticNetworkNoEntropy1'))
      # Network 2
      self._critic_network_no_entropy_2 = self._critic_network_no_entropy_1.copy(
          name='CriticNetworkNoEntropy2')
      self._critic_network_no_entropy_2.create_variables()
      self._target_critic_network_no_entropy_2 = (
          common.maybe_copy_target_network_with_checks(
              self._critic_network_no_entropy_2, None,
              'TargetCriticNetworkNoEntropy2'))

    if critic_network_2 is not None:
      self._critic_network_2 = critic_network_2
    else:
      self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
      # Do not use target_critic_network_2 if critic_network_2 is None.
      target_critic_network_2 = None
    self._critic_network_2.create_variables()
    if target_critic_network_2:
      target_critic_network_2.create_variables()
    self._target_critic_network_2 = (
        common.maybe_copy_target_network_with_checks(self._critic_network_2,
                                                     target_critic_network_2,
                                                     'TargetCriticNetwork2'))

    if actor_network:
      actor_network.create_variables()
    self._actor_network = actor_network

    policy = actor_policy_ctor(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        training=False)

    self._train_policy = actor_policy_ctor(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        training=True)

    self._log_alpha = common.create_variable(
        'initial_log_alpha',
        initial_value=initial_log_alpha,
        dtype=tf.float32,
        trainable=True)

    if target_entropy is None:
      target_entropy = self._get_default_target_entropy(action_spec)

    self._use_log_alpha_in_alpha_loss = use_log_alpha_in_alpha_loss
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer
    self._critic_no_entropy_optimizer = critic_no_entropy_optimizer
    self._alpha_optimizer = alpha_optimizer
    self._actor_loss_weight = actor_loss_weight
    self._critic_loss_weight = critic_loss_weight
    self._alpha_loss_weight = alpha_loss_weight
    self._td_errors_loss_fn = td_errors_loss_fn
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_entropy = target_entropy
    self._gradient_clipping = gradient_clipping
    self._debug_summaries = debug_summaries
    self._summarize_grads_and_vars = summarize_grads_and_vars
    self._update_target = self._get_target_updater(
        tau=self._target_update_tau, period=self._target_update_period)

    train_sequence_length = 2 if not critic_network.state_spec else None

    super(SacAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=train_sequence_length,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _check_action_spec(self, action_spec):
    flat_action_spec = tf.nest.flatten(action_spec)
    for spec in flat_action_spec:
      if spec.dtype.is_integer:
        raise NotImplementedError(
            'SacAgent does not currently support discrete actions. '
            'Action spec: {}'.format(action_spec))

  def _get_default_target_entropy(self, action_spec):
    # If target_entropy was not passed, set it to -dim(A)/2.0
    # Note that the original default entropy target is -dim(A) in the SAC paper.
    # However this formulation has also been used in practice by the original
    # authors and has in our experience been more stable for gym/mujoco.
    flat_action_spec = tf.nest.flatten(action_spec)
    target_entropy = -np.sum([
        np.product(single_spec.shape.as_list())
        for single_spec in flat_action_spec
    ]) / 2.0
    return target_entropy

  def _initialize(self):
    """Returns an op to initialize the agent.

    Copies weights from the Q networks to the target Q network.
    """
    common.soft_variables_update(
        self._critic_network_1.variables,
        self._target_critic_network_1.variables,
        tau=1.0)
    common.soft_variables_update(
        self._critic_network_2.variables,
        self._target_critic_network_2.variables,
        tau=1.0)
    if self._critic_network_no_entropy_1 is not None:
      common.soft_variables_update(
          self._critic_network_no_entropy_1.variables,
          self._target_critic_network_no_entropy_1.variables,
          tau=1.0)
      common.soft_variables_update(
          self._critic_network_no_entropy_2.variables,
          self._target_critic_network_no_entropy_2.variables,
          tau=1.0)

  def _train(self, experience, weights):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    squeeze_time_dim = not self._critic_network_1.state_spec
    time_steps, policy_steps, next_time_steps = (
        trajectory.experience_to_transitions(experience, squeeze_time_dim))
    actions = policy_steps.action

    trainable_critic_variables = list(
        object_identity.ObjectIdentitySet(
            self._critic_network_1.trainable_variables +
            self._critic_network_2.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self._critic_loss_weight * self.critic_loss(
          time_steps,
          actions,
          next_time_steps,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    critic_no_entropy_loss = None
    if self._critic_network_no_entropy_1 is not None:
      trainable_critic_no_entropy_variables = list(
          object_identity.ObjectIdentitySet(
              self._critic_network_no_entropy_1.trainable_variables +
              self._critic_network_no_entropy_2.trainable_variables))
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        assert trainable_critic_no_entropy_variables, (
            'No trainable critic_no_entropy variables to optimize.')
        tape.watch(trainable_critic_no_entropy_variables)
        critic_no_entropy_loss = self._critic_loss_weight * self.critic_no_entropy_loss(
            time_steps,
            actions,
            next_time_steps,
            td_errors_loss_fn=self._td_errors_loss_fn,
            gamma=self._gamma,
            reward_scale_factor=self._reward_scale_factor,
            weights=weights,
            training=True)

      tf.debugging.check_numerics(
          critic_no_entropy_loss,
          'Critic (without entropy) loss is inf or nan.')
      critic_no_entropy_grads = tape.gradient(
          critic_no_entropy_loss, trainable_critic_no_entropy_variables)
      self._apply_gradients(critic_no_entropy_grads,
                            trainable_critic_no_entropy_variables,
                            self._critic_no_entropy_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self._actor_loss_weight * self.actor_loss(
          time_steps, weights=weights)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    alpha_variable = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert alpha_variable, 'No alpha variable to optimize.'
      tape.watch(alpha_variable)
      alpha_loss = self._alpha_loss_weight * self.alpha_loss(
          time_steps, weights=weights)
    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_grads = tape.gradient(alpha_loss, alpha_variable)
    self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss_' + self.name,
          data=critic_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss_' + self.name,
          data=actor_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss_' + self.name,
          data=alpha_loss,
          step=self.train_step_counter)
      if critic_no_entropy_loss is not None:
        tf.compat.v2.summary.scalar(
            name='critic_no_entropy_loss_' + self.name,
            data=critic_no_entropy_loss,
            step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = critic_loss + actor_loss + alpha_loss
    if critic_no_entropy_loss is not None:
      total_loss += critic_no_entropy_loss

    extra = SacLossInfo(
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        critic_no_entropy_loss=critic_no_entropy_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)

  def _apply_gradients(self, gradients, variables, optimizer):
    # list(...) is required for Python3.
    grads_and_vars = list(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_target'):

      def update():
        """Update target network."""
        critic_update_1 = common.soft_variables_update(
            self._critic_network_1.variables,
            self._target_critic_network_1.variables,
            tau,
            tau_non_trainable=1.0)

        critic_2_update_vars = common.deduped_network_variables(
            self._critic_network_2, self._critic_network_1)

        target_critic_2_update_vars = common.deduped_network_variables(
            self._target_critic_network_2, self._target_critic_network_1)

        critic_update_2 = common.soft_variables_update(
            critic_2_update_vars,
            target_critic_2_update_vars,
            tau,
            tau_non_trainable=1.0)

        if self._critic_network_no_entropy_1 is None:
          return tf.group(critic_update_1, critic_update_2)
        else:
          critic_no_entropy_update_1 = common.soft_variables_update(
              self._critic_network_no_entropy_1.variables,
              self._target_critic_network_no_entropy_1.variables,
              tau,
              tau_non_trainable=1.0)

          critic_no_entropy_2_update_vars = common.deduped_network_variables(
              self._critic_network_no_entropy_2,
              self._critic_network_no_entropy_1)

          target_critic_no_entropy_2_update_vars = common.deduped_network_variables(
              self._target_critic_network_no_entropy_2,
              self._target_critic_network_no_entropy_1)

          critic_no_entropy_update_2 = common.soft_variables_update(
              critic_no_entropy_2_update_vars,
              target_critic_no_entropy_2_update_vars,
              tau,
              tau_non_trainable=1.0)

          return tf.group(critic_update_1, critic_update_2,
                          critic_no_entropy_update_1,
                          critic_no_entropy_update_2)

      return common.Periodically(update, period, 'update_targets')

  def _actions_and_log_probs(self, time_steps):
    """Get actions and corresponding log probabilities from policy."""
    # Get raw action distribution from policy, and initialize bijectors list.
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._train_policy.get_initial_state(batch_size)
    action_distribution = self._train_policy.distribution(
        time_steps, policy_state=policy_state).action

    # Sample actions and log_pis from transformed distribution.
    actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
    log_pi = common.log_probability(action_distribution, actions,
                                    self.action_spec)

    return actions, log_pi

  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  td_errors_loss_fn,
                  gamma=1.0,
                  reward_scale_factor=1.0,
                  weights=None,
                  training=False):
    """Computes the critic loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute
        elementwise (per-batch-entry) loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      training: Whether this loss is being used for training.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      nest_utils.assert_same_structure(actions, self.action_spec)
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)
      nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

      next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
      target_input = (next_time_steps.observation, next_actions)
      target_q_values1, unused_network_state1 = self._target_critic_network_1(
          target_input, next_time_steps.step_type, training=False)
      target_q_values2, unused_network_state2 = self._target_critic_network_2(
          target_input, next_time_steps.step_type, training=False)
      target_q_values = (
          tf.minimum(target_q_values1, target_q_values2) -
          tf.exp(self._log_alpha) * next_log_pis)

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_input = (time_steps.observation, actions)
      pred_td_targets1, _ = self._critic_network_1(
          pred_input, time_steps.step_type, training=training)
      pred_td_targets2, _ = self._critic_network_2(
          pred_input, time_steps.step_type, training=training)
      critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
      critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
      critic_loss = critic_loss1 + critic_loss2

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        critic_loss = tf.reduce_sum(input_tensor=critic_loss, axis=1)

      agg_loss = common.aggregate_losses(
          per_example_loss=critic_loss,
          sample_weight=weights,
          regularization_loss=(self._critic_network_1.losses +
                               self._critic_network_2.losses))
      critic_loss = agg_loss.total_loss

      self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                        pred_td_targets2)

      return critic_loss

  def critic_no_entropy_loss(self,
                             time_steps,
                             actions,
                             next_time_steps,
                             td_errors_loss_fn,
                             gamma=1.0,
                             reward_scale_factor=1.0,
                             weights=None,
                             training=False):
    """Computes the critic loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute
        elementwise (per-batch-entry) loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      training: Whether this loss is being used for training.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_no_entropy_loss'):
      nest_utils.assert_same_structure(actions, self.action_spec)
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)
      nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

      next_actions, _ = self._actions_and_log_probs(next_time_steps)
      target_input = (next_time_steps.observation, next_actions)
      target_q_values1, unused_network_state1 = self._target_critic_network_no_entropy_1(
          target_input, next_time_steps.step_type, training=False)
      target_q_values2, unused_network_state2 = self._target_critic_network_no_entropy_2(
          target_input, next_time_steps.step_type, training=False)
      target_q_values = tf.minimum(
          target_q_values1, target_q_values2
      )  # entropy has been removed from the target critic function

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_input = (time_steps.observation, actions)
      pred_td_targets1, _ = self._critic_network_no_entropy_1(
          pred_input, time_steps.step_type, training=training)
      pred_td_targets2, _ = self._critic_network_no_entropy_2(
          pred_input, time_steps.step_type, training=training)
      critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
      critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
      critic_loss = critic_loss1 + critic_loss2

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        critic_loss = tf.reduce_sum(input_tensor=critic_loss, axis=1)

      agg_loss = common.aggregate_losses(
          per_example_loss=critic_loss,
          sample_weight=weights,
          regularization_loss=(self._critic_network_no_entropy_1.losses +
                               self._critic_network_no_entropy_2.losses))
      critic_no_entropy_loss = agg_loss.total_loss

      self._critic_no_entropy_loss_debug_summaries(td_targets, pred_td_targets1,
                                                   pred_td_targets2)

      return critic_no_entropy_loss

  def actor_loss(self, time_steps, weights=None):
    """Computes the actor_loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)

      actions, log_pi = self._actions_and_log_probs(time_steps)
      target_input = (time_steps.observation, actions)
      target_q_values1, _ = self._critic_network_1(
          target_input, time_steps.step_type, training=False)
      target_q_values2, _ = self._critic_network_2(
          target_input, time_steps.step_type, training=False)
      target_q_values = tf.minimum(target_q_values1, target_q_values2)
      actor_loss = tf.exp(self._log_alpha) * log_pi - target_q_values
      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        actor_loss = tf.reduce_sum(input_tensor=actor_loss, axis=1)
      reg_loss = self._actor_network.losses if self._actor_network else None
      agg_loss = common.aggregate_losses(
          per_example_loss=actor_loss,
          sample_weight=weights,
          regularization_loss=reg_loss)
      actor_loss = agg_loss.total_loss
      self._actor_loss_debug_summaries(actor_loss, actions, log_pi,
                                       target_q_values, time_steps)

      return actor_loss

  def alpha_loss(self, time_steps, weights=None):
    """Computes the alpha_loss for EC-SAC training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      alpha_loss: A scalar alpha loss.
    """
    with tf.name_scope('alpha_loss'):
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)

      unused_actions, log_pi = self._actions_and_log_probs(time_steps)
      entropy_diff = tf.stop_gradient(-log_pi - self._target_entropy)
      if self._use_log_alpha_in_alpha_loss:
        alpha_loss = (self._log_alpha * entropy_diff)
      else:
        alpha_loss = (tf.exp(self._log_alpha) * entropy_diff)

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        alpha_loss = tf.reduce_sum(input_tensor=alpha_loss, axis=1)
      else:
        alpha_loss = tf.expand_dims(alpha_loss, 0)

      agg_loss = common.aggregate_losses(
          per_example_loss=alpha_loss, sample_weight=weights)
      alpha_loss = agg_loss.total_loss

      self._alpha_loss_debug_summaries(alpha_loss, entropy_diff)

      return alpha_loss

  def _critic_loss_debug_summaries(self, td_targets, pred_td_targets1,
                                   pred_td_targets2):
    if self._debug_summaries:
      td_errors1 = td_targets - pred_td_targets1
      td_errors2 = td_targets - pred_td_targets2
      td_errors = tf.concat([td_errors1, td_errors2], axis=0)
      common.generate_tensor_summaries('td_errors', td_errors,
                                       self.train_step_counter)
      common.generate_tensor_summaries('td_targets', td_targets,
                                       self.train_step_counter)
      common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                       self.train_step_counter)
      common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                       self.train_step_counter)

  def _critic_no_entropy_loss_debug_summaries(self, td_targets,
                                              pred_td_targets1,
                                              pred_td_targets2):
    if self._debug_summaries:
      td_errors1 = td_targets - pred_td_targets1
      td_errors2 = td_targets - pred_td_targets2
      td_errors = tf.concat([td_errors1, td_errors2], axis=0)
      common.generate_tensor_summaries('td_errors_no_entropy_critic', td_errors,
                                       self.train_step_counter)
      common.generate_tensor_summaries('td_targets_no_entropy_critic',
                                       td_targets, self.train_step_counter)
      common.generate_tensor_summaries('pred_td_targets1_no_entropy_critic',
                                       pred_td_targets1,
                                       self.train_step_counter)
      common.generate_tensor_summaries('pred_td_targets2_no_entropy_critic',
                                       pred_td_targets2,
                                       self.train_step_counter)

  def _actor_loss_debug_summaries(self, actor_loss, actions, log_pi,
                                  target_q_values, time_steps):
    if self._debug_summaries:
      common.generate_tensor_summaries('actor_loss', actor_loss,
                                       self.train_step_counter)
      try:
        common.generate_tensor_summaries('actions', actions,
                                         self.train_step_counter)
      except ValueError:
        pass  # Guard against internal SAC variants that do not directly
        # generate actions.

      common.generate_tensor_summaries('log_pi', log_pi,
                                       self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='entropy_avg',
          data=-tf.reduce_mean(input_tensor=log_pi),
          step=self.train_step_counter)
      common.generate_tensor_summaries('target_q_values', target_q_values,
                                       self.train_step_counter)
      batch_size = nest_utils.get_outer_shape(time_steps,
                                              self._time_step_spec)[0]
      policy_state = self._train_policy.get_initial_state(batch_size)
      action_distribution = self._train_policy.distribution(
          time_steps, policy_state).action
      if isinstance(action_distribution, tfp.distributions.Normal):
        common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                         self.train_step_counter)
        common.generate_tensor_summaries('act_stddev',
                                         action_distribution.scale,
                                         self.train_step_counter)
      elif isinstance(action_distribution, tfp.distributions.Categorical):
        common.generate_tensor_summaries('act_mode', action_distribution.mode(),
                                         self.train_step_counter)
      try:
        common.generate_tensor_summaries('entropy_action',
                                         action_distribution.entropy(),
                                         self.train_step_counter)
      except NotImplementedError:
        pass  # Some distributions do not have an analytic entropy.

  def _alpha_loss_debug_summaries(self, alpha_loss, entropy_diff):
    if self._debug_summaries:
      common.generate_tensor_summaries('alpha_loss', alpha_loss,
                                       self.train_step_counter)
      common.generate_tensor_summaries('entropy_diff', entropy_diff,
                                       self.train_step_counter)

      tf.compat.v2.summary.scalar(
          name='log_alpha', data=self._log_alpha, step=self.train_step_counter)

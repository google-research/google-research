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

"""Twin Delayed Deep Deterministic policy gradient (TD3) agent.

TD3 extends DDPG by adding an extra critic network and using the minimum of the
two critic values to reduce overestimation bias.

"Addressing Function Approximation Error in Actor-Critic Methods"
by Fujimoto et al.

For the full paper, see https://arxiv.org/abs/1802.09477.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.policies import gaussian_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity


class Td3Info(collections.namedtuple('Td3Info', ('actor_loss', 'critic_loss'))):
  pass


@gin.configurable
class Td3Agent(tf_agent.TFAgent):
  """A TD3 Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               critic_network,
               actor_optimizer,
               critic_optimizer,
               exploration_noise_std=0.1,
               critic_network_2=None,
               target_actor_network=None,
               target_critic_network=None,
               target_critic_network_2=None,
               target_update_tau=1.0,
               target_update_period=1,
               actor_update_period=1,
               dqda_clipping=None,
               td_errors_loss_fn=None,
               gamma=1.0,
               reward_scale_factor=1.0,
               target_policy_noise=0.2,
               target_policy_noise_clip=0.5,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name=None):
    """Creates a Td3Agent Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type).
      critic_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, action, step_type).
      actor_optimizer: The default optimizer to use for the actor network.
      critic_optimizer: The default optimizer to use for the critic network.
      exploration_noise_std: Scale factor on exploration policy noise.
      critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
        the second critic network during Q learning.  The weights from
        `critic_network` are copied if this is not provided.
      target_actor_network: (Optional.)  A `tf_agents.network.Network` to be
        used as the target actor network during Q learning. Every
        `target_update_period` train steps, the weights from `actor_network` are
        copied (possibly withsmoothing via `target_update_tau`) to `
        target_actor_network`.  If `target_actor_network` is not provided, it is
        created by making a copy of `actor_network`, which initializes a new
        network with the same structure and its own layers and weights.
        Performing a `Network.copy` does not work when the network instance
        already has trainable parameters (e.g., has already been built, or when
        the network is sharing layers with another).  In these cases, it is up
        to you to build a copy having weights that are not shared with the
        original `actor_network`, so that this can be used as a target network.
        If you provide a `target_actor_network` that shares any weights with
        `actor_network`, a warning will be logged but no exception is thrown.
      target_critic_network: (Optional.) Similar network as target_actor_network
        but for the critic_network. See documentation for target_actor_network.
      target_critic_network_2: (Optional.) Similar network as
        target_actor_network but for the critic_network_2. See documentation for
        target_actor_network. Will only be used if 'critic_network_2' is also
        specified.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      actor_update_period: Period for the optimization step on actor network.
      dqda_clipping: A scalar or float clips the gradient dqda element-wise
        between [-dqda_clipping, dqda_clipping]. Default is None representing no
        clippiing.
      td_errors_loss_fn:  A function for computing the TD errors loss. If None,
        a default value of elementwise huber_loss is used.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      target_policy_noise: Scale factor on target action noise
      target_policy_noise_clip: Value to clip noise.
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
    self._actor_network = actor_network
    actor_network.create_variables()
    if target_actor_network:
      target_actor_network.create_variables()
    self._target_actor_network = common.maybe_copy_target_network_with_checks(
        self._actor_network, target_actor_network, 'TargetActorNetwork')

    self._critic_network_1 = critic_network
    critic_network.create_variables()
    if target_critic_network:
      target_critic_network.create_variables()
    self._target_critic_network_1 = (
        common.maybe_copy_target_network_with_checks(self._critic_network_1,
                                                     target_critic_network,
                                                     'TargetCriticNetwork1'))

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

    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer

    self._exploration_noise_std = exploration_noise_std
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_update_period = actor_update_period
    self._dqda_clipping = dqda_clipping
    self._td_errors_loss_fn = (
        td_errors_loss_fn or common.element_wise_huber_loss)
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_policy_noise = target_policy_noise
    self._target_policy_noise_clip = target_policy_noise_clip
    self._gradient_clipping = gradient_clipping

    self._update_target = self._get_target_updater(target_update_tau,
                                                   target_update_period)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        clip=True)
    collect_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        clip=False)
    collect_policy = gaussian_policy.GaussianPolicy(
        collect_policy, scale=self._exploration_noise_std, clip=True)

    super(Td3Agent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=2 if not self._actor_network.state_spec else None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _initialize(self):
    """Initialize the agent.

    Copies weights from the actor and critic networks to the respective
    target actor and critic networks.
    """
    common.soft_variables_update(
        self._critic_network_1.variables,
        self._target_critic_network_1.variables,
        tau=1.0)
    common.soft_variables_update(
        self._critic_network_2.variables,
        self._target_critic_network_2.variables,
        tau=1.0)
    common.soft_variables_update(
        self._actor_network.variables,
        self._target_actor_network.variables,
        tau=1.0)

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target networks are updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets'):

      def update():  # pylint: disable=missing-docstring
        # TODO(b/124381161): What about observation normalizer variables?
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

        actor_update_vars = common.deduped_network_variables(
            self._actor_network, self._critic_network_1, self._critic_network_2)
        target_actor_update_vars = common.deduped_network_variables(
            self._target_actor_network, self._target_critic_network_1,
            self._target_critic_network_2)

        actor_update = common.soft_variables_update(
            actor_update_vars,
            target_actor_update_vars,
            tau,
            tau_non_trainable=1.0)
        return tf.group(critic_update_1, critic_update_2, actor_update)

      return common.Periodically(update, period, 'update_targets')

  def _train(self, experience, weights=None):
    # TODO(b/120034503): Move the conversion to transitions to the base class.
    squeeze_time_dim = not self._actor_network.state_spec
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
      critic_loss = self.critic_loss(
          time_steps, actions, next_time_steps, weights=weights, training=True)
    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self.actor_loss(time_steps, weights=weights, training=True)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

    # We only optimize the actor every actor_update_period training steps.
    def optimize_actor():
      actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
      return self._apply_gradients(actor_grads, trainable_actor_variables,
                                   self._actor_optimizer)

    remainder = tf.math.mod(self.train_step_counter, self._actor_update_period)
    tf.cond(
        pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)

    self.train_step_counter.assign_add(1)
    self._update_target()

    # TODO(b/124382360): Compute per element TD loss and return in loss_info.
    total_loss = actor_loss + critic_loss

    return tf_agent.LossInfo(total_loss, Td3Info(actor_loss, critic_loss))

  def _apply_gradients(self, gradients, variables, optimizer):
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    return optimizer.apply_gradients(grads_and_vars)

  @common.function
  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  weights=None,
                  training=False):
    """Computes the critic loss for TD3 training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
      training: Whether this loss is being used for training.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      target_actions, _ = self._target_actor_network(
          next_time_steps.observation,
          next_time_steps.step_type,
          training=training)

      # Add gaussian noise to each action before computing target q values
      def add_noise_to_action(action):  # pylint: disable=missing-docstring
        dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                        scale=self._target_policy_noise * \
                                        tf.ones_like(action))
        noise = dist.sample()
        noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                                 self._target_policy_noise_clip)
        return action + noise

      noisy_target_actions = tf.nest.map_structure(add_noise_to_action,
                                                   target_actions)

      # Target q-values are the min of the two networks
      target_q_input_1 = (next_time_steps.observation, noisy_target_actions)
      target_q_values_1, _ = self._target_critic_network_1(
          target_q_input_1, next_time_steps.step_type, training=False)
      target_q_input_2 = (next_time_steps.observation, noisy_target_actions)
      target_q_values_2, _ = self._target_critic_network_2(
          target_q_input_2, next_time_steps.step_type, training=False)
      target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

      td_targets = tf.stop_gradient(
          self._reward_scale_factor * next_time_steps.reward +
          self._gamma * next_time_steps.discount * target_q_values)

      pred_input_1 = (time_steps.observation, actions)
      pred_td_targets_1, _ = self._critic_network_1(
          pred_input_1, time_steps.step_type, training=training)
      pred_input_2 = (time_steps.observation, actions)
      pred_td_targets_2, _ = self._critic_network_2(
          pred_input_2, time_steps.step_type, training=training)
      pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]

      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='td_targets', data=td_targets, step=self.train_step_counter)
        with tf.name_scope('td_targets'):
          tf.compat.v2.summary.scalar(
              name='mean',
              data=tf.reduce_mean(input_tensor=td_targets),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              name='max',
              data=tf.reduce_max(input_tensor=td_targets),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              name='min',
              data=tf.reduce_min(input_tensor=td_targets),
              step=self.train_step_counter)

        for td_target_idx in range(2):
          pred_td_targets = pred_td_targets_all[td_target_idx]
          td_errors = td_targets - pred_td_targets
          with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
            tf.compat.v2.summary.histogram(
                name='td_errors', data=td_errors, step=self.train_step_counter)
            tf.compat.v2.summary.histogram(
                name='pred_td_targets',
                data=pred_td_targets,
                step=self.train_step_counter)
            with tf.name_scope('td_errors'):
              tf.compat.v2.summary.scalar(
                  name='mean',
                  data=tf.reduce_mean(input_tensor=td_errors),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='mean_abs',
                  data=tf.reduce_mean(input_tensor=tf.abs(td_errors)),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='max',
                  data=tf.reduce_max(input_tensor=td_errors),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='min',
                  data=tf.reduce_min(input_tensor=td_errors),
                  step=self.train_step_counter)
            with tf.name_scope('pred_td_targets'):
              tf.compat.v2.summary.scalar(
                  name='mean',
                  data=tf.reduce_mean(input_tensor=pred_td_targets),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='max',
                  data=tf.reduce_max(input_tensor=pred_td_targets),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='min',
                  data=tf.reduce_min(input_tensor=pred_td_targets),
                  step=self.train_step_counter)

      critic_loss = (
          self._td_errors_loss_fn(td_targets, pred_td_targets_1) +
          self._td_errors_loss_fn(td_targets, pred_td_targets_2))
      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        critic_loss = tf.reduce_sum(input_tensor=critic_loss, axis=1)

      if weights is not None:
        critic_loss *= weights

      critic_loss = tf.reduce_mean(input_tensor=critic_loss)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='critic_loss', data=critic_loss, step=self.train_step_counter)

      return critic_loss

  @common.function
  def actor_loss(self, time_steps, weights=None, training=False):
    """Computes the actor_loss for TD3 training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
      training: Whether this loss is being used for training.
      # TODO(b/124383618): Add an action norm regularizer.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      actions, _ = self._actor_network(
          time_steps.observation, time_steps.step_type, training=training)
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(actions)
        q_values, _ = self._critic_network_1((time_steps.observation, actions),
                                             time_steps.step_type,
                                             training=False)
        actions = tf.nest.flatten(actions)

      dqdas = tape.gradient([q_values], actions)

      actor_losses = []
      for dqda, action in zip(dqdas, actions):
        if self._dqda_clipping is not None:
          dqda = tf.clip_by_value(dqda, -1 * self._dqda_clipping,
                                  self._dqda_clipping)
        loss = common.element_wise_squared_loss(
            tf.stop_gradient(dqda + action), action)
        if nest_utils.is_batched_nested_tensors(
            time_steps, self.time_step_spec, num_outer_dims=2):
          # Sum over the time dimension.
          loss = tf.reduce_sum(loss, axis=1)
        if weights is not None:
          loss *= weights
        loss = tf.reduce_mean(loss)
        actor_losses.append(loss)

      actor_loss = tf.add_n(actor_losses)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='actor_loss', data=actor_loss, step=self.train_step_counter)

    return actor_loss

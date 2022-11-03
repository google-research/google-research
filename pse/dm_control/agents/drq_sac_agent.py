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

"""Implements the changes required for SAC for the DrQ Algorithm.

Reference paper:
https://arxiv.org/pdf/2004.13649.pdf
"""

import gin
import numpy as np
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.policies import actor_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class DrQSacAgent(sac_agent.SacAgent):
  """A SAC Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               critic_network,
               actor_network,
               actor_optimizer,
               critic_optimizer,
               alpha_optimizer,
               actor_loss_weight=1.0,
               critic_loss_weight=1.0,
               alpha_loss_weight=1.0,
               actor_policy_ctor=actor_policy.ActorPolicy,
               critic_network_2=None,
               target_critic_network=None,
               target_critic_network_2=None,
               actor_update_frequency=2,
               target_update_tau=1.0,
               target_update_period=1,
               td_errors_loss_fn=tf.math.squared_difference,
               gamma=1.0,
               reward_scale_factor=1.0,
               initial_log_alpha=np.log(0.1),
               use_log_alpha_in_alpha_loss=False,
               num_augmentations=2,
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
      actor_update_frequency: Frequency to calculate and update the actor and
        alpha values.
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
      num_augmentations: K value in the original paper.
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
    super(DrQSacAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        critic_network=critic_network,
        actor_network=actor_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        actor_loss_weight=actor_loss_weight,
        critic_loss_weight=critic_loss_weight,
        alpha_loss_weight=alpha_loss_weight,
        actor_policy_ctor=actor_policy_ctor,
        critic_network_2=critic_network_2,
        target_critic_network=target_critic_network,
        target_critic_network_2=target_critic_network_2,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        initial_log_alpha=initial_log_alpha,
        use_log_alpha_in_alpha_loss=use_log_alpha_in_alpha_loss,
        target_entropy=target_entropy,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name)

    self._actor_update_frequency = actor_update_frequency

    # We use the observation in the trajectory as the first augmentation.
    self._num_augmentations = num_augmentations
    if num_augmentations > 1:
      augmented_obs_spec = tuple([
          dict(pixels=self.collect_data_spec.observation['pixels'])
          for _ in range(num_augmentations - 1)
      ])

      self._train_argspec = dict(
          augmented_obs=augmented_obs_spec,
          augmented_next_obs=augmented_obs_spec)

  def _initialize(self):
    # Call the target updater once here to avoid having it trigger on the first
    # train call. Instead it will start counding down the target_update_period
    # when training begins.
    self._update_target()
    super(DrQSacAgent, self)._initialize()

  @common.function(autograph=True)
  def _train(self, experience, weights, augmented_obs=None,
             augmented_next_obs=None):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object. If augmentations > 1 then a
        tuple of the form: ``` (trajectory, [augmentation_1, ... ,
          augmentation_{K-1}]) ``` is expected.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      augmented_obs: List of length num_augmentations - 1 of random crops of the
        trajectory's observation.
      augmented_next_obs: List of length num_augmentations - 1 of random crops
        of the trajectory's next_observation.

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

    trainable_critic_variables = (
        self._critic_network_1.trainable_variables +
        self._critic_network_2.trainable_variables)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)

      critic_loss = self._critic_loss_weight * self.critic_loss(
          time_steps,
          actions,
          next_time_steps,
          augmented_obs,
          augmented_next_obs,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    total_loss = critic_loss
    actor_loss = tf.constant(0.0, tf.float32)
    alpha_loss = tf.constant(0.0, tf.float32)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)

    # Only perform actor and alpha updates periodically
    if self.train_step_counter % self._actor_update_frequency == 0:
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
            name='actor_loss', data=actor_loss, step=self.train_step_counter)
        tf.compat.v2.summary.scalar(
            name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

      total_loss = critic_loss + actor_loss + alpha_loss

    self.train_step_counter.assign_add(1)
    self._update_target()

    # NOTE: Consider keeping track of previous actor/alpha loss.
    extra = sac_agent.SacLossInfo(
        critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)

  @common.function(autograph=True)
  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  augmented_obs,
                  augmented_next_obs,
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
      augmented_obs: List of observations.
      augmented_next_obs: List of next_observations.
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

      td_targets = self._compute_td_targets(next_time_steps,
                                            reward_scale_factor, gamma)

      # Compute td_targets with augmentations.
      for i in range(self._num_augmentations - 1):
        augmented_next_time_steps = next_time_steps._replace(
            observation=augmented_next_obs[i])

        augmented_td_targets = self._compute_td_targets(
            augmented_next_time_steps, reward_scale_factor, gamma)

        td_targets = td_targets + augmented_td_targets

      # Average td_target estimation over augmentations.
      if self._num_augmentations > 1:
        td_targets = td_targets / self._num_augmentations

      pred_td_targets1, pred_td_targets2, critic_loss = (
          self._compute_prediction_critic_loss(
              (time_steps.observation, actions), td_targets, time_steps,
              training, td_errors_loss_fn))

      # Add Q Augmentations to the critic loss.
      for i in range(self._num_augmentations - 1):
        augmented_time_steps = time_steps._replace(observation=augmented_obs[i])
        _, _, loss = (
            self._compute_prediction_critic_loss(
                (augmented_time_steps.observation, actions), td_targets,
                augmented_time_steps, training, td_errors_loss_fn))
        critic_loss = critic_loss + loss

      agg_loss = common.aggregate_losses(
          per_example_loss=critic_loss,
          sample_weight=weights,
          regularization_loss=(self._critic_network_1.losses +
                               self._critic_network_2.losses))
      critic_loss = agg_loss.total_loss

      self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                        pred_td_targets2)

      return critic_loss

  def _compute_td_targets(self, next_time_steps, reward_scale_factor, gamma):
    next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
    target_input = (next_time_steps.observation, next_actions)
    target_q_values1, unused_network_state1 = self._target_critic_network_1(
        target_input, next_time_steps.step_type, training=False)
    target_q_values2, unused_network_state2 = self._target_critic_network_2(
        target_input, next_time_steps.step_type, training=False)
    target_q_values = (
        tf.minimum(target_q_values1, target_q_values2) -
        tf.exp(self._log_alpha) * next_log_pis)

    return tf.stop_gradient(reward_scale_factor * next_time_steps.reward +
                            gamma * next_time_steps.discount * target_q_values)

  def _compute_prediction_critic_loss(self, pred_input, td_targets, time_steps,
                                      training, td_errors_loss_fn):
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
    return pred_td_targets1, pred_td_targets2, critic_loss

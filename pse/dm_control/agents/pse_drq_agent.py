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

"""Helpers for DrQ SAC."""

import collections

import gin
import numpy as np
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from pse.dm_control.agents import drq_sac_agent
from pse.dm_control.utils import helper_utils
from pse.dm_control.utils import model_utils

SacContrastiveLossInfo = collections.namedtuple(
    'SacContrastiveLossInfo',
    ('critic_loss', 'actor_loss', 'alpha_loss', 'contrastive_loss'))


def image_aug(traj, meta, img_pad, num_augmentations):
  """Padding and cropping."""
  paddings = tf.constant([[img_pad, img_pad], [img_pad, img_pad], [0, 0]])

  obs = traj.observation['pixels'][0]
  cropped_shape = obs.shape
  next_obs = traj.observation['pixels'][1]
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in tf. Use 'SYMMETRIC' instead.
  obs = tf.pad(obs, paddings, 'SYMMETRIC')
  next_obs = tf.pad(next_obs, paddings, 'SYMMETRIC')

  def get_random_crop(padded_obs):
    return tf.image.random_crop(padded_obs, cropped_shape)

  aug_obs = tf.expand_dims(get_random_crop(obs), 0)
  aug_next_obs = tf.expand_dims(get_random_crop(next_obs), 0)
  traj.observation['pixels'] = tf.concat([aug_obs, aug_next_obs], axis=0)

  augmented_obs = []
  augmented_next_obs = []

  for _ in range(num_augmentations - 1):
    augmented_obs.append(dict(pixels=get_random_crop(obs)))
    augmented_next_obs.append(dict(pixels=get_random_crop(next_obs)))

  return dict(
      experience=traj,
      augmented_obs=tuple(augmented_obs),
      augmented_next_obs=tuple(augmented_next_obs)), meta


@gin.configurable
class DrQSacModifiedAgent(drq_sac_agent.DrQSacAgent):

  def __init__(self,
               time_step_spec,
               action_spec,
               critic_network,
               actor_network,
               actor_optimizer,
               critic_optimizer,
               alpha_optimizer,
               contrastive_optimizer,
               actor_loss_weight=1.0,
               critic_loss_weight=1.0,
               alpha_loss_weight=1.0,
               contrastive_loss_weight=1.0,
               contrastive_loss_temperature=0.5,
               actor_policy_ctor=model_utils.ActorPolicy,
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
      contrastive_optimizer: The default optimizer to use for contrastive loss.
      actor_loss_weight: The weight on actor loss.
      critic_loss_weight: The weight on critic loss.
      alpha_loss_weight: The weight on alpha loss.
      contrastive_loss_weight: The weight on contrastive loss.
      contrastive_loss_temperature: The temperature used in contrastive loss.
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
    super(DrQSacModifiedAgent, self).__init__(
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

    self._validate_args = False
    self._contrastive_optimizer = contrastive_optimizer
    self._contrastive_loss_weight = contrastive_loss_weight
    self._contrastive_loss_temperature = contrastive_loss_temperature

  def contrastive_metric_loss(self, data_tuple, return_representation=False,
                              add_summary=True):
    out = helper_utils.representation_alignment_loss(
        self._actor_network,
        data_tuple,
        use_coupling_weights=True,
        temperature=self._contrastive_loss_temperature,
        return_representation=return_representation)

    if add_summary:
      cme_loss = out[0] if return_representation else out
      with tf.name_scope('Losses'):
        tf.compat.v2.summary.scalar(
            name='contrastive_loss',
            data=cme_loss,
            step=self.train_step_counter)

    return out

  @common.function(autograph=True)
  def _train(self,
             experience,
             weights,
             episode_data=None,
             augmented_obs=None,
             augmented_next_obs=None):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object. If augmentations > 1 then a
        tuple of the form: ``` (trajectory, [augmentation_1, ... ,
          augmentation_{K-1}]) ``` is expected.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      episode_data: Tuple of (episode, episode, metric) for contrastive loss.
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

    # Contrastive loss for PSEs
    contrastive_loss = 0.0
    if self._contrastive_loss_weight > 0:
      contrastive_vars = self._actor_network.encoder_variables
      with tf.GradientTape(
          watch_accessed_variables=True, persistent=True) as tape:
        contrastive_loss = (
            self._contrastive_loss_weight *
            self.contrastive_metric_loss(episode_data))
      total_loss = total_loss + contrastive_loss
      tf.debugging.check_numerics(contrastive_loss,
                                  'Contrastive loss is inf or nan.')

      contrastive_grads = tape.gradient(contrastive_loss, contrastive_vars)
      self._apply_gradients(contrastive_grads, contrastive_vars,
                            self._contrastive_optimizer)
      del tape

    self.train_step_counter.assign_add(1)
    self._update_target()

    # NOTE: Consider keeping track of previous actor/alpha loss.
    extra = SacContrastiveLossInfo(
        critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss,
        contrastive_loss=contrastive_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)

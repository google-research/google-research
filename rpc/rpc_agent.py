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

"""TFAgents that implement RPC."""

import gin
from six.moves import range
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks import utils
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity


class RpAgent(sac_agent.SacAgent):
  """Implements the RPC agent."""

  def _train(self, experience, weights):
    """Modifies the default _train step in two ways.

      1. Passes actions and next time steps to actor loss.
      2. Clips the dual parameter.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.
    """
    transition = self._as_transition(experience)
    time_steps, policy_steps, next_time_steps = transition
    actions = policy_steps.action

    trainable_critic_variables = list(object_identity.ObjectIdentitySet(
        self._critic_network_1.trainable_variables +
        self._critic_network_2.trainable_variables))

    tf.debugging.check_numerics(
        tf.reduce_mean(time_steps.reward), 'ts.reward is inf or nan.')
    tf.debugging.check_numerics(
        tf.reduce_mean(next_time_steps.reward), 'next_ts.reward is inf or nan.')
    tf.debugging.check_numerics(
        tf.reduce_mean(actions), 'Actions is inf or nan.')

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self._critic_loss_weight*self.critic_loss(
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

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self._actor_loss_weight*self.actor_loss(
          time_steps, actions, next_time_steps, weights=weights)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    alpha_variable = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert alpha_variable, 'No alpha variable to optimize.'
      tape.watch(alpha_variable)
      alpha_loss = self._alpha_loss_weight*self.alpha_loss(
          time_steps, weights=weights)
    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_grads = tape.gradient(alpha_loss, alpha_variable)
    self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = critic_loss + actor_loss + alpha_loss

    extra = sac_agent.SacLossInfo(
        critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

    return LossInfo(loss=total_loss, extra=extra)

  @gin.configurable
  def actor_loss(self,
                 time_steps,
                 actions,
                 next_time_steps,
                 weights=None):
    """Computes the actor_loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      actor_loss: A scalar actor loss.
    """
    prev_time_steps, prev_actions, time_steps = time_steps, actions, next_time_steps  # pylint: disable=line-too-long
    with tf.name_scope('actor_loss'):
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)

      actions, log_pi = self._actions_and_log_probs(time_steps)
      target_input = (time_steps.observation, actions)
      target_q_values1, _ = self._critic_network_1(
          target_input, step_type=time_steps.step_type, training=False)
      target_q_values2, _ = self._critic_network_2(
          target_input, step_type=time_steps.step_type, training=False)
      target_q_values = tf.minimum(target_q_values1, target_q_values2)
      actor_loss = tf.exp(self._log_alpha) * log_pi - target_q_values

      ### Flatten time dimension. We'll add it back when adding the loss.
      num_outer_dims = nest_utils.get_outer_rank(time_steps,
                                                 self.time_step_spec)
      has_time_dim = (num_outer_dims == 2)
      if has_time_dim:
        batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
        obs = batch_squash.flatten(time_steps.observation)
        prev_obs = batch_squash.flatten(prev_time_steps.observation)
        prev_actions = batch_squash.flatten(prev_actions)
      else:
        obs = time_steps.observation
        prev_obs = prev_time_steps.observation
      z = self._actor_network._z_encoder(obs, training=True)  # pylint: disable=protected-access
      prior = self._actor_network._predictor((prev_obs, prev_actions),  # pylint: disable=protected-access
                                             training=True)

      # kl is a vector of length batch_size, which has already been summed over
      # the latent dimension z.
      kl = tfp.distributions.kl_divergence(z, prior)
      if has_time_dim:
        kl = batch_squash.unflatten(kl)

      kl_coef = tf.stop_gradient(
          tf.exp(self._actor_network._log_kl_coefficient))  # pylint: disable=protected-access
      # The actor loss trains both the predictor and the encoder.
      actor_loss += kl_coef * kl

      if actor_loss.shape.rank > 1:
        # Sum over the time dimension.
        actor_loss = tf.reduce_sum(
            actor_loss, axis=range(1, actor_loss.shape.rank))
      reg_loss = self._actor_network.losses if self._actor_network else None
      agg_loss = common.aggregate_losses(
          per_example_loss=actor_loss,
          sample_weight=weights,
          regularization_loss=reg_loss)
      actor_loss = agg_loss.total_loss
      self._actor_loss_debug_summaries(actor_loss, actions, log_pi,
                                       target_q_values, time_steps)
      tf.compat.v2.summary.scalar(
          name='encoder_kl',
          data=tf.reduce_mean(kl),
          step=self.train_step_counter)

      return actor_loss

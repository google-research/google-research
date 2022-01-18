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

"""Behavioral Clonning training."""
import typing

from dm_env import specs as dm_env_specs
import tensorflow as tf

import tensorflow_probability as tfp

from rl_repr.batch_rl.action_embed import ActionFourierLearner
from rl_repr.batch_rl.embed import EmbedNet
from rl_repr.batch_rl.embed import StochasticEmbedNet

tfd = tfp.distributions


class LatentBehavioralCloning(object):
  """Training class for behavioral clonning."""

  def __init__(self,
               state_dim,
               action_spec,
               embed_model,
               hidden_dims = (256, 256),
               finetune=False,
               finetune_primitive=True,
               kl_regularizer=None,
               latent_bc_lr_decay=None,
               stochastic=True,
               learning_rate=None):
    self.action_spec = action_spec
    assert embed_model is not None
    self.embed_model = embed_model
    self.finetune = finetune
    self.finetune_primitive = finetune_primitive
    self.kl_regularizer = kl_regularizer
    self.stochastic = stochastic

    if stochastic:
      self.policy = StochasticEmbedNet(
          state_dim,
          embedding_dim=self.embed_model.latent_dim,
          hidden_dims=hidden_dims)
    else:
      self.policy = EmbedNet(
          state_dim,
          embedding_dim=self.embed_model.latent_dim,
          hidden_dims=hidden_dims)

    if hasattr(self.embed_model, 'log_alpha'):
      self.log_alpha = self.embed_model.log_alpha
    else:
      self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.target_entropy = -self.action_spec.shape[0]

    self.trainable_variables = self.policy.trainable_variables
    if self.finetune:
      self.trainable_variables += self.embed_model.embedder.variables
    if self.finetune_primitive:
      self.trainable_variables += self.embed_model.primitive_policy.trainable_variables
    if self.kl_regularizer:
      self.trainable_variables += [self.log_alpha]

    if self.kl_regularizer == 'posterior':
      self.posterior = StochasticEmbedNet(
          state_dim,
          embedding_dim=self.embed_model.latent_dim,
          hidden_dims=hidden_dims)
      self.trainable_variables += self.posterior.variables

    learning_rate = learning_rate or 1e-4
    if latent_bc_lr_decay:
      boundaries = [180_000, 190_000]
      values = [
          learning_rate, learning_rate / latent_bc_lr_decay,
          learning_rate / latent_bc_lr_decay / latent_bc_lr_decay
      ]
      learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries, values)

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,)

  @tf.function
  def update_step(self, dataset_iter):
    """Performs a single training step.

    Args:
      dataset_iter: Iterator over dataset samples.

    Returns:
      Dictionary with losses to track.
    """
    states, actions, _, _, _ = next(dataset_iter)
    assert len(states.shape) == 3
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.trainable_variables)

      if isinstance(self.embed_model, ActionFourierLearner):
        if self.stochastic:
          embed = self.embed_model(
              states, actions, stop_gradient=(not self.finetune))
          mean_logvar = self.policy(
              states[:, 0, :], stop_gradient=False, sample=False)
          mean, logvar = tf.split(mean_logvar, 2, axis=-1)
          dist = tfd.MultivariateNormalDiag(
              loc=mean, scale_diag=tf.exp(0.5 * logvar))
          latent_loss = -dist.log_prob(embed)
        else:
          embed = self.embed_model(
              states, actions, stop_gradient=(not self.finetune))
          latent_action = self.policy(states[:, 0, :], stop_gradient=False)
          latent_loss = tf.reduce_sum((embed - latent_action)**2, axis=-1)

        primitive_loss = -self.embed_model.primitive_policy.log_probs(
            tf.concat([states[:, 0, :], embed], axis=-1), actions[:, 0, :])
      else:
        assert self.stochastic
        embed = self.embed_model(
            states, actions, stop_gradient=(not self.finetune))
        mean_logvar = self.policy(
            states[:, 0, :], stop_gradient=False, sample=False)
        mean, logvar = tf.split(mean_logvar, 2, axis=-1)
        dist = tfd.MultivariateNormalDiag(
            loc=mean, scale_diag=tf.exp(0.5 * logvar))
        latent_loss = -dist.log_prob(embed)

        batch_size = tf.shape(states)[0]
        policy_input_obs = tf.concat([
            tf.reshape(states,
                       [batch_size * self.embed_model.sequence_length, -1]),
            tf.repeat(embed, self.embed_model.sequence_length, axis=0)
        ], -1)
        policy_input_act = tf.reshape(
            actions, [batch_size * self.embed_model.sequence_length, -1])
        policy_log_probs = self.embed_model.primitive_policy.log_probs(
            policy_input_obs, policy_input_act)
        primitive_loss = -tf.reduce_sum(
            tf.reshape(policy_log_probs,
                       [batch_size, self.embed_model.sequence_length]), -1)

      alpha = tf.exp(self.log_alpha)
      kl_loss = tf.convert_to_tensor(0.)
      alpha_loss = tf.convert_to_tensor(0.)

      if self.kl_regularizer == 'uniform':
        _, log_probs = self.embed_model.primitive_policy(
            tf.concat([states[:, 0, :], embed], axis=-1),
            sample=True,
            with_log_probs=True)
        primitive_loss += tf.stop_gradient(alpha) * log_probs
        alpha_loss = alpha * tf.stop_gradient(-log_probs - self.target_entropy)
      elif self.kl_regularizer == 'prior':
        mean_logvar = self.policy(
            states[:, 0, :], stop_gradient=False, sample=False)
        z_mean, z_logvar = tf.split(mean_logvar, 2, axis=-1)
        prior_mean, prior_logvar = self.embed_model.prior_mean_logvar(
            states[:, 0, :])
        kl_loss = -tf.stop_gradient(alpha) * 0.5 * tf.reduce_sum(
            1.0 + z_logvar - prior_logvar -
            tf.exp(-1 * prior_logvar) * tf.pow(z_mean - prior_mean, 2) -
            tf.exp(z_logvar - prior_logvar), -1)
        alpha_loss = alpha * tf.stop_gradient(kl_loss - self.target_entropy)
      elif self.kl_regularizer == 'posterior':
        mean_logvar = self.policy(
            states[:, 0, :], stop_gradient=False, sample=False)
        z_mean, z_logvar = tf.split(mean_logvar, 2, axis=-1)
        posterior = self.posterior(
            states[:, 0, :], stop_gradient=False, sample=False)
        posterior_mean, posterior_logvar = tf.split(posterior, 2, axis=-1)
        kl_loss = -tf.stop_gradient(alpha) * 0.5 * tf.reduce_sum(
            1.0 + z_logvar - posterior_logvar -
            tf.exp(-1 * posterior_logvar) * tf.pow(z_mean - posterior_mean, 2) -
            tf.exp(z_logvar - posterior_logvar), -1)
        alpha_loss = alpha * tf.stop_gradient(kl_loss - self.target_entropy)

      loss = tf.reduce_mean(latent_loss) + tf.reduce_mean(
          primitive_loss) + tf.reduce_mean(kl_loss) + tf.reduce_mean(alpha_loss)

    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    return {
        'loss': tf.reduce_mean(loss),
        'latent_loss': tf.reduce_mean(latent_loss),
        'kl_loss': tf.reduce_mean(kl_loss),
        'alpha': alpha,
        'alpha_loss': tf.reduce_mean(alpha_loss),
        'primitive_loss': tf.reduce_mean(primitive_loss),
    }

  def act(self,
          states,
          actions=None,  # pylint: disable=unused-argument
          rewards=None,  # pylint: disable=unused-argument
          latent_action=None):
    states = tf.cast(states, tf.float32)
    if latent_action is None:
      latent_action = self.policy(states, stop_gradient=False)
    return self.embed_model.primitive_policy(
        tf.concat([states, latent_action], axis=-1),
        sample=False), latent_action

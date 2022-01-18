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

# python3
"""Embedding for state action representation learning."""

import typing

from dm_env import specs as dm_env_specs
import tensorflow as tf

from rl_repr.batch_rl import policies
from rl_repr.batch_rl.embed import EmbedNet
from rl_repr.batch_rl.embed import RNNEmbedNet
from rl_repr.batch_rl.embed import StochasticEmbedNet
from rl_repr.batch_rl.embed import StochasticRNNEmbedNet


class ActionFourierLearner(tf.keras.Model):
  """A learner for state-action Fourier Features approximated EBM."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               fourier_dim=None,
               sequence_length=2,
               hidden_dims = (256, 256),
               shuffle_rate = 0.1,
               mixup_rate = 0.,
               kl_regularizer = None,
               learning_rate = None):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      fourier_dim: Fourier feature size.
      sequence_length: Context length.
      hidden_dims: List of hidden dimensions.
      shuffle_rate: Rate of shuffled embeddings.
      mixup_rate: Rate of mixup embeddings.
      kl_regularizer: Apply uniform KL to action decoder.
      learning_rate: Learning rate.
    """
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_spec.shape[0]
    self.embedding_dim = embedding_dim
    self.fourier_dim = fourier_dim
    self.latent_dim = self.fourier_dim or self.embedding_dim
    self.sequence_length = sequence_length
    self.shuffle_rate = shuffle_rate
    self.mixup_rate = mixup_rate
    self.kl_regularizer = kl_regularizer

    self.embedder = EmbedNet(
        self.state_dim +
        (self.action_dim if self.sequence_length == 2 else self.embedding_dim),
        embedding_dim=self.embedding_dim,
        hidden_dims=hidden_dims)
    self.next_embedder = EmbedNet(
        state_dim, embedding_dim=self.embedding_dim, hidden_dims=hidden_dims)

    self.trajectory_embedder = RNNEmbedNet(
        [self.sequence_length, self.action_dim + state_dim],
        embedding_dim=self.embedding_dim)

    self.primitive_policy = policies.DiagGuassianPolicy(
        state_dim + (self.fourier_dim or self.embedding_dim),
        action_spec,
        hidden_dims=hidden_dims)

    learning_rate = learning_rate or 3e-4
    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)  #, beta_1=0.0)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.target_entropy = -action_spec.shape[0]

    if self.fourier_dim:
      self.omega = tf.Variable(
          tf.random.normal([self.fourier_dim, self.embedding_dim]),
          trainable=False)
      self.shift = tf.Variable(
          tf.random.uniform([self.fourier_dim], minval=0, maxval=2 * 3.14159),
          trainable=False)
      self.average_embed = tf.Variable(
          tf.zeros([self.embedding_dim]), trainable=False)
      self.average_square = tf.Variable(
          tf.ones([self.embedding_dim]), trainable=False)

    self.pretrain_variables = (
        self.embedder.variables + self.next_embedder.variables +
        self.primitive_policy.variables + self.trajectory_embedder.variables +
        [self.log_alpha])

  def fourier_project(self, embed, update_moving_averages=True):
    average_embed = self.average_embed
    average_square = self.average_square
    stddev_embed = tf.sqrt(tf.maximum(1e-8, average_square - average_embed**2))
    normalized_omegas = self.omega / stddev_embed[None, :]
    projection = tf.matmul(
        embed - tf.stop_gradient(average_embed),
        tf.stop_gradient(normalized_omegas)[:, :],
        transpose_b=True)
    projection /= self.embedding_dim**0.5
    embed_linear = tf.math.cos(projection + tf.stop_gradient(self.shift))
    if update_moving_averages:
      self.update_moving_averages(embed)
    return embed_linear

  def update_moving_averages(self, embed):
    tt = 0.0005
    _ = self.average_embed.assign((1 - tt) * self.average_embed +
                                  tt * tf.reduce_mean(embed, [0])),
    _ = self.average_square.assign((1 - tt) * self.average_square +
                                   tt * tf.reduce_mean(embed**2, [0]))

  @tf.function
  def call(self,
           states,
           actions,
           rewards = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: 3 dimensional state tensors.
      actions: 3 dimensional action tensors.
      rewards: Optional rewards.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    if self.sequence_length == 2:
      trajectory = actions[:, 0, :]
    else:
      trajectory = self.trajectory_embedder(
          tf.concat([states, actions], axis=-1), stop_gradient=stop_gradient)
    embed = self.embedder(
        tf.concat([states[:, 0, :], trajectory], axis=-1),
        stop_gradient=stop_gradient)
    if self.fourier_dim:
      embed = self.fourier_project(embed, update_moving_averages=False)
    return embed

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes energies between (embedding, other_embedding).

    Args:
      embeddings: B x d
      other_embeddings: B x d

    Returns:
      Energy.
    """
    energies = tf.reduce_sum(
        -tf.square(embeddings[:, None, :] - other_embeddings[None, :, :]),
        axis=-1)
    return energies

  def fit(self, states,
          actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.pretrain_variables)

      this_state = states[:, 0, :]
      this_action = actions[:, 0, :]
      if self.sequence_length == 2:
        trajectory = this_action
      else:
        trajectory = self.trajectory_embedder(
            tf.concat([states, actions], axis=-1), stop_gradient=False)
      this_embed = self.embedder(
          tf.concat([this_state, trajectory], axis=-1), stop_gradient=False)
      next_embed = self.next_embedder(states[:, 1, :], stop_gradient=False)

      if self.shuffle_rate > 0:
        shuffle1 = tf.random.shuffle(tf.range(tf.shape(this_embed)[0]))
        shuffle2 = tf.random.shuffle(tf.range(tf.shape(next_embed)[0]))
        rand1 = tf.random.uniform([tf.shape(this_embed)[0]])
        rand2 = tf.random.uniform([tf.shape(next_embed)[0]])
        this_embed = tf.where(rand1[:, None] < self.shuffle_rate,
                              tf.gather(this_embed, shuffle1), this_embed)
        this_state = tf.where(rand1[:, None] < self.shuffle_rate,
                              tf.gather(this_state, shuffle1), this_state)
        this_action = tf.where(rand1[:, None] < self.shuffle_rate,
                               tf.gather(this_action, shuffle1), this_action)
        next_embed = tf.where(rand2[:, None] < self.shuffle_rate,
                              tf.gather(next_embed, shuffle2), next_embed)

      if self.mixup_rate > 0:
        shuffle1 = tf.random.shuffle(tf.range(tf.shape(this_embed)[0]))
        shuffle2 = tf.random.shuffle(tf.range(tf.shape(next_embed)[0]))
        this_mixed = (
            1 - self.mixup_rate) * this_embed + self.mixup_rate * tf.gather(
                this_embed, shuffle1)
        next_mixed = (
            1 - self.mixup_rate) * next_embed + self.mixup_rate * tf.gather(
                next_embed, shuffle2)
        embeddings1 = tf.concat([this_embed, this_mixed], axis=0)
        embeddings2 = tf.concat([next_embed, next_mixed], axis=0)
      else:
        embeddings1 = this_embed
        embeddings2 = next_embed

      energies = self.compute_energy(embeddings1, embeddings2)
      pos_loss = tf.linalg.diag_part(energies)
      neg_loss = tf.reduce_logsumexp(energies, -1)

      model_loss = -pos_loss + neg_loss
      correct = tf.cast(pos_loss >= tf.reduce_max(energies, axis=-1),
                        tf.float32)

      if self.fourier_dim:
        this_embed = self.fourier_project(this_embed)

      primitive_policy_in = tf.concat([this_state, this_embed], axis=-1)
      recon_loss = -self.primitive_policy.log_probs(primitive_policy_in,
                                                    this_action)
      if self.kl_regularizer:
        _, policy_log_probs = self.primitive_policy(
            primitive_policy_in, sample=True, with_log_probs=True)
        alpha = tf.exp(self.log_alpha)
        alpha_loss = alpha * tf.stop_gradient(-policy_log_probs -
                                              self.target_entropy)
        recon_loss += tf.stop_gradient(alpha) * policy_log_probs
      else:
        alpha = tf.convert_to_tensor(0.)
        alpha_loss = tf.convert_to_tensor(0.)

      loss = tf.reduce_mean(model_loss) + tf.reduce_mean(
          recon_loss) + tf.reduce_mean(alpha_loss)

    grads = tape.gradient(loss, self.pretrain_variables)

    self.optimizer.apply_gradients(zip(grads, self.pretrain_variables))

    return {
        'loss': loss,
        'model_loss': tf.reduce_mean(model_loss),
        'recon_loss': tf.reduce_mean(recon_loss),
        'alpha_loss': tf.reduce_mean(alpha_loss),
        'alpha': alpha,
        'pos': tf.reduce_mean(pos_loss),
        'neg': tf.reduce_mean(neg_loss),
        'correct': tf.reduce_mean(correct),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)


class ActionOpalLearner(tf.keras.Model):
  """A learner for OPAL latent actions."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               latent_dim = 8,
               sequence_length = 2,
               embed_state = False,
               action_only = False,
               learning_rate = None):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      latent_dim: Latent action dim.
      sequence_length: Context length.
      embed_state: Also embed state.
      action_only: Only input actions to trajectory embedder.
      learning_rate: Learning rate.
    """
    super().__init__()
    self.input_dim = state_dim
    self.latent_dim = latent_dim
    self.sequence_length = sequence_length
    self.action_only = action_only
    self.embed_state = embed_state

    self.embedder = StochasticEmbedNet(
        state_dim, embedding_dim=embedding_dim, hidden_dims=hidden_dims)
    self.prior = StochasticEmbedNet(
        embedding_dim if embed_state else state_dim,
        embedding_dim=latent_dim,
        hidden_dims=hidden_dims)
    self.primitive_policy = policies.DiagGuassianPolicy(state_dim + latent_dim,
                                                        action_spec)

    action_dim = action_spec.shape[0]
    self.trajectory_embedder = StochasticRNNEmbedNet(
        [self.sequence_length, action_dim + (0 if action_only else state_dim)],
        embedding_dim=latent_dim)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)  #, beta_1=0.0)

    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions,
           rewards = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: 3 dimensional state tensors.
      actions: 3 dimensional action tensors.
      rewards: Optional rewards.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    assert len(states.shape) == 3
    if self.action_only:
      trajectory = actions
    else:
      trajectory = tf.concat([states, actions], -1)
    trajectory_embedding = self.trajectory_embedder(
        trajectory, stop_gradient=stop_gradient, sample=False)
    z_mean, z_logvar = tf.split(trajectory_embedding, 2, axis=-1)
    z_sample = z_mean + tf.random.normal(tf.shape(z_mean)) * tf.exp(
        0.5 * z_logvar)
    return z_sample

  @tf.function
  def prior_mean_logvar(self, initial_states):
    if self.embed_state:
      state_embedding = self.embedder(
          initial_states, stop_gradient=True, sample=False)
      embed_mean, embed_logvar = tf.split(state_embedding, 2, axis=-1)
      initial_states = embed_mean + tf.random.normal(
          tf.shape(embed_mean)) * tf.exp(0.5 * embed_logvar)

    prior = self.prior(initial_states, stop_gradient=True, sample=False)
    prior_mean, prior_logvar = tf.split(prior, 2, axis=-1)
    return prior_mean, prior_logvar

  def fit(self, states,
          actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      initial_state = states[:, 0, :]

      if self.embed_state:
        state_embedding = self.embedder(
            initial_state, stop_gradient=False, sample=False)
        embed_mean, embed_logvar = tf.split(state_embedding, 2, axis=-1)
        embed_kl_loss = -0.5 * tf.reduce_sum(
            1.0 + embed_logvar - tf.pow(embed_mean, 2) - tf.exp(embed_logvar),
            -1)
        initial_state = embed_mean + tf.random.normal(
            tf.shape(embed_mean)) * tf.exp(0.5 * embed_logvar)

      prior = self.prior(initial_state, stop_gradient=False, sample=False)
      prior_mean, prior_logvar = tf.split(prior, 2, axis=-1)

      if self.action_only:
        trajectory = actions
      else:
        trajectory = tf.concat([states, actions], -1)
      trajectory_embedding = self.trajectory_embedder(
          trajectory, stop_gradient=False, sample=False)
      z_mean, z_logvar = tf.split(trajectory_embedding, 2, axis=-1)
      z_sample = z_mean + tf.random.normal(tf.shape(z_mean)) * tf.exp(
          0.5 * z_logvar)

      policy_input_obs = tf.concat([
          tf.reshape(states, [batch_size * self.sequence_length, -1]),
          tf.repeat(z_sample, self.sequence_length, axis=0)
      ], -1)
      policy_input_act = tf.reshape(actions,
                                    [batch_size * self.sequence_length, -1])
      policy_log_probs = self.primitive_policy.log_probs(
          policy_input_obs, policy_input_act)
      reconstruct_loss = -tf.reduce_sum(
          tf.reshape(policy_log_probs, [batch_size, self.sequence_length]), -1)
      prior_kl_loss = -0.5 * tf.reduce_sum(
          1.0 + z_logvar - prior_logvar - tf.exp(-1 * prior_logvar) *
          tf.pow(z_mean - prior_mean, 2) - tf.exp(z_logvar - prior_logvar), -1)

      loss = tf.reduce_mean(1.0 * reconstruct_loss + 0.1 * prior_kl_loss)

      if self.embed_state:
        loss += tf.reduce_mean(0.1 * embed_kl_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(zip(grads, self.all_variables))

    return {
        'loss': loss,
        'reconstruct_loss': tf.reduce_mean(reconstruct_loss),
        'prior_kl_loss': tf.reduce_mean(prior_kl_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

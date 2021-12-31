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

"""DeepMDP implementation following Gelada et al. (2019).
"""
import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import losses
from representation_batch_rl.representation_batch_rl import tf_utils


class SuperModelLearner(tf.keras.Model):
  """A learner for model-based representation learning.

  Encompasses forward models, inverse models, as well as latent models like
  DeepMDP.
  """

  def __init__(
      self,
      observation_spec,
      action_spec,
      embedding_dim = 256,
      num_distributions=None,
      hidden_dims = (256, 256),
      sequence_length = 2,
      learning_rate=None,
      latent_dim = 256,
      reward_weight = 1.0,
      forward_weight
       = 1.0,  # Predict last state given prev actions/states.
      inverse_weight = 1.0,  # Predict last action given states.
      state_prediction_mode = 'energy',
      num_augmentations = 0,
      rep_learn_keywords = 'outer',
      batch_size = 256):
    """Creates networks.

    Args:
      observation_spec: State spec.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions for discrete
        embedding.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
      latent_dim: Dimension of the latent variable.
      reward_weight: Weight on the reward loss.
      forward_weight: Weight on the forward loss.
      inverse_weight: Weight on the inverse loss.
      state_prediction_mode: One of ['latent', 'energy'].
      num_augmentations: Num of random crops
      rep_learn_keywords: Representation learning loss to add.
      batch_size: Batch size
    """
    super().__init__()
    action_dim = action_spec.maximum.item() + 1
    self.observation_spec = observation_spec
    self.action_dim = action_dim
    self.action_spec = action_spec
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    self.sequence_length = sequence_length
    self.latent_dim = latent_dim
    self.reward_weight = reward_weight
    self.forward_weight = forward_weight
    self.inverse_weight = inverse_weight
    self.state_prediction_mode = state_prediction_mode
    self.num_augmentations = num_augmentations
    self.rep_learn_keywords = rep_learn_keywords.split('__')
    self.batch_size = batch_size
    self.tau = 0.005
    self.discount = 0.99

    critic_kwargs = {}

    if observation_spec.shape == (64, 64, 3):
      # IMPALA for Procgen
      def conv_stack():
        return make_impala_cnn_network(
            depths=[16, 32, 32], use_batch_norm=False, dropout_rate=0.)

      state_dim = 256
    else:
      # Reduced architecture for DMC
      def conv_stack():
        return ConvStack(observation_spec.shape)
      state_dim = 50

    conv_stack_critic = conv_stack()
    conv_target_stack_critic = conv_stack()

    if observation_spec.shape == (64, 64, 3):
      conv_stack_critic.output_size = state_dim
      conv_target_stack_critic.output_size = state_dim

    critic_kwargs['encoder'] = ImageEncoder(
        conv_stack_critic, feature_dim=state_dim, bprop_conv_stack=True)
    critic_kwargs['encoder_target'] = ImageEncoder(
        conv_target_stack_critic, feature_dim=state_dim, bprop_conv_stack=True)

    self.embedder = tf_utils.EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    if self.sequence_length > 2:
      self.latent_embedder = tf_utils.RNNEmbedNet(
          [self.sequence_length - 2, self.embedding_dim + self.embedding_dim],
          embedding_dim=self.latent_dim)

    self.reward_decoder = tf_utils.EmbedNet(
        self.latent_dim + self.embedding_dim + self.embedding_dim,
        embedding_dim=1,
        hidden_dims=hidden_dims)

    forward_decoder_out = (
        self.embedding_dim if (self.state_prediction_mode
                               in ['latent', 'energy']) else self.input_dim)
    forward_decoder_dists = (
        self.num_distributions if
        (self.state_prediction_mode in ['latent', 'energy']) else None)
    self.forward_decoder = tf_utils.StochasticEmbedNet(
        self.latent_dim + self.embedding_dim + self.embedding_dim,
        embedding_dim=forward_decoder_out,
        num_distributions=forward_decoder_dists,
        hidden_dims=hidden_dims)

    self.weight = tf.Variable(tf.eye(self.embedding_dim))

    self.action_encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                self.embedding_dim, use_bias=True
            ),  # , kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.embedding_dim)
        ],
        name='action_encoder')

    if self.num_augmentations == 0:
      dummy_state = tf.constant(
          np.zeros(shape=[1] + list(observation_spec.shape)))
      self.obs_spec = list(observation_spec.shape)
    else:  # account for padding of +4 everywhere and then cropping out 68
      dummy_state = tf.constant(np.zeros(shape=[1, 68, 68, 3]))
      self.obs_spec = [68, 68, 3]

    @tf.function
    def init_models():
      critic_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder_target'](dummy_state)
      self.action_encoder(
          tf.cast(tf.one_hot([1], depth=action_dim), tf.float32))

    init_models()

    hidden_dims = (256, 256)
    # self.actor = policies.CategoricalPolicy(state_dim, action_spec,
    #  hidden_dims=hidden_dims, encoder=actor_kwargs['encoder'])

    self.critic = critic.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder'],
        discrete_actions=True,
        linear='linear_Q' in self.rep_learn_keywords)
    self.critic_target = critic.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder_target'],
        discrete_actions=True,
        linear='linear_Q' in self.rep_learn_keywords)

    @tf.function
    def init_models2():
      dummy_state = tf.zeros((1, 68, 68, 3), dtype=tf.float32)
      phi_s = self.critic.encoder(dummy_state)
      phi_a = tf.eye(15, dtype=tf.float32)
      if 'linear_Q' in self.rep_learn_keywords:
        _ = self.critic.critic1.state_encoder(phi_s)
        _ = self.critic.critic2.state_encoder(phi_s)
        _ = self.critic.critic1.action_encoder(phi_a)
        _ = self.critic.critic2.action_encoder(phi_a)
        _ = self.critic_target.critic1.state_encoder(phi_s)
        _ = self.critic_target.critic2.state_encoder(phi_s)
        _ = self.critic_target.critic1.action_encoder(phi_a)
        _ = self.critic_target.critic2.action_encoder(phi_a)

    init_models2()

    critic.soft_update(self.critic, self.critic_target, tau=1.0)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)

    self.all_variables = (
        self.embedder.trainable_variables +
        self.reward_decoder.trainable_variables +
        self.forward_decoder.trainable_variables +
        self.action_encoder.trainable_variables +
        self.critic.trainable_variables +
        self.critic_target.trainable_variables)

    self.model_dict = {
        'action_encoder': self.action_encoder,
        'weight': self.weight,
        'forward_decoder': self.forward_decoder,
        'reward_decoder': self.reward_decoder,
        'embedder': self.embedder,
        'critic': self.critic,
        'critic_target': self.critic_target,
        'critic_optimizer': self.critic_optimizer,
        'optimizer': self.optimizer
    }

  @tf.function
  def call(self,
           states,
           actions=None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of states.
      actions: Optional actions
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    features = self.critic.encoder(states)
    return self.embedder(features, stop_gradient=stop_gradient)

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between (embedding, other_embedding).

    Args:
      embeddings: Tensor of embedding vectors
      other_embeddings: Tensor of future embeddings

    Returns:
      energy
    """
    transformed_embeddings = tf.matmul(embeddings, self.weight)
    energies = tf.matmul(
        transformed_embeddings, other_embeddings, transpose_b=True)
    return energies

  def fit_embedding(self, states, actions,
                    next_states, next_actions,
                    rewards,
                    discounts):
    """Updates critic parameters.

    Args:
      states: Batch of states.
      actions: Batch of actions.
      next_states: Batch of next states.
      next_actions: Batch of next actions
      rewards: Batch of rewards.
      discounts: Batch of masks indicating the end of the episodes.

    Returns:
      Dictionary with information to track.
    """
    states = tf.transpose(
        tf.stack([states, next_states])[:, 0], (1, 0, 2, 3, 4))
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      actions = tf.transpose(
          tf.one_hot(tf.stack([actions, next_actions]), depth=self.action_dim),
          (1, 0, 2))
      actions = tf.reshape(actions,
                           [batch_size * self.sequence_length, self.action_dim])
      actions = self.action_encoder(actions)
      actions = tf.reshape(
          actions, [batch_size, self.sequence_length, self.embedding_dim])

      all_states = tf.reshape(states, [batch_size * self.sequence_length] +
                              self.obs_spec)
      all_features = self.critic.encoder(all_states)
      all_embeddings = self.embedder(all_features, stop_gradient=False)
      embeddings = tf.reshape(
          all_embeddings,
          [batch_size, self.sequence_length, self.embedding_dim])

      if self.sequence_length > 2:
        latent_embedder_in = tf.concat(
            [embeddings[:, :-2, :], actions[:, :-2, :]], -1)
        latent = self.latent_embedder(latent_embedder_in, stop_gradient=False)
      else:
        latent = tf.zeros([batch_size, self.latent_dim])

      reward_decoder_in = tf.concat(
          [latent, embeddings[:, -2, :], actions[:, -2, :]], -1)
      reward_pred = self.reward_decoder(reward_decoder_in, stop_gradient=False)
      reward_loss = tf.square(rewards - reward_pred[Ellipsis, 0])

      forward_decoder_in = tf.concat(
          [latent, embeddings[:, -2, :], actions[:, -2, :]], -1)
      forward_pred_sample, forward_pred_raw = self.forward_decoder(
          forward_decoder_in,
          sample=True,
          sample_and_raw_output=True,
          stop_gradient=False)
      if self.state_prediction_mode in ['latent', 'energy']:
        true_sample = embeddings[:, -1, :]
      elif self.state_prediction_mode == 'raw':
        true_sample = states[:, -1, :]
      else:
        assert False, 'bad prediction mode'

      if self.state_prediction_mode in ['latent', 'raw']:
        if self.num_distributions and self.state_prediction_mode == 'latent':
          forward_loss = losses.categorical_kl(true_sample, forward_pred_raw)
        else:
          forward_pred_mean, forward_pred_logvar = tf.split(
              forward_pred_raw, 2, axis=-1)
          forward_pred_dist = tfp.distributions.MultivariateNormalDiag(
              forward_pred_mean, tf.exp(0.5 * forward_pred_logvar))
          forward_loss = -forward_pred_dist.log_prob(true_sample)
      else:
        energies = self.compute_energy(forward_pred_sample, true_sample)
        positive_loss = tf.linalg.diag_part(energies)
        negative_loss = tf.reduce_logsumexp(energies, axis=-1)

        forward_loss = -positive_loss + negative_loss

      loss = tf.reduce_mean(
          # alpha_loss +
          self.reward_weight * reward_loss +
          # self.inverse_weight * inverse_loss +
          self.forward_weight * forward_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
        # 'alpha': alpha,
        # 'alpha_loss': tf.reduce_mean(alpha_loss),
        'reward_loss': tf.reduce_mean(reward_loss),
        # 'inverse_loss': tf.reduce_mean(inverse_loss),
        'forward_loss': tf.reduce_mean(forward_loss),
    }

  def fit_critic(self, states, actions,
                 next_states, next_actions,
                 rewards,
                 discounts):
    """Updates critic parameters.

    Args:
      states: Batch of states.
      actions: Batch of actions.
      next_states: Batch of next states.
      next_actions: Batch of next actions from training policy.
      rewards: Batch of rewards.
      discounts: Batch of masks indicating the end of the episodes.

    Returns:
      Dictionary with information to track.
    """
    action_indices = tf.stack(
        [tf.range(tf.shape(actions)[0], dtype=tf.int64), actions], axis=-1)
    next_action_indices = tf.stack(
        [tf.range(tf.shape(next_actions)[0], dtype=tf.int64), next_actions],
        axis=-1)

    if self.num_augmentations > 0:
      target_q = 0.
      for i in range(self.num_augmentations):
        next_q1_i, next_q2_i = self.critic_target(next_states[i], actions=None)
        target_q_i = tf.expand_dims(
            rewards, 1) + self.discount * tf.expand_dims(
                discounts, 1) * tf.minimum(next_q1_i, next_q2_i)
        target_q += target_q_i
      target_q /= self.num_augmentations
    else:
      next_q1, next_q2 = self.critic_target(next_states, actions=None)
      target_q = tf.expand_dims(rewards, 1) + self.discount * tf.expand_dims(
          discounts, 1) * tf.minimum(next_q1, next_q2)

    target_q = tf.gather_nd(target_q, indices=next_action_indices)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables)

      if self.num_augmentations > 0:
        critic_loss = 0.
        q1 = 0.
        q2 = 0.
        for i in range(self.num_augmentations):
          q1_i, q2_i = self.critic(
              states[i], stop_grad_features=True, actions=None)
          critic_loss_i = (
              tf.losses.mean_squared_error(
                  target_q, tf.gather_nd(q1_i, indices=action_indices)) +
              tf.losses.mean_squared_error(
                  target_q, tf.gather_nd(q2_i, indices=action_indices)))
          q1 += q1_i
          q2 += q2_i
          critic_loss += critic_loss_i
        q1 /= self.num_augmentations
        q2 /= self.num_augmentations
        critic_loss /= self.num_augmentations
      else:
        q1, q2 = self.critic(states, stop_grad_features=True, actions=None)

      critic_loss = (
          tf.losses.mean_squared_error(
              target_q, tf.gather_nd(q1, indices=action_indices)) +
          tf.losses.mean_squared_error(
              target_q, tf.gather_nd(q2, indices=action_indices)))

    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.trainable_variables))

    critic.soft_update(self.critic, self.critic_target, tau=self.tau)

    return {
        'q1': tf.reduce_mean(q1),
        'q2': tf.reduce_mean(q2),
        'critic_loss': critic_loss
    }

  @tf.function
  def update_step(self,
                  replay_buffer_iter,
                  train_target='both'):
    """Performs a single training step for critic and embedding.

    Args:
      replay_buffer_iter: A tensorflow graph iteratable object.
      train_target: string specifying whether update RL and or representation

    Returns:
      Dictionary with losses to track.
    """
    transition = next(replay_buffer_iter)
    numpy_dataset = isinstance(replay_buffer_iter, np.ndarray)
    # observation: n_batch x n_timesteps x 1 x H*W*3*n_frames x 1 ->
    # n_batch x H x W x 3*n_frames
    if not numpy_dataset:
      states = transition.observation[:, 0]
      next_states = transition.observation[:, 1]
      actions = transition.action[:, 0]
      rewards = transition.reward[:, 0]
      discounts = transition.discount[:, 0]

      if transition.observation.dtype == tf.uint8:
        states = tf.cast(states, tf.float32) / 255.
        next_states = tf.cast(next_states, tf.float32) / 255.
    else:
      states, actions, rewards, next_states, discounts = transition

    if self.num_augmentations > 0:
      states, next_states = tf_utils.image_aug(
          states,
          next_states,
          img_pad=4,
          num_augmentations=self.num_augmentations,
          obs_dim=64,
          channels=3,
          cropped_shape=[self.batch_size, 68, 68, 3])

    next_actions = self.act(next_states, data_aug=True)

    if train_target == 'both':
      ssl_dict = self.fit_embedding(states, actions, next_states, next_actions,
                                    rewards, discounts)
      critic_dict = self.fit_critic(states, actions, next_states, next_actions,
                                    rewards, discounts)
    elif train_target == 'encoder':
      ssl_dict = self.fit_embedding(states, actions, next_states, next_actions,
                                    rewards, discounts)
      critic_dict = {}
    elif train_target == 'rl':
      ssl_dict = {}
      critic_dict = self.fit_critic(states, actions, next_states, next_actions,
                                    rewards, discounts)

    return {**ssl_dict, **critic_dict}

  def get_input_state_dim(self):
    return self.embedder.embedding_dim

  @tf.function
  def act(self, states, data_aug=False):
    if data_aug and self.num_augmentations > 0:
      states = states[0]
    if self.num_augmentations > 0:
      # use pad of 2 to bump 64 to 68 with 2 + 64 + 2 on each side
      img_pad = 2
      paddings = tf.constant(
          [[0, 0], [img_pad, img_pad], [img_pad, img_pad], [0, 0]],
          dtype=tf.int32)
      states = tf.cast(
          tf.pad(tf.cast(states * 255., tf.int32), paddings, 'SYMMETRIC'),
          tf.float32) / 255.

    q1, q2 = self.critic(states, stop_grad_features=True, actions=None)
    q = tf.minimum(q1, q2)
    actions = tf.argmax(q, -1)
    return actions


class DeepMdpLearner(SuperModelLearner):
  """A learner for DeepMDP."""

  def __init__(self,
               observation_spec,
               action_spec,
               embedding_dim = 256,
               num_distributions=None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate=None,
               num_augmentations=0,
               rep_learn_keywords = 'outer',
               batch_size = 256):
    super().__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        embedding_dim=embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        reward_weight=1.0,
        inverse_weight=0.0,
        forward_weight=1.0,
        state_prediction_mode='latent',
        num_augmentations=num_augmentations,
        rep_learn_keywords='outer',
        batch_size=batch_size)

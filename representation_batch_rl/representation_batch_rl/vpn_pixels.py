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

"""Value prediction network implementation.
"""
import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import tf_utils


class ValuePredictionNetworkLearner(tf.keras.Model):
  """A learner for model-based representation learning.

  Encompasses forward models, inverse models, as well as latent models like
  DeepMDP.
  """

  def __init__(self,
               observation_spec,
               action_spec,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate=None,
               discount = 0.95,
               target_update_period = 1000,
               num_augmentations=0,
               rep_learn_keywords = 'outer',
               batch_size = 256):
    """Creates networks.

    Args:
      observation_spec: State spec.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input.
      learning_rate: Learning rate.
      discount: discount factor.
      target_update_period: How frequently update target?
      num_augmentations: Number of DrQ random crops.
      rep_learn_keywords: Representation learning loss to add.
      batch_size: batch size.
    """
    super().__init__()
    action_dim = action_spec.maximum.item()+1
    self.observation_spec = observation_spec
    self.action_dim = action_dim
    self.action_spec = action_spec
    self.embedding_dim = embedding_dim
    self.sequence_length = sequence_length
    self.discount = discount
    self.tau = 0.005
    self.discount = 0.99
    self.target_update_period = target_update_period
    self.num_augmentations = num_augmentations
    self.rep_learn_keywords = rep_learn_keywords.split('__')
    self.batch_size = batch_size

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
        hidden_dims=hidden_dims)
    self.f_value = tf_utils.create_mlp(
        self.embedding_dim, 1, hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_value_target = tf_utils.create_mlp(
        self.embedding_dim, 1, hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_trans = tf_utils.create_mlp(
        self.embedding_dim + self.embedding_dim, self.embedding_dim,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_out = tf_utils.create_mlp(
        self.embedding_dim + self.embedding_dim, 2,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish)

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
    critic.soft_update(self.f_value, self.f_value_target, tau=1.0)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    self.all_variables = (
        self.embedder.trainable_variables + self.f_value.trainable_variables +
        self.f_value_target.trainable_variables +
        self.f_trans.trainable_variables + self.f_out.trainable_variables +
        self.critic.trainable_variables +
        self.critic_target.trainable_variables)

    self.model_dict = {
        'action_encoder': self.action_encoder,
        'f_out': self.f_out,
        'f_trans': self.f_trans,
        'f_value_target': self.f_value_target,
        'f_value': self.f_value,
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
      actions: batch of actions
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    features = self.critic.encoder(states)
    return self.embedder(features, stop_gradient=stop_gradient)

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between every pair of (embedding, other_embedding)."""
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
      next_actions: batch of next actions
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
          [batch_size, self.sequence_length, self.embedding_dim])[:, 0, :]

      all_pred_values = []
      all_pred_rewards = []
      all_pred_discounts = []
      for idx in range(self.sequence_length):
        pred_value = self.f_value(embeddings)[Ellipsis, 0]
        pred_reward, pred_discount = tf.unstack(
            self.f_out(tf.concat([embeddings, actions[:, idx, :]], -1)),
            axis=-1)
        pred_embeddings = embeddings + self.f_trans(
            tf.concat([embeddings, actions[:, idx, :]], -1))

        all_pred_values.append(pred_value)
        all_pred_rewards.append(pred_reward)
        all_pred_discounts.append(pred_discount)

        embeddings = pred_embeddings

      last_value = tf.stop_gradient(
          self.f_value_target(embeddings)[Ellipsis, 0]) / (1 - self.discount)
      all_true_values = []
      # for idx in range(self.sequence_length - 1, -1, -1):
      value = self.discount * discounts * last_value + rewards  #[:, idx]
      all_true_values.append(value)
      last_value = value
      all_true_values = all_true_values[::-1]

      reward_error = tf.stack(all_pred_rewards, -1)[:, 0] - rewards
      value_error = tf.stack(
          all_pred_values,
          -1) - (1 - self.discount) * tf.stack(all_true_values, -1)
      reward_loss = tf.reduce_sum(tf.math.square(reward_error), -1)
      value_loss = tf.reduce_sum(tf.math.square(value_error), -1)

      loss = tf.reduce_mean(reward_loss + value_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))
    if self.optimizer.iterations % self.target_update_period == 0:
      critic.soft_update(self.f_value, self.f_value_target, tau=self.tau)

    return {
        'embed_loss': loss,
        'reward_loss': tf.reduce_mean(reward_loss),
        'value_loss': tf.reduce_mean(value_loss),
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
    transition = next(replay_buffer_iter)
    numpy_dataset = isinstance(replay_buffer_iter, np.ndarray)
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


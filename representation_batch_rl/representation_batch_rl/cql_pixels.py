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

"""Implementation of DDPG."""

import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import tf_utils


class CQL(object):
  """Class performing CQL training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 1e-4,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_entropy = 0.0,
               reg = 0.0,
               num_cql_actions = 10,
               bc_pretraining_steps = 40_000,
               min_q_weight = 10.0,
               num_augmentations = 1,
               rep_learn_keywords = 'outer',
               batch_size = 256):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_entropy: Target entropy.
      reg: Coefficient for out of distribution regularization.
      num_cql_actions: Number of actions to sample for CQL loss.
      bc_pretraining_steps: Use BC loss instead of CQL loss for N steps.
      min_q_weight: CQL alpha.
      num_augmentations: Num of random crops
      rep_learn_keywords: Representation learning loss to add.
      batch_size: Batch size
    """
    self.num_augmentations = num_augmentations
    self.batch_size = batch_size
    self.rep_learn_keywords = rep_learn_keywords.split('__')

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
    # Combine and stop_grad some of the above conv stacks
    critic_kwargs['encoder'] = ImageEncoder(
        conv_stack_critic, feature_dim=state_dim, bprop_conv_stack=True)
    # Note: the target critic does not share any weights.
    critic_kwargs['encoder_target'] = ImageEncoder(
        conv_target_stack_critic, feature_dim=state_dim, bprop_conv_stack=True)

    if self.num_augmentations == 0:
      dummy_state = tf.constant(
          np.zeros(shape=[1] + list(observation_spec.shape)))
    else:  # account for padding of +4 everywhere and then cropping out 68
      dummy_state = tf.constant(np.zeros(shape=[1, 68, 68, 3]))

    @tf.function
    def init_models():
      critic_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder_target'](dummy_state)

    init_models()

    hidden_dims = (256, 256)
    # self.actor = policies.CategoricalPolicy(state_dim, action_spec,
    #               hidden_dims=hidden_dims, encoder=actor_kwargs['encoder'])
    action_dim = action_spec.maximum.item() + 1

    self.action_dim = action_dim

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.log_cql_alpha = self.log_alpha
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

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
      """This function initializes all auxiliary networks (state and action encoders) with dummy input (Procgen-specific, 68x68x3, 15 actions).
      """
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
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    self.tau = tau

    self.reg = reg
    self.target_entropy = target_entropy
    self.discount = discount

    self.num_cql_actions = num_cql_actions
    self.bc_pretraining_steps = bc_pretraining_steps
    self.min_q_weight = min_q_weight

    self.bc = None

    self.model_dict = {
        'critic': self.critic,
        'critic_target': self.critic_target,
        'critic_optimizer': self.critic_optimizer,
        'alpha_optimizer': self.alpha_optimizer
    }

  @property
  def alpha(self):
    return tf.constant(0.)

  @property
  def cql_alpha(self):
    return tf.exp(self.log_cql_alpha)

  def fit_critic(self, states, actions,
                 next_states, next_actions, rewards,
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

    if self.num_augmentations > 1:
      target_q = 0.
      for i in range(self.num_augmentations):
        next_q1_i, next_q2_i = self.critic_target(next_states[i], actions=None)
        target_q_i = tf.expand_dims(
            rewards, 1) + self.discount * tf.expand_dims(
                discounts, 1) * tf.minimum(next_q1_i, next_q2_i)
        target_q += target_q_i
      target_q /= self.num_augmentations
    elif self.num_augmentations == 1:
      next_q1, next_q2 = self.critic_target(
          next_states[0], actions=None, stop_grad_features=False)
      target_q = tf.expand_dims(
          rewards, 1) + self.discount * tf.expand_dims(
              discounts, 1) * tf.minimum(next_q1, next_q2)
    else:
      next_q1, next_q2 = self.critic_target(next_states, actions=None)
      target_q = tf.expand_dims(rewards, 1) + self.discount * tf.expand_dims(
          discounts, 1) * tf.minimum(next_q1, next_q2)

    target_q = tf.gather_nd(target_q, indices=next_action_indices)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables)

      if self.num_augmentations > 1:
        critic_loss = 0.
        q1 = 0.
        q2 = 0.
        for i in range(self.num_augmentations):
          q1_i, q2_i = self.critic(states[i], actions=None)
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
      elif self.num_augmentations == 1:
        q1, q2 = self.critic(states[0], actions=None)
        critic_loss = (
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q1, indices=action_indices)) +
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q2, indices=action_indices)))
      else:
        # Ensure num_augmentations is non-negative
        assert self.num_augmentations == 0
        q1, q2 = self.critic(states, actions=None)
        critic_loss = (
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q1, indices=action_indices)) +
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q2, indices=action_indices)))
      q = tf.minimum(q1, q2)
      cql_logsumexp = tf.reduce_logsumexp(q, 1)
      cql_loss = tf.reduce_mean(cql_logsumexp -
                                tf.gather_nd(q, indices=action_indices))

      critic_loss += (self.reg * cql_loss)

    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.trainable_variables))

    critic.soft_update(self.critic, self.critic_target, tau=self.tau)

    return {
        'q1': tf.reduce_mean(q1),
        'q2': tf.reduce_mean(q2),
        'critic_loss': critic_loss,
        'cql_loss': cql_loss
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
    del train_target
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

    critic_dict = self.fit_critic(states, actions, next_states, next_actions,
                                  rewards, discounts)

    return critic_dict

  @tf.function
  def act(self, states, data_aug=False):
    """Act with batch of states.

    Args:
      states: tf.tensor n_batch x 64 x 64 x 3
      data_aug: bool, whether to use stochastic data aug (else deterministic)

    Returns:
      action: tf.tensor
    """
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

    q1, q2 = self.critic(states, actions=None)
    q = tf.minimum(q1, q2)
    actions = tf.argmax(q, -1)
    return actions

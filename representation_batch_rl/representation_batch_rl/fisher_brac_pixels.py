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
"""Implementation of Fisher-BRAC from pixels."""

import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import behavioral_cloning_pixels as behavioral_cloning
from representation_batch_rl.representation_batch_rl import policies_pixels as policies
from representation_batch_rl.representation_batch_rl import tf_utils


class FBRAC(object):
  """Class performing BRAC training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               alpha_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_entropy = 0.0,
               f_reg = 1.0,
               reward_bonus = 5.0,
               num_augmentations = 1,
               env_name = '',
               batch_size = 256):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      alpha_lr: Temperature learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_entropy: Target entropy.
      f_reg: Critic regularization weight.
      reward_bonus: Bonus added to the rewards.
      num_augmentations: Number of DrQ augmentations (crops)
      env_name: Env name
      batch_size: Batch size
    """
    self.num_augmentations = num_augmentations
    self.discrete_actions = False if len(action_spec.shape) else True
    self.batch_size = batch_size

    actor_kwargs = {'hidden_dims': (1024, 1024)}
    critic_kwargs = {'hidden_dims': (1024, 1024)}

    # DRQ encoder params.
    # https://github.com/denisyarats/drq/blob/master/config.yaml#L73

    # Make 4 sets of weights:
    # - BC
    # - Actor
    # - Critic
    # - Critic (target)

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

    conv_stack_bc = conv_stack()
    conv_stack_actor = conv_stack()
    conv_stack_critic = conv_stack()
    conv_target_stack_critic = conv_stack()

    if observation_spec.shape == (64, 64, 3):
      conv_stack_bc.output_size = state_dim
      conv_stack_actor.output_size = state_dim
      conv_stack_critic.output_size = state_dim
      conv_target_stack_critic.output_size = state_dim
    # Combine and stop_grad some of the above conv stacks
    actor_kwargs['encoder_bc'] = ImageEncoder(
        conv_stack_bc, feature_dim=state_dim, bprop_conv_stack=True)
    actor_kwargs['encoder'] = ImageEncoder(
        conv_stack_critic, feature_dim=state_dim, bprop_conv_stack=False)
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
      actor_kwargs['encoder_bc'](dummy_state)
      actor_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder_target'](dummy_state)

    init_models()

    if self.discrete_actions:
      hidden_dims = ()
      self.actor = policies.CategoricalPolicy(
          state_dim,
          action_spec,
          hidden_dims=hidden_dims,
          encoder=actor_kwargs['encoder'])
      action_dim = action_spec.maximum.item() + 1
    else:
      hidden_dims = (256, 256, 256)
      self.actor = policies.DiagGuassianPolicy(
          state_dim,
          action_spec,
          hidden_dims=hidden_dims,
          encoder=actor_kwargs['encoder'])
      action_dim = action_spec.shape[0]

    self.action_dim = action_dim

    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

    self.target_entropy = target_entropy
    self.discount = discount
    self.tau = tau

    self.bc = behavioral_cloning.BehavioralCloning(
        observation_spec,
        action_spec,
        mixture=True,
        encoder=actor_kwargs['encoder_bc'],
        num_augmentations=self.num_augmentations,
        env_name=env_name,
        batch_size=batch_size)

    self.critic = critic.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder'])
    self.critic_target = critic.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder_target'])

    critic.soft_update(self.critic, self.critic_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    self.f_reg = f_reg
    self.reward_bonus = reward_bonus

    self.model_dict = {
        'critic': self.critic,
        'critic_target': self.critic_target,
        'actor': self.actor,
        'bc': self.bc,
        'critic_optimizer': self.critic_optimizer,
        'alpha_optimizer': self.alpha_optimizer,
        'actor_optimizer': self.actor_optimizer
    }

  def dist_critic(self, states, actions, target=False, stop_gradient=False):
    """Distribution critic (via offset).

    Args:
      states: batch of states
      actions: batch of actions
      target: whether to use target for q1,q2
      stop_gradient: whether to stop_grad log-probs
    Returns:
      dist
    """
    if target:
      q1, q2 = self.critic_target(states, actions)
    else:
      q1, q2 = self.critic(states, actions)
    if self.discrete_actions:
      # expects (n_batch,) tensor instead of (n_batch x n_actions)
      actions = tf.argmax(actions, 1)

    log_probs = self.bc.policy.log_probs(states, actions)
    if stop_gradient:
      log_probs = tf.stop_gradient(log_probs)
    return (q1 + log_probs, q2 + log_probs)

  def fit_critic(self, states, actions,
                 next_states, rewards,
                 discounts):
    """Updates critic parameters.

    Args:
      states: Batch of states.
      actions: Batch of actions.
      next_states: Batch of next states.
      rewards: Batch of rewards.
      discounts: Batch of masks indicating the end of the episodes.

    Returns:
      Dictionary with information to track.
    """

    if self.num_augmentations > 0:
      next_actions = self.actor(next_states[0], sample=True)
      policy_actions = self.actor(states[0], sample=True)
      target_q = 0.
      for i in range(self.num_augmentations):
        next_target_q1_i, next_target_q2_i = self.dist_critic(
            next_states[i], next_actions, target=True)
        target_q_i = rewards + self.discount * discounts * tf.minimum(
            next_target_q1_i, next_target_q2_i)
        target_q += target_q_i
      target_q /= self.num_augmentations
    else:
      next_actions = self.actor(next_states, sample=True)
      policy_actions = self.actor(states, sample=True)
      if self.discrete_actions:
        next_actions = tf.cast(
            tf.one_hot(next_actions, depth=self.action_dim), tf.float32)
        policy_actions = tf.cast(
            tf.one_hot(policy_actions, depth=self.action_dim), tf.float32)

      next_target_q1, next_target_q2 = self.dist_critic(
          next_states, next_actions, target=True)
      target_q = rewards + self.discount * discounts * tf.minimum(
          next_target_q1, next_target_q2)

    critic_variables = self.critic.trainable_variables

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(critic_variables)
      if self.num_augmentations > 0:
        critic_loss = 0.
        for i in range(self.num_augmentations):
          q1, q2 = self.dist_critic(states[i], actions, stop_gradient=True)
          with tf.GradientTape(
              watch_accessed_variables=False, persistent=True) as tape2:
            tape2.watch([policy_actions])

            q1_reg, q2_reg = self.critic(states[i], policy_actions)

          q1_grads = tape2.gradient(q1_reg, policy_actions)
          q2_grads = tape2.gradient(q2_reg, policy_actions)

          q1_grad_norm = tf.reduce_sum(tf.square(q1_grads), axis=-1)
          q2_grad_norm = tf.reduce_sum(tf.square(q2_grads), axis=-1)

          del tape2

          q_reg = tf.reduce_mean(q1_grad_norm + q2_grad_norm)

          critic_loss_i = (
              tf.losses.mean_squared_error(target_q, q1) +
              tf.losses.mean_squared_error(target_q, q2) + self.f_reg * q_reg)
          critic_loss += critic_loss_i
        critic_loss /= self.num_augmentations
      else:
        q1, q2 = self.dist_critic(states, actions, stop_gradient=True)
        with tf.GradientTape(
            watch_accessed_variables=False, persistent=True) as tape2:
          tape2.watch([policy_actions])

          q1_reg, q2_reg = self.critic(states, policy_actions)

        q1_grads = tape2.gradient(q1_reg, policy_actions)
        q2_grads = tape2.gradient(q2_reg, policy_actions)

        q1_grad_norm = tf.reduce_sum(tf.square(q1_grads), axis=-1)
        q2_grad_norm = tf.reduce_sum(tf.square(q2_grads), axis=-1)

        del tape2

        q_reg = tf.reduce_mean(q1_grad_norm + q2_grad_norm)

        critic_loss = (tf.losses.mean_squared_error(target_q, q1) +
                       tf.losses.mean_squared_error(target_q, q2) +
                       self.f_reg * q_reg)

    critic_grads = tape.gradient(critic_loss, critic_variables)

    self.critic_optimizer.apply_gradients(zip(critic_grads, critic_variables))

    critic.soft_update(self.critic, self.critic_target, tau=self.tau)

    return {
        'q1': tf.reduce_mean(q1),
        'q2': tf.reduce_mean(q2),
        'critic_loss': critic_loss,
        'q1_grad': tf.reduce_mean(q1_grad_norm),
        'q2_grad': tf.reduce_mean(q2_grad_norm)
    }

  @property
  def alpha(self):
    return tf.constant(0.)
    # return tf.exp(self.log_alpha)

  def fit_actor(self, states):
    """Updates critic parameters.

    Args:
      states: A batch of states.

    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
      if self.discrete_actions:
        actions = tf.cast(
            tf.one_hot(actions, depth=self.action_dim), tf.float32)
      q1, q2 = self.dist_critic(states, actions)
      q = tf.minimum(q1, q2)
      actor_loss = tf.reduce_mean(self.alpha * log_probs - q)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

#     with tf.GradientTape(watch_accessed_variables=False) as tape:
#       tape.watch([self.log_alpha])
#       alpha_loss = tf.reduce_mean(self.alpha *
#                                   (-log_probs - self.target_entropy))

#     alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
#     self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
    alpha_loss = tf.consant(0.)

    return {
        'actor_loss': actor_loss,
        'alpha': self.alpha,
        'alpha_loss': alpha_loss
    }

  @tf.function
  def update_step(self, replay_buffer_iter,
                  numpy_dataset):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: A tensorflow graph iteratable object.
      numpy_dataset: Is the dataset a NumPy array?

    Returns:
      Dictionary with losses to track.
    """

    transition = next(replay_buffer_iter)

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

    # states, actions, rewards, discounts, next_states = next(replay_buffer_iter
    rewards = rewards + self.reward_bonus

    if self.discrete_actions:
      actions = tf.cast(tf.one_hot(actions, depth=self.action_dim), tf.float32)

    critic_dict = self.fit_critic(states, actions, next_states, rewards,
                                  discounts)

    actor_dict = self.fit_actor(
        states[0] if self.num_augmentations > 0 else states)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

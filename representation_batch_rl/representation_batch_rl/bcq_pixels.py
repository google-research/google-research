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

# python3
"""Implementation of BCQ from pixels."""

import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import policies_pixels as policies
from representation_batch_rl.representation_batch_rl import tf_utils


class BCQ(object):
  """Class performing BCQ training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               num_augmentations = 1):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      num_augmentations: Number of DrQ-style augmentations to perform on pixels
    """

    self.num_augmentations = num_augmentations
    self.discrete_actions = False if len(action_spec.shape) else True

    actor_kwargs = {}
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

    conv_stack_actor = conv_stack()
    conv_stack_critic = conv_stack()
    conv_target_stack_critic = conv_stack()

    if observation_spec.shape == (64, 64, 3):
      conv_stack_actor.output_size = state_dim
      conv_stack_critic.output_size = state_dim
      conv_target_stack_critic.output_size = state_dim
    # Combine and stop_grad some of the above conv stacks
    actor_kwargs['encoder'] = ImageEncoder(
        conv_stack_actor, feature_dim=state_dim, bprop_conv_stack=True)
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
      actor_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder_target'](dummy_state)

    init_models()

    if self.discrete_actions:
      action_dim = action_spec.maximum.item() + 1
      self.actor = policies.CVAEPolicyPixelsDiscrete(
          state_dim,
          action_spec,
          action_dim * 2,
          encoder=actor_kwargs['encoder'])

    else:
      action_dim = action_spec.shape[0]
      self.actor = policies.CVAEPolicyPixels(
          state_dim,
          action_spec,
          action_dim * 2,
          encoder=actor_kwargs['encoder'])

    self.action_dim = action_dim
    self.state_dim = state_dim

    if self.discrete_actions:
      self.action_encoder = tf.keras.Sequential(
          [
              tf.keras.layers.Dense(
                  state_dim, use_bias=True
              ),  # , kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)
              tf.keras.layers.ReLU(),
              # tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(action_dim)
          ],
          name='action_encoder')
      dummy_psi_act = tf.constant(np.zeros(shape=[1, state_dim]))
      self.action_encoder(dummy_psi_act)
    else:
      self.action_encoder = None

    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.critic_learner = critic.CriticLearner(
        state_dim,
        action_dim,
        critic_lr,
        discount,
        tau,
        encoder=critic_kwargs['encoder'],
        encoder_target=critic_kwargs['encoder_target'])

    self.bc = None
    self.threshold = 0.3

    self.model_dict = {
        'critic_learner': self.critic_learner,
        'action_encoder': self.action_encoder,
        'actor': self.actor,
        'actor_optimizer': self.actor_optimizer
    }

  @tf.function
  def select_actions(self, states, num_candidates = 10):
    """Samples argmax actions for a batch of states.

    Args:
      states: Batch of states.
      num_candidates: Number of candidate actions to sample.

    Returns:
      Batch of actions.
    """
    if self.discrete_actions:
      # tf.shape(states)[0]
      q1, q2 = self.critic_learner.critic(
          tf.repeat(states, self.action_dim, axis=0),
          tf.cast(
              tf.concat([tf.eye(self.action_dim, dtype=tf.float32)] *
                        states.shape[0], 0), tf.float32),
          return_features=False)
      q = tf.minimum(q1, q2)
      q = tf.reshape(q, (-1, self.action_dim))
      features = self.critic_learner.critic.encoder(states)
      act_logits = self.action_encoder(features)
      act_log_softmax = tf.nn.log_softmax(act_logits, axis=1)
      act_log_softmax = tf.math.exp(act_log_softmax)
      act_log_softmax = tf.cast(
          tf.math.greater(
              act_log_softmax /
              tf.math.reduce_max(act_log_softmax, axis=1, keepdims=True),
              self.threshold), tf.float32)

      # Use large negative number to mask actions from argmax
      next_action = tf.argmax(
          act_log_softmax * q + (1 - act_log_softmax) * -1e8, axis=1)
      return next_action
    else:
      states = tf.repeat(states, num_candidates, axis=0)
      actions = self.actor(states)
      q1, q2 = self.critic_learner.critic(states, actions)
      q = tf.minimum(q1, q2)
      q = tf.reshape(q, [-1, num_candidates])
      max_inds = tf.math.argmax(q, -1)

      indices = tf.stack([
          tf.range(0, tf.shape(max_inds)[0], dtype=tf.int64), max_inds], 1)
      actions = tf.reshape(actions, [-1, num_candidates, actions.shape[-1]])
      return tf.gather_nd(actions, indices)

  @tf.function
  def fit_actor(self, states,
                actions):
    """Updates actor parameters.

    Args:
      states: Batch of states.
      actions: Batch of states.

    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      mean, logvar = self.actor.encode(states, actions)
      z = self.actor.reparameterize(mean, logvar)
      recon = self.actor.decode(states, z)
      kl_loss = -0.5 * tf.reduce_sum(1.0 + logvar - tf.pow(mean, 2) -
                                     tf.exp(logvar), -1)
      mse_loss = tf.reduce_sum(tf.square(recon - actions), -1)

      kl_loss = tf.reduce_mean(kl_loss)
      mse_loss = tf.reduce_mean(mse_loss)
      actor_loss = kl_loss + mse_loss

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    return {'actor_loss': actor_loss, 'kl_loss': kl_loss, 'mse_loss': mse_loss}

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
          num_augmentations=self.num_augmentations)

    if not self.discrete_actions:
      actor_dict = self.fit_actor(
          states[0] if self.num_augmentations > 0 else states, actions)

    next_actions = self.select_actions(
        next_states[0] if self.num_augmentations > 0 else next_states)

    if self.discrete_actions:
      actions = tf.cast(tf.one_hot(actions, depth=self.action_dim), tf.float32)
      next_actions = tf.cast(
          tf.one_hot(next_actions, depth=self.action_dim), tf.float32)

    critic_dict = self.critic_learner.fit_critic(
        states[0] if self.num_augmentations > 0 else states, actions,
        next_states[0] if self.num_augmentations > 0 else next_states,
        next_actions, rewards, discounts)

    if self.discrete_actions:
      return critic_dict
    else:
      return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.select_actions(states, num_candidates=100)

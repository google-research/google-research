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
"""Implementation of DDPG."""

import typing
from dm_env import specs as dm_env_specs
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl import policies
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder


class SAC(object):
  """Class performing SAC training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               alpha_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_update_period = 1,
               target_entropy = 0.0,
               cross_norm = False,
               pcl_actor_update = False):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      alpha_lr: Temperature learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_update_period: Target network update period.
      target_entropy: Target entropy.
      cross_norm: Whether to fit cross norm critic.
      pcl_actor_update: Whether to use PCL actor update.
    """
    actor_kwargs = {}
    critic_kwargs = {}

    if len(observation_spec.shape) == 3:  # Image observations.
      # DRQ encoder params.
      # https://github.com/denisyarats/drq/blob/master/config.yaml#L73
      state_dim = 50

      # Actor and critic encoders share conv weights only.
      conv_stack = ConvStack(observation_spec.shape)

      actor_kwargs['encoder'] = ImageEncoder(
          conv_stack, state_dim, bprop_conv_stack=False)
      actor_kwargs['hidden_dims'] = (1024, 1024)

      critic_kwargs['encoder'] = ImageEncoder(
          conv_stack, state_dim, bprop_conv_stack=True)
      critic_kwargs['hidden_dims'] = (1024, 1024)

      if not cross_norm:
        # Note: the target critic does not share any weights.
        critic_kwargs['encoder_target'] = ImageEncoder(
            ConvStack(observation_spec.shape), state_dim, bprop_conv_stack=True)

    else:  # 1D state observations.
      assert len(observation_spec.shape) == 1
      state_dim = observation_spec.shape[0]

    if cross_norm:
      beta_1 = 0.0
    else:
      beta_1 = 0.9

    self.actor = policies.DiagGuassianPolicy(state_dim, action_spec,
                                             **actor_kwargs)
    self.actor_optimizer = tf.keras.optimizers.Adam(
        learning_rate=actor_lr, beta_1=beta_1)

    self.log_alpha = tf.Variable(tf.math.log(0.1), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha_lr, beta_1=beta_1)

    if cross_norm:
      assert 'encoder_target' not in critic_kwargs
      self.critic_learner = critic.CrossNormCriticLearner(
          state_dim, action_spec.shape[0], critic_lr, discount, tau,
          **critic_kwargs)
    else:
      self.critic_learner = critic.CriticLearner(
          state_dim, action_spec.shape[0], critic_lr, discount, tau,
          target_update_period, **critic_kwargs)

    self.target_entropy = target_entropy
    self.discount = discount

    self.pcl_actor_update = pcl_actor_update

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

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
      q1, q2 = self.critic_learner.critic(states, actions)
      q = tf.minimum(q1, q2)
      actor_loss = tf.reduce_mean(self.alpha * log_probs - q)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.alpha *
                                  (-log_probs - self.target_entropy))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {
        'actor_loss': actor_loss,
        'alpha': self.alpha,
        'alpha_loss': alpha_loss
    }

  def fit_actor_mix(self, states,
                    data_actions):
    """Updates critic parameters.

    Args:
      states: Batch of states.
      data_actions: Batch of actions from replay buffer.

    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      out = self.actor.trunk(states)

      with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(out)
        actions, log_probs = self.actor(
            states, out=out, sample=True, with_log_probs=True)
        q1, q2 = self.critic_learner.critic(states, actions)
        q = tf.minimum(q1, q2)
      out_q_grad = tape2.gradient(q, [out])[0]

      with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(out)
        data_log_probs = self.actor.log_probs(states, data_actions, out=out)
      out_data_grad = tape2.gradient(data_log_probs, [out])[0]

      q_grad_norm = tf.reduce_mean(tf.norm(out_q_grad, axis=-1))
      data_grad_norm = tf.reduce_mean(tf.norm(out_data_grad, axis=-1))

      out_data_grad *= q_grad_norm / (data_grad_norm + 1e-3)

      q1, q2 = self.critic_learner.critic(states, data_actions)
      q_data = tf.minimum(q1, q2)
      adv = tf.cast(q_data > q, tf.float32)[:, tf.newaxis]

      q_grad = tf.reduce_sum(
          out * tf.stop_gradient(out_q_grad *
                                 (1.0 - adv) + out_data_grad * adv), -1)

      actor_loss = tf.reduce_mean(self.alpha * log_probs - q_grad)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.alpha *
                                  (-log_probs - self.target_entropy))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {
        'actor_loss': actor_loss,
        'alpha': self.alpha,
        'alpha_loss': alpha_loss
    }

  def get_v(self, states, n_samples=4):
    states = tf.repeat(states, n_samples, axis=0)
    actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
    q1, q2 = self.critic_learner.critic(states, actions)
    q = tf.minimum(q1, q2) / self.alpha
    q = q - log_probs - tf.math.log(tf.cast(n_samples, tf.float32))
    q = tf.reshape(q, [-1, n_samples])
    return self.alpha * tf.math.reduce_logsumexp(q, -1)

  def fit_actor_pcl(self, states,
                    actions):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      Actor loss.
    """
    q1, q2 = self.critic_learner.critic(states, actions)
    q = tf.minimum(q1, q2)

    v = self.get_v(states)
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self.actor.trainable_variables)

      actor_log_probs = self.actor.log_probs(states, actions)

      adv = tf.stop_gradient(q - v - self.alpha * actor_log_probs)

      weights = tf.cast(actor_log_probs > -100, dtype=tf.float32)
      weights = tf.stop_gradient(weights)
      actor_loss = -tf.reduce_mean(actor_log_probs * adv * weights)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    _, log_probs = self.actor(states, sample=True, with_log_probs=True)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.alpha *
                                  (-log_probs - self.target_entropy / 2.0))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {
        'actor_loss': actor_loss,
        'alpha': self.alpha,
        'alpha_loss': alpha_loss,
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.

    Returns:
      Dictionary with losses to track.
    """

    states, actions, rewards, discounts, next_states = next(replay_buffer_iter)

    next_actions, next_log_probs = self.actor(
        next_states, sample=True, with_log_probs=True)

    entropy_rewards = self.discount * discounts * self.alpha * next_log_probs
    rewards -= entropy_rewards
    critic_dict = self.critic_learner.fit_critic(states, actions, next_states,
                                                 next_actions, rewards,
                                                 discounts)

    if self.pcl_actor_update:
      actor_dict = self.fit_actor_pcl(states, actions)
    else:
      actor_dict = self.fit_actor(states)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

  def save_weights(self, path):
    self.actor.save_weights(path+'__actor')
    self.critic_learner.critic.save_weights(path+'__critic')

  def load_weights(self, path):
    self.actor.load_weights(path+'__actor')
    self.critic_learner.critic.load_weights(path+'__critic')

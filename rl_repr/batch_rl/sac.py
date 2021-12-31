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

"""Soft actor-critic module."""

import typing
from dm_env import specs as dm_env_specs
import tensorflow as tf

from rl_repr.batch_rl import critic
from rl_repr.batch_rl import policies


class SAC(object):
  """Class performing SAC training."""

  def __init__(self,
               state_dim,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               alpha_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_update_period = 1,
               target_entropy = 0.0,
               cross_norm = False,
               pcl_actor_update = False,
               embed_model=None,
               other_embed_model=None,
               network='default',
               finetune = False):
    """Creates networks.

    Args:
      state_dim: State size.
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
      embed_model: Pretrained embedder.
      other_embed_model: Pretrained embedder. Used for critic if specified.
      network: Type of actor/critic net.
      finetune: Whether to finetune the pretrained embedder.
    """
    self.action_spec = action_spec
    self.embed_model = embed_model
    self.other_embed_model = other_embed_model or embed_model

    self.finetune = finetune

    input_state_dim = (
        self.embed_model.get_input_state_dim()
        if self.embed_model else state_dim)

    if cross_norm:
      beta_1 = 0.0
    else:
      beta_1 = 0.9

    hidden_dims = ([] if network == 'none' else
                   (256,) if network == 'small' else
                   (256, 256))
    self.actor = policies.DiagGuassianPolicy(
        input_state_dim, action_spec, hidden_dims=hidden_dims)
    self.actor_optimizer = tf.keras.optimizers.Adam(
        learning_rate=actor_lr, beta_1=beta_1)

    self.log_alpha = tf.Variable(tf.math.log(0.1), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha_lr, beta_1=beta_1)

    if cross_norm:
      assert network == 'default'
      self.critic_learner = critic.CrossNormCriticLearner(
          input_state_dim, action_spec.shape[0], critic_lr, discount, tau)
    else:
      self.critic_learner = critic.CriticLearner(
          input_state_dim,
          action_spec.shape[0],
          critic_lr,
          discount,
          tau,
          target_update_period,
          hidden_dims=hidden_dims)

    self.target_entropy = target_entropy
    self.discount = discount

    self.pcl_actor_update = pcl_actor_update

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  def fit_actor(self, get_actor_inputs_fn, get_critic_inputs_fn, extra_vars):
    """Updates critic parameters.

    Args:
      get_actor_inputs_fn: Actor input function.
      get_critic_inputs_fn: Critic input function.
      extra_vars: Extra variable from embedder.

    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables + extra_vars)
      states = get_actor_inputs_fn()
      critic_states, _, _, _, _, _ = get_critic_inputs_fn()
      actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
      q1, q2 = self.critic_learner.critic(critic_states, actions)
      q = tf.minimum(q1, q2)
      actor_loss = tf.reduce_mean(self.alpha * log_probs - q)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables + extra_vars)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables + extra_vars))

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
    q = tf.minimum(q1, q2)
    q = q - self.alpha * log_probs
    q = tf.reshape(q, [-1, n_samples])
    return tf.reduce_mean(q, -1)

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

      adv = tf.stop_gradient(q - self.alpha * actor_log_probs - v)

      weights = tf.cast(actor_log_probs > -100, dtype=tf.float32)
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

    def get_actor_inputs_fn(states=states, actions=actions, rewards=rewards):
      if self.embed_model:
        states = self.embed_model(
            states, actions, rewards, stop_gradient=not self.finetune)
      return states

    def get_critic_inputs_fn(states=states, actions=actions, rewards=rewards,
                             discounts=discounts, next_states=next_states):
      if self.other_embed_model:
        nn_actions = actions[:, 1:, :] if len(actions.shape) == 3 else actions
        nn_rewards = rewards[:, 1:] if len(actions.shape) == 3 else rewards
        actor_next_states = self.embed_model(next_states, nn_actions,
                                             nn_rewards)
        states = self.other_embed_model(
            states, actions, rewards, stop_gradient=not self.finetune)
        next_states = self.other_embed_model(
            next_states,
            nn_actions,
            nn_rewards,
            stop_gradient=not self.finetune)
        if hasattr(self.other_embed_model,
                   'ctx_length') and self.other_embed_model.ctx_length:
          assert (len(actions.shape) == 3)
          actions = actions[:, self.other_embed_model.ctx_length - 1, :]
          rewards = rewards[:, self.other_embed_model.ctx_length - 1]
          discounts = discounts[:, self.other_embed_model.ctx_length - 1]
      else:
        actor_next_states = next_states

      next_actions, next_log_probs = self.actor(
          actor_next_states, sample=True, with_log_probs=True)
      next_actions = tf.stop_gradient(next_actions)
      next_log_probs = tf.stop_gradient(next_log_probs)

      entropy_rewards = self.discount * discounts * self.alpha * next_log_probs
      rewards -= entropy_rewards

      return states, actions, next_states, next_actions, rewards, discounts

    actor_extra_vars = self.embed_model.trainable_variables if self.embed_model and self.finetune else []
    critic_extra_vars = self.other_embed_model.trainable_variables if self.other_embed_model and self.finetune else []

    critic_dict = self.critic_learner.fit_critic(get_critic_inputs_fn, critic_extra_vars)

    if self.pcl_actor_update:
      assert False, 'not implemented'
    else:
      actor_dict = self.fit_actor(get_actor_inputs_fn, get_critic_inputs_fn,
                                  actor_extra_vars)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states, actions=None, rewards=None):
    if self.embed_model:
      states = self.embed_model(states, actions, rewards)
    return self.actor(states, sample=False)

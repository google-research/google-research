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

"""Behavior-regularized actor-critic module."""

import typing
from dm_env import specs as dm_env_specs
import tensorflow as tf

from rl_repr.batch_rl import behavioral_cloning
from rl_repr.batch_rl import critic
from rl_repr.batch_rl import policies


class BRAC(object):
  """Class performing BRAC training."""

  def __init__(self,
               state_dim,
               action_spec,
               actor_lr = 3e-5,
               critic_lr = 3e-4,
               alpha_lr = 1e-4,
               discount = 0.99,
               tau = 0.005,
               target_entropy = 0.0,
               bc_alpha = 1.0,
               embed_model=None,
               other_embed_model=None,
               bc_embed_model=None,
               network='default',
               finetune=False):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      alpha_lr: Temperature learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_entropy: Target entropy.
      bc_alpha: Policy regularization weight.
      embed_model: Pretrained embedder.
      other_embed_model: Pretrained embedder. Used for critic if specified.
      bc_embed_model: Pretrained embedder. Used for behavior
        cloning if specified.
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
    hidden_dims = ([] if network == 'none' else
                   (256,) if network == 'small' else
                   (256, 256))
    self.actor = policies.DiagGuassianPolicy(
        input_state_dim, action_spec, hidden_dims=hidden_dims)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.log_alpha = tf.Variable(tf.math.log(0.1), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

    self.target_entropy = target_entropy
    self.discount = discount
    self.tau = tau

    self.bc = behavioral_cloning.BehavioralCloning(
        state_dim,
        action_spec,
        mixture=True,
        hidden_dims=hidden_dims,
        embed_model=bc_embed_model or self.embed_model,
        finetune=self.finetune)

    self.bc_alpha = bc_alpha

    action_dim = action_spec.shape[0]
    self.critic = critic.Critic(
        input_state_dim, action_dim, hidden_dims=hidden_dims)
    self.critic_target = critic.Critic(
        input_state_dim, action_dim, hidden_dims=hidden_dims)
    critic.soft_update(self.critic, self.critic_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  def fit_critic(self, get_critic_inputs_fn,
                 extra_vars):
    """Updates critic parameters.

    Args:
      get_critic_inputs_fn: Critic input function.
      extra_vars: Extra variable from embedder.

    Returns:
      Dictionary with information to track.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables + extra_vars)
      (states, actions, next_states, next_actions, rewards,
       discounts) = get_critic_inputs_fn()

      next_target_q1, next_target_q2 = self.critic_target(
          next_states, next_actions)
      target_q = rewards + self.discount * discounts * tf.minimum(
          next_target_q1, next_target_q2)
      target_q = tf.stop_gradient(target_q)

      q1, q2 = self.critic(states, actions)

      critic_loss = tf.losses.mean_squared_error(target_q, q1) + \
                    tf.losses.mean_squared_error(target_q, q2)
    critic_grads = tape.gradient(critic_loss,
                                 self.critic.trainable_variables + extra_vars)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.trainable_variables + extra_vars))

    critic.soft_update(self.critic, self.critic_target, tau=self.tau)

    return {'q1': tf.reduce_mean(q1), 'q2': tf.reduce_mean(q2),
            'critic_loss': critic_loss}

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  def fit_actor(self, get_actor_inputs_fn, get_critic_inputs_fn,
                get_bc_inputs_fn, extra_vars):
    """Updates critic parameters.

    Args:
      get_actor_inputs_fn: Actor input function.
      get_critic_inputs_fn: Critic input function.
      get_bc_inputs_fn: Behavior cloning input function.
      extra_vars: Extra variable from embedder.

    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables + extra_vars)
      states, actions = get_actor_inputs_fn()
      bc_states = get_bc_inputs_fn()
      critic_states, _, _, _, _, _ = get_critic_inputs_fn()
      actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
      q1, q2 = self.critic(critic_states, actions)
      q = tf.minimum(q1, q2)
      bc_log_probs = self.bc.policy.log_probs(bc_states, actions)
      actor_loss = tf.reduce_mean(self.alpha * log_probs -
                                  self.bc_alpha * bc_log_probs - q)

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

  @tf.function
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.

    Returns:
      Dictionary with losses to track.
    """

    states, actions, rewards, discounts, next_states = next(
        replay_buffer_iter)

    def get_actor_inputs_fn(states=states, actions=actions, rewards=rewards):
      if self.embed_model:
        states = self.embed_model(
            states, actions, rewards, stop_gradient=not self.finetune)
      if hasattr(self.embed_model,
                 'ctx_length') and self.embed_model.ctx_length:
        assert (len(actions.shape) == 3)
        actions = actions[:, self.embed_model.ctx_length - 1, :]
      return states, actions

    def get_critic_inputs_fn(states=states, actions=actions, rewards=rewards,
                             discounts=discounts, next_states=next_states):
      if self.other_embed_model:
        nn_actions = actions[:, 1:, :] if len(actions.shape) == 3 else actions
        nn_rewards = rewards[:, 1:] if len(actions.shape) == 3 else rewards
        actor_states = self.embed_model(states, actions, rewards)
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
        actor_states = states
        actor_next_states = next_states

      next_actions = tf.stop_gradient(self.actor(actor_next_states, sample=True))

      return states, actions, next_states, next_actions, rewards, discounts

    def get_bc_inputs_fn(states=states, actions=actions, rewards=rewards):
      if self.bc.embed_model:
        states = self.bc.embed_model(
            states, actions, rewards, stop_gradient=not self.finetune)
      if hasattr(self.bc.embed_model,
                 'ctx_length') and self.bc.embed_model.ctx_length:
        assert (len(actions.shape) == 3)
        actions = actions[:, self.bc.embed_model.ctx_length - 1, :]
      return states

    actor_extra_vars = self.embed_model.trainable_variables if self.embed_model and self.finetune else []
    critic_extra_vars = self.other_embed_model.trainable_variables if self.other_embed_model and self.finetune else []

    critic_dict = self.fit_critic(get_critic_inputs_fn, critic_extra_vars)

    actor_dict = self.fit_actor(get_actor_inputs_fn, get_critic_inputs_fn, get_bc_inputs_fn,
                                actor_extra_vars)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states, actions=None, rewards=None):
    if self.embed_model:
      states = self.embed_model(states, actions, rewards)
    return self.actor(states, sample=False)

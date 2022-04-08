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
import tensorflow as tf
from tf_agents.specs.tensor_spec import BoundedTensorSpec
from tf_agents.specs.tensor_spec import TensorSpec

from fisher_brc import behavioral_cloning
from fisher_brc import critic
from fisher_brc import policies


class FBRC(object):
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
               reward_bonus = 5.0):
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
    """
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]

    hidden_dims = (256, 256, 256)
    self.actor = policies.DiagGuassianPolicy(state_dim, action_spec,
                                             hidden_dims=hidden_dims)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

    self.target_entropy = target_entropy
    self.discount = discount
    self.tau = tau

    self.bc = behavioral_cloning.BehavioralCloning(
        observation_spec, action_spec, mixture=True)

    action_dim = action_spec.shape[0]
    self.critic = critic.Critic(state_dim, action_dim, hidden_dims=hidden_dims)
    self.critic_target = critic.Critic(state_dim, action_dim,
                                       hidden_dims=hidden_dims)
    critic.soft_update(self.critic, self.critic_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    self.f_reg = f_reg
    self.reward_bonus = reward_bonus

  def dist_critic(self, states, actions, target=False, stop_gradient=False):
    if target:
      q1, q2 = self.critic_target(states, actions)
    else:
      q1, q2 = self.critic(states, actions)
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
    next_actions = self.actor(next_states, sample=True)
    policy_actions = self.actor(states, sample=True)

    next_target_q1, next_target_q2 = self.dist_critic(
        next_states, next_actions, target=True)
    target_q = rewards + self.discount * discounts * tf.minimum(
        next_target_q1, next_target_q2)

    critic_variables = self.critic.trainable_variables

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(critic_variables)
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

      critic_loss = tf.losses.mean_squared_error(target_q, q1) + \
          tf.losses.mean_squared_error(target_q, q2) + self.f_reg * q_reg

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
      q1, q2 = self.dist_critic(states, actions)
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

  @tf.function
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.

    Returns:
      Dictionary with losses to track.
    """

    states, actions, rewards, discounts, next_states = next(replay_buffer_iter)
    rewards = rewards + self.reward_bonus

    critic_dict = self.fit_critic(states, actions, next_states, rewards,
                                  discounts)

    actor_dict = self.fit_actor(states)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

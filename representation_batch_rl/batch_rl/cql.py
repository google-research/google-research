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
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl import policies


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
               num_augmentations=0,
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
      num_augmentations: Number of DrQ-style random crops
      rep_learn_keywords: Representation learning loss to add.
      batch_size: batch size
    """
    del num_augmentations, rep_learn_keywords
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]
    self.batch_size = batch_size

    self.bc = None

    hidden_dims = (256, 256, 256)
    self.actor = policies.DiagGuassianPolicy(
        state_dim, action_spec, hidden_dims=hidden_dims)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.log_cql_alpha = self.log_alpha
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    action_dim = action_spec.shape[0]
    self.critic = critic.Critic(state_dim, action_dim, hidden_dims=hidden_dims)
    self.critic_target = critic.Critic(
        state_dim, action_dim, hidden_dims=hidden_dims)
    critic.soft_update(self.critic, self.critic_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    self.tau = tau

    self.reg = reg
    self.target_entropy = target_entropy
    self.discount = discount

    self.num_cql_actions = num_cql_actions
    self.bc_pretraining_steps = bc_pretraining_steps
    self.min_q_weight = min_q_weight

    self.model_dict = {
        'critic': self.critic,
        'actor': self.actor,
        'critic_target': self.critic_target,
        'actor_optimizer': self.actor_optimizer,
        'critic_optimizer': self.critic_optimizer,
        'alpha_optimizer': self.alpha_optimizer
    }

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  @property
  def cql_alpha(self):
    return tf.exp(self.log_cql_alpha)

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
      q1, q2 = self.critic(states, actions)
      q = tf.minimum(q1, q2)
      actor_loss = tf.reduce_mean(self.alpha * log_probs - q)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.log_alpha *
                                  (-log_probs - self.target_entropy))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {
        'actor_loss': actor_loss,
        'alpha': self.alpha,
        'alpha_loss': alpha_loss
    }

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

    next_q1, next_q2 = self.critic_target(next_states, next_actions)
    target_q = rewards + self.discount * discounts * tf.minimum(
        next_q1, next_q2)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables)

      q1, q2 = self.critic(states, actions)
      critic_loss = (tf.losses.mean_squared_error(target_q, q1) +
                     tf.losses.mean_squared_error(target_q, q2))

      n_states = tf.repeat(states[tf.newaxis, :, :], self.num_cql_actions, 0)
      n_states = tf.reshape(n_states, [-1, n_states.get_shape()[-1]])

      n_rand_actions = tf.random.uniform(
          [tf.shape(n_states)[0],
           actions.get_shape()[-1]], self.actor.action_spec.minimum,
          self.actor.action_spec.maximum)

      n_actions, n_log_probs = self.actor(
          n_states, sample=True, with_log_probs=True)

      q1_rand, q2_rand = self.critic(n_states, n_rand_actions)
      q1_curr_actions, q2_curr_actions = self.critic(n_states, n_actions)

      log_u = -tf.reduce_mean(
          tf.repeat((tf.math.log(2.0 * self.actor.action_scale) *
                     n_rand_actions.shape[-1])[tf.newaxis, :],
                    tf.shape(n_states)[0], 0), 1)

      log_probs_all = tf.concat([n_log_probs, log_u], 0)
      q1_all = tf.concat([q1_curr_actions, q1_rand], 0)
      q2_all = tf.concat([q2_curr_actions, q2_rand], 0)

      def get_qf_loss(q, log_probs):
        q -= log_probs
        q = tf.reshape(q, [-1, tf.shape(states)[0]])
        return tf.math.reduce_logsumexp(q, axis=0)

      min_qf1_loss = get_qf_loss(q1_all, log_probs_all)
      min_qf2_loss = get_qf_loss(q2_all, log_probs_all)

      cql_loss = tf.reduce_mean((min_qf1_loss - q1) + (min_qf2_loss - q2))
      critic_loss += self.min_q_weight * cql_loss

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
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.

    Returns:
      Dictionary with losses to track.
    """

    transition = next(replay_buffer_iter)
    states = transition.observation[:, 0]
    actions = transition.action[:, 0]
    rewards = transition.reward[:, 0]
    next_states = transition.observation[:, 1]
    discounts = transition.discount[:, 0]

    next_actions, _ = self.actor(next_states, sample=True, with_log_probs=True)

    # entropy_rewards = self.discount * discounts * self.alpha * next_log_probs
    # rewards -= entropy_rewards
    critic_dict = self.fit_critic(states, actions, next_states, next_actions,
                                  rewards, discounts)
    actor_dict = self.fit_actor(states)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

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


class PCL(object):
  """Class performing PCL training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               alpha_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_update_period = 1,
               target_entropy = 0.0):
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
    """
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]

    self.actor = policies.DiagGuassianPolicy(state_dim, action_spec)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    self.log_alpha = tf.Variable(tf.math.log(0.1), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

    self.tau = tau
    self.target_update_period = target_update_period

    self.value = critic.CriticNet(state_dim)
    self.value_target = critic.CriticNet(state_dim)
    critic.soft_update(self.value, self.value_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    self.target_entropy = target_entropy
    self.discount = discount

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  def fit_alpha(self, states):
    """Updates critic parameters.

    Args:
      states: A batch of states.

    Returns:
      Actor loss.
    """
    _, log_probs = self.actor(states, sample=True, with_log_probs=True)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.log_alpha *
                                  (-log_probs - self.target_entropy))

    alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
    self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return {
        'alpha':
            self.alpha,
        'alpha_loss':
            alpha_loss,
        'entropy':
            -tf.reduce_mean(log_probs)
    }

  def fit(self, states, actions, next_states,
          rewards,
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
    next_v = self.value_target(next_states)
    target_q = rewards + self.discount * discounts * next_v

    all_vars = (
        list(self.actor.trainable_variables) +
        list(self.value.trainable_variables))
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(all_vars)

      actor_log_probs = self.actor.log_probs(states, actions)
      q = self.value(states) + self.alpha * actor_log_probs

      adv = tf.stop_gradient(target_q - q)
      actor_loss = -tf.reduce_mean(actor_log_probs * adv)
      critic_loss = tf.losses.mean_squared_error(target_q, q)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    critic_grads = tape.gradient(critic_loss, self.value.trainable_variables)

    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))
    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.value.trainable_variables))

    del tape
    if self.critic_optimizer.iterations % self.target_update_period == 0:
      critic.soft_update(self.value, self.value_target, tau=self.tau)

    return {'q': tf.reduce_mean(q), 'critic_loss': critic_loss,
            'actor_log_probs': tf.reduce_mean(actor_log_probs),
            'adv': tf.reduce_mean(adv)}

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

    critic_dict = self.fit(states, actions, next_states, rewards, discounts)
    actor_dict = self.fit_alpha(states)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

  def save_weights(self, path):
    pass

  def load_weights(self, path):
    pass

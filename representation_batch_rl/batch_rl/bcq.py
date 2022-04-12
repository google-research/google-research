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


class BCQ(object):
  """Class performing BCQ training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               num_augmentations=0):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      num_augmentations: number of DrQ-style random crops
    """
    del num_augmentations
    self.bc = None
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]

    self.actor = policies.CVAEPolicy(state_dim, action_spec,
                                     action_spec.shape[0] * 2)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.critic_learner = critic.CriticLearner(state_dim, action_spec.shape[0],
                                               critic_lr, discount, tau)

    self.model_dict = {
        'critic_learner': self.critic_learner,
        'actor': self.actor,
        'actor_optimizer': self.actor_optimizer
    }

  def select_actions(self, states, num_candidates = 10):
    """Samples argmax actions for a batch of states.

    Args:
      states: Batch of states.
      num_candidates: Number of candidate actions to sample.

    Returns:
      Batch of actions.
    """
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

    actor_dict = self.fit_actor(states, actions)

    next_actions = self.select_actions(next_states)

    critic_dict = self.critic_learner.fit_critic(states, actions, next_states,
                                                 next_actions, rewards,
                                                 discounts)
    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.select_actions(states, num_candidates=100)

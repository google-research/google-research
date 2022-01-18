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
"""Implementation of DDPG."""

import typing
from dm_env import specs as dm_env_specs
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl import policies


class DDPG(object):
  """Class performing DDPG training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               cross_norm = False):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      cross_norm: Whether to fit cross norm critic.
    """
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]

    if cross_norm:
      beta_1 = 0.0
    else:
      beta_1 = 0.9

    self.actor = policies.DeterministicPolicy(state_dim, action_spec, 0.3)
    self.actor_optimizer = tf.keras.optimizers.Adam(
        learning_rate=actor_lr, beta_1=beta_1)

    if cross_norm:
      self.critic_learner = critic.CrossNormCriticLearner(
          state_dim, action_spec.shape[0], critic_lr, discount, tau)
    else:
      self.critic_learner = critic.CriticLearner(state_dim,
                                                 action_spec.shape[0],
                                                 critic_lr, discount, tau)

  def fit_actor(self, states):
    """Updates critic parameters.

    Args:
      states: A batch of states.

    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      actions = self.actor(states, sample=False)
      q1, q2 = self.critic_learner.critic(states, actions)
      q = tf.minimum(q1, q2)
      actor_loss = -tf.reduce_mean(q)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    return {'actor_loss': actor_loss}

  @tf.function
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.

    Returns:
      Dictionary with losses to track.
    """

    states, actions, rewards, discounts, next_states = next(replay_buffer_iter)

    next_actions = self.actor(next_states, sample=False)
    critic_dict = self.critic_learner.fit_critic(states, actions, next_states,
                                                 next_actions, rewards,
                                                 discounts)
    actor_dict = self.fit_actor(states)

    return {**actor_dict, **critic_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

  def save_weights(self, path):
    pass

  def load_weights(self, path):
    pass

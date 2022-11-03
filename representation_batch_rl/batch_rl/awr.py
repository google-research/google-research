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


class AWR(object):
  """Class performing AWR/CRR training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               f = 'bin_max',
               temperature = 0.05):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      f: Advantage transformation.
      temperature: Temperature parameter.
    """
    assert len(observation_spec.shape) == 1
    state_dim = observation_spec.shape[0]

    self.actor = policies.DiagGuassianPolicy(state_dim, action_spec)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.critic_learner = critic.CriticLearner(state_dim, action_spec.shape[0],
                                               critic_lr, discount, tau)

    self.f = f
    self.temperature = temperature

  def estimate_value(self,
                     states,
                     num_samples = 4,
                     reduction = 'max'):
    """Estimates value as in CRR.

    Args:
      states: States to estimate values for.
      num_samples: Number of samples for the estimate.
      reduction: What reduction to use.

    Returns:
      Value estimates.
    """
    state_dim = states.shape[-1]
    states = tf.reshape(
        tf.repeat(states[:, tf.newaxis], repeats=num_samples, axis=1),
        [-1, state_dim])
    actions = self.actor(states, sample=True)
    q1, q2 = self.critic_learner.critic(states, actions)
    q = tf.minimum(q1, q2)
    q = tf.reshape(q, [-1, num_samples])
    if 'max' in reduction:
      q = tf.reduce_max(q, -1)
    elif 'mean' in reduction:
      q = tf.reduce_mean(q, -1)

    return q

  def fit_actor(self, states,
                actions):
    """Updates critic parameters.

    Args:
      states: Batch of states.
      actions: Batch of actions.

    Returns:
      Actor loss.
    """
    v = self.estimate_value(states, reduction=self.f.split('_')[-1])
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      log_probs = self.actor.log_probs(states, actions)

      q1, q2 = self.critic_learner.critic(states, actions)
      q = tf.minimum(q1, q2)

      if self.f in 'bin_max':
        a = tf.cast(q > v, tf.float32)
      else:
        a = tf.minimum(tf.exp((q - v) / self.temperature), 20)
      actor_loss = -tf.reduce_mean(log_probs * a)

    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))

    return {'actor_loss': actor_loss, 'weights': tf.reduce_mean(a)}

  @tf.function
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.

    Returns:
      Dict with information to track.
    """

    states, actions, rewards, discounts, next_states = next(replay_buffer_iter)

    next_actions = self.actor(next_states, sample=True)

    critic_dict = self.critic_learner.fit_critic(states, actions, next_states,
                                                 next_actions, rewards,
                                                 discounts)
    actor_dict = self.fit_actor(states, actions)

    return {**critic_dict, **actor_dict}

  @tf.function
  def act(self, states):
    return self.actor(states, sample=False)

  def save_weights(self, path):
    pass

  def load_weights(self, path):
    pass

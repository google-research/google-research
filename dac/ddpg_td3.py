# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Implementation of actor and critic classes for TD3 and update rules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import Actor
from common import CriticDDPG
import numpy as np
import tensorflow.compat.v1 as tf
from utils import soft_update
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe


class CriticTD3(tf.keras.Model):
  """Implementation of a critic."""

  def __init__(self, input_dim):
    """Initializes a policy network.

    Args:
      input_dim: size of the input space
    """
    super(CriticTD3, self).__init__()
    self.critic1 = CriticDDPG(input_dim)
    self.critic2 = CriticDDPG(input_dim)

  def call(self, inputs, actions):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.

    Returns:
      Values of observations.
    """
    return self.critic1(inputs, actions), self.critic2(inputs, actions)


class DDPG(object):
  """Implementation of DDPG and TD3."""

  def __init__(self,
               input_dim,
               action_dim,
               discount=0.99,
               tau=0.005,
               actor_lr=1e-3,
               critic_lr=1e-3,
               use_td3=True,
               policy_noise=0.2,
               policy_noise_clip=0.5,
               policy_update_freq=2,
               get_reward=None,
               use_absorbing_state=False):
    """Initializes actor, critic, target networks and optimizers.

    The class handles absorbing state properly. Absorbing state corresponds to
    a state which a policy gets in after reaching a goal state and stays there
    forever. For most RL problems, we can just assign 0 to all reward after
    the goal. But for GAIL, we need to have an actual absorbing state.

    Args:
       input_dim: size of the observation space.
       action_dim: size of the action space.
       discount: reward discount.
       tau: target networks update coefficient.
       actor_lr: actor learning rate.
       critic_lr: critic learning rate.
       use_td3: whether to use standard ddpg or td3.
       policy_noise: std of gaussian added to critic action input.
       policy_noise_clip: clip added gaussian noise.
       policy_update_freq: perform policy update once per n steps.
       get_reward: a function that given (s,a,s') returns a reward.
       use_absorbing_state: whether to use an absorbing state or not.
    """
    self.discount = discount
    self.tau = tau

    self.use_td3 = use_td3
    self.policy_noise = policy_noise
    self.policy_noise_clip = policy_noise_clip
    self.policy_update_freq = policy_update_freq
    self.get_reward = get_reward
    self.use_absorbing_state = use_absorbing_state

    with tf.variable_scope('actor'):
      self.actor = Actor(input_dim, action_dim)
      with tf.variable_scope('target'):
        self.actor_target = Actor(input_dim, action_dim)

      self.initial_actor_lr = actor_lr
      self.actor_lr = contrib_eager_python_tfe.Variable(actor_lr, name='lr')
      self.actor_step = contrib_eager_python_tfe.Variable(
          0, dtype=tf.int64, name='step')
      self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
      self.actor_optimizer._create_slots(self.actor.variables)  # pylint: disable=protected-access

    soft_update(self.actor.variables, self.actor_target.variables)

    with tf.variable_scope('critic'):
      if self.use_td3:
        self.critic = CriticTD3(input_dim + action_dim)
        with tf.variable_scope('target'):
          self.critic_target = CriticTD3(input_dim + action_dim)
      else:
        self.critic = CriticDDPG(input_dim + action_dim)
        with tf.variable_scope('target'):
          self.critic_target = CriticDDPG(input_dim + action_dim)

      self.critic_step = contrib_eager_python_tfe.Variable(
          0, dtype=tf.int64, name='step')
      self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr)
      self.critic_optimizer._create_slots(self.critic.variables)  # pylint: disable=protected-access

    soft_update(self.critic.variables, self.critic_target.variables)

  def _update_critic_ddpg(self, obs, action, next_obs, reward, mask):
    """Updates parameters of ddpg critic given samples from the batch.

    Args:
       obs: A tfe.Variable with a batch of observations.
       action: A tfe.Variable with a batch of actions.
       next_obs: A tfe.Variable with a batch of next observations.
       reward: A tfe.Variable with a batch of rewards.
       mask: A tfe.Variable with a batch of masks.
    """
    if self.use_absorbing_state:
      # Starting from the goal state we can execute only non-actions.
      a_mask = tf.maximum(0, mask)
      q_next = self.critic_target(next_obs,
                                  self.actor_target(next_obs) * a_mask)
      q_target = reward + self.discount * q_next
    else:
      # Without an absorbing state we assign rewards of 0.
      q_next = self.critic_target(next_obs, self.actor_target(next_obs))
      q_target = reward + self.discount * mask * q_next

    with tf.GradientTape() as tape:
      q_pred = self.critic(obs, action)
      critic_loss = tf.losses.mean_squared_error(q_target, q_pred)

    grads = tape.gradient(critic_loss, self.critic.variables)
    self.critic_optimizer.apply_gradients(
        zip(grads, self.critic.variables), global_step=self.critic_step)

    with contrib_summary.record_summaries_every_n_global_steps(
        100, self.critic_step):
      contrib_summary.scalar('critic/loss', critic_loss, step=self.critic_step)

  def _update_critic_td3(self, obs, action, next_obs, reward, mask):
    """Updates parameters of td3 critic given samples from the batch.

    Args:
       obs: A tfe.Variable with a batch of observations.
       action: A tfe.Variable with a batch of actions.
       next_obs: A tfe.Variable with a batch of next observations.
       reward: A tfe.Variable with a batch of rewards.
       mask: A tfe.Variable with a batch of masks.
    """
    # Avoid using tensorflow random functions since it's impossible to get
    # the state of the random number generator used by TensorFlow.
    target_action_noise = np.random.normal(
        size=action.get_shape(), scale=self.policy_noise).astype('float32')
    target_action_noise = contrib_eager_python_tfe.Variable(target_action_noise)

    target_action_noise = tf.clip_by_value(
        target_action_noise, -self.policy_noise_clip, self.policy_noise_clip)

    noisy_action_targets = self.actor_target(next_obs) + target_action_noise

    clipped_noisy_action_targets = tf.clip_by_value(noisy_action_targets, -1, 1)

    if self.use_absorbing_state:
      # Starting from the goal state we can execute only non-actions.
      a_mask = tf.maximum(0, mask)
      q_next1, q_next2 = self.critic_target(
          next_obs, clipped_noisy_action_targets * a_mask)
      q_next = tf.reduce_min(
          tf.concat([q_next1, q_next2], -1), -1, keepdims=True)
      q_target = reward + self.discount * q_next
    else:
      q_next1, q_next2 = self.critic_target(next_obs,
                                            clipped_noisy_action_targets)
      q_next = tf.reduce_min(
          tf.concat([q_next1, q_next2], -1), -1, keepdims=True)
      q_target = reward + self.discount * mask * q_next

    with tf.GradientTape() as tape:
      q_pred1, q_pred2 = self.critic(obs, action)
      critic_loss = tf.losses.mean_squared_error(
          q_target, q_pred1) + tf.losses.mean_squared_error(
              q_target, q_pred2)

    grads = tape.gradient(critic_loss, self.critic.variables)
    self.critic_optimizer.apply_gradients(
        zip(grads, self.critic.variables), global_step=self.critic_step)

    if self.use_absorbing_state:
      with contrib_summary.record_summaries_every_n_global_steps(
          100, self.critic_step):
        a_mask = tf.maximum(0, -mask)
        if tf.reduce_sum(a_mask).numpy() > 0:
          contrib_summary.scalar(
              'critic/absorbing_reward',
              tf.reduce_sum(reward * a_mask) / tf.reduce_sum(a_mask),
              step=self.critic_step)

    with contrib_summary.record_summaries_every_n_global_steps(
        100, self.critic_step):
      contrib_summary.scalar('critic/loss', critic_loss, step=self.critic_step)

  def _update_actor(self, obs, mask):
    """Updates parameters of critic given samples from the batch.

    Args:
       obs: A tfe.Variable with a batch of observations.
       mask: A tfe.Variable with a batch of masks.
    """
    with tf.GradientTape() as tape:
      if self.use_td3:
        q_pred, _ = self.critic(obs, self.actor(obs))
      else:
        q_pred = self.critic(obs, self.actor(obs))
      if self.use_absorbing_state:
        # Don't update the actor for absorbing states.
        # And skip update if all states are absorbing.
        a_mask = 1.0 - tf.maximum(0, -mask)
        if tf.reduce_sum(a_mask) < 1e-8:
          return
        actor_loss = -tf.reduce_sum(q_pred * a_mask) / tf.reduce_sum(a_mask)
      else:
        actor_loss = -tf.reduce_mean(q_pred)

    grads = tape.gradient(actor_loss, self.actor.variables)
    # Clipping makes training more stable.
    grads, _ = tf.clip_by_global_norm(grads, 40.0)
    self.actor_optimizer.apply_gradients(
        zip(grads, self.actor.variables), global_step=self.actor_step)

    with contrib_summary.record_summaries_every_n_global_steps(
        100, self.actor_step):
      contrib_summary.scalar('actor/loss', actor_loss, step=self.actor_step)

  def update(self, batch, update_actor=True):
    """Updates parameters of TD3 actor and critic given samples from the batch.

    Args:
       batch: A list of timesteps from environment.
       update_actor: a boolean variable, whether to perform a policy update.
    """
    obs = contrib_eager_python_tfe.Variable(
        np.stack(batch.obs).astype('float32'))
    action = contrib_eager_python_tfe.Variable(
        np.stack(batch.action).astype('float32'))
    next_obs = contrib_eager_python_tfe.Variable(
        np.stack(batch.next_obs).astype('float32'))
    mask = contrib_eager_python_tfe.Variable(
        np.stack(batch.mask).astype('float32'))

    if self.get_reward is not None:
      reward = self.get_reward(obs, action, next_obs)
    else:
      reward = contrib_eager_python_tfe.Variable(
          np.stack(batch.reward).astype('float32'))

    if self.use_td3:
      self._update_critic_td3(obs, action, next_obs, reward, mask)
    else:
      self._update_critic_ddpg(obs, action, next_obs, reward, mask)

    if self.critic_step.numpy() % self.policy_update_freq == 0:
      if update_actor:
        self._update_actor(obs, mask)
        soft_update(self.actor.variables, self.actor_target.variables, self.tau)
      soft_update(self.critic.variables, self.critic_target.variables, self.tau)

  @property
  def variables(self):
    """Returns all variables including optimizer variables.

    Returns:
      A dictionary of actor/critic/actor_target/critic_target/optimizers
      variables.
    """
    actor_vars = (
        self.actor.variables + self.actor_target.variables +
        self.actor_optimizer.variables() + [self.actor_step])
    critic_vars = (
        self.critic.variables + self.critic_target.variables +
        self.critic_optimizer.variables() + [self.critic_step])

    return actor_vars + critic_vars

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

"""Implementation of twin_sac, a mix of TD3 (https://arxiv.org/abs/1802.09477) and SAC (https://arxiv.org/abs/1801.01290, https://arxiv.org/abs/1812.05905).

Overall structure and hyperparameters are taken from TD3. However, the algorithm
itself represents a version of SAC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from value_dice import keras_utils

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = np.finfo(np.float32).eps


def soft_update(net, target_net, tau=0.005):
  """Soft target network update used in https://arxiv.org/pdf/1509.02971.pdf.

  Args:
    net: A neural network.
    target_net: A target neural network to apply a soft update to.
    tau: A soft update coefficient.
  """
  for var, target_var in zip(net.variables, target_net.variables):
    new_value = var * tau + target_var * (1 - tau)
    target_var.assign(new_value)


class Actor(tf.keras.Model):
  """Gaussian policy with TanH squashing."""

  def __init__(self, state_dim, action_dim):
    """Creates an actor.

    Args:
      state_dim: State size.
      action_dim: Action size.
    """
    super(Actor, self).__init__()
    self.trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_shape=(state_dim,),
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(2 * action_dim, kernel_initializer='orthogonal')
    ])

  def get_dist_and_mode(self, states):
    """Returns a tf.Distribution for given states.

    Args:
      states: A batch of states.
    """
    out = self.trunk(states)
    mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
    mode = tf.nn.tanh(mu)

    log_std = tf.nn.tanh(log_std)
    assert LOG_STD_MAX > LOG_STD_MIN
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)

    dist = ds.TransformedDistribution(
        ds.Sample(
            ds.Normal(tf.zeros(mu.shape[:-1]), 1.0),
            sample_shape=mu.shape[-1:]),
        tfp.bijectors.Chain([
            tfp.bijectors.Tanh(),
            tfp.bijectors.Shift(shift=mu),
            tfp.bijectors.ScaleMatvecDiag(scale_diag=std)
        ]))

    return dist, mode

  @tf.function
  def get_log_prob(self, states, actions):
    """Evaluate log probs for actions conditined on states.

    Args:
      states: A batch of states.
      actions: A batch of actions to evaluate log probs on.

    Returns:
      Log probabilities of actions.
    """
    dist, _ = self.get_dist_and_mode(states)
    log_probs = dist.log_prob(actions)
    log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
    return log_probs

  @tf.function
  def call(self, states):
    """Computes actions for given inputs.

    Args:
      states: A batch of states.

    Returns:
      A mode action, a sampled action and log probability of the sampled action.
    """
    dist, mode = self.get_dist_and_mode(states)
    samples = dist.sample()
    log_probs = dist.log_prob(samples)
    log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
    return mode, samples, log_probs


class Critic(tf.keras.Model):
  """A critic network that estimates a dual Q-function."""

  def __init__(self, state_dim, action_dim):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
    """
    super(Critic, self).__init__()
    self.critic1 = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_shape=(state_dim + action_dim,),
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(1, kernel_initializer='orthogonal')
    ])
    self.critic2 = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_shape=(state_dim + action_dim,),
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(1, kernel_initializer='orthogonal')
    ])

  @tf.function
  def call(self, states, actions):
    """Returns Q-value estimates for given states and actions.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      Two estimates of Q-values.
    """
    x = tf.concat([states, actions], -1)

    q1 = self.critic1(x)

    q2 = self.critic2(x)

    return q1, q2


class SAC(object):
  """Class performing Soft Actor Critic training."""

  def __init__(self,
               state_dim,
               action_dim,
               log_interval,
               actor_lr=1e-3,
               critic_lr=1e-3,
               alpha_init=1.0,
               learn_alpha=True,
               rewards_fn=lambda s, a, r: r):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      log_interval: Log losses every N steps.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      alpha_init: Initial temperature value.
      learn_alpha: Whether to learn alpha or not.
      rewards_fn: A function of (s, a, r) that returns or overwrites rewards.
    """
    self.actor = Actor(state_dim, action_dim)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    self.avg_actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
    self.avg_alpha_loss = tf.keras.metrics.Mean('alpha_loss', dtype=tf.float32)
    self.avg_actor_entropy = tf.keras.metrics.Mean(
        'actor_entropy', dtype=tf.float32)
    self.avg_alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)

    self.critic = Critic(state_dim, action_dim)
    self.critic_target = Critic(state_dim, action_dim)
    soft_update(self.critic, self.critic_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    self.avg_critic_loss = tf.keras.metrics.Mean(
        'critic_loss', dtype=tf.float32)

    self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
    self.learn_alpha = learn_alpha
    self.alpha_optimizer = tf.keras.optimizers.Adam()

    self.log_interval = log_interval
    self.rewards_fn = rewards_fn

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  def fit_critic(self, states, actions, next_states, rewards, masks, discount):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      next_states: A batch of next states.
      rewards: A batch of rewards.
      masks: A batch of masks indicating the end of the episodes.
      discount: An MDP discount factor.

    Returns:
      Critic loss.
    """
    _, next_actions, log_probs = self.actor(next_states)

    target_q1, target_q2 = self.critic_target(next_states, next_actions)
    target_v = tf.minimum(target_q1, target_q2) - self.alpha * log_probs
    target_q = rewards + discount * masks * target_v

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.variables)

      q1, q2 = self.critic(states, actions)
      critic_loss = (
          tf.losses.mean_squared_error(target_q, q1) +
          tf.losses.mean_squared_error(target_q, q2))
      critic_loss = tf.reduce_mean(critic_loss)

    critic_grads = tape.gradient(critic_loss, self.critic.variables)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.variables))

    return critic_loss

  def fit_actor(self, states, target_entropy):
    """Updates actor parameters.

    Args:
      states: A batch of states.
      target_entropy: Target entropy value for alpha.

    Returns:
      Actor and alpha losses.
    """
    is_non_absorbing_mask = tf.cast(tf.equal(states[:, -1:], 0.0), tf.float32)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.variables)
      _, actions, log_probs = self.actor(states)
      q1, q2 = self.critic(states, actions)
      q = tf.minimum(q1, q2)
      actor_loss = tf.reduce_sum(is_non_absorbing_mask *
                                 (self.alpha * log_probs - q)) / (
                                     tf.reduce_sum(is_non_absorbing_mask) + EPS)

      actor_loss += keras_utils.orthogonal_regularization(self.actor.trunk)

    actor_grads = tape.gradient(actor_loss, self.actor.variables)
    self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_sum(is_non_absorbing_mask * self.alpha *
                                 (-log_probs - target_entropy)) / (
                                     tf.reduce_sum(is_non_absorbing_mask) + EPS)

    if self.learn_alpha:
      alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
      self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return actor_loss, alpha_loss, -log_probs

  @tf.function
  def train_bc(self, expert_dataset_iter):
    """Performs a single training step of behavior clonning.

    The method optimizes MLE on the expert dataset.

    Args:
      expert_dataset_iter: An tensorflow graph iteratable object.
    """

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.variables)
      states, actions, _ = next(expert_dataset_iter)
      log_probs = self.actor.get_log_prob(states, actions)
      actor_loss = tf.reduce_mean(
          -log_probs) + keras_utils.orthogonal_regularization(self.actor.trunk)

    actor_grads = tape.gradient(actor_loss, self.actor.variables)
    self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.variables))

    self.avg_actor_loss(actor_loss)

    if tf.equal(self.actor_optimizer.iterations % self.log_interval, 0):
      tf.summary.scalar(
          'train bc/actor_loss',
          self.avg_actor_loss.result(),
          step=self.actor_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_actor_loss)

  @tf.function
  def train(self,
            replay_buffer_iter,
            discount=0.99,
            tau=0.005,
            target_entropy=0,
            actor_update_freq=2):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object.
      discount: A discount used to compute returns.
      tau: A soft updates discount.
      target_entropy: A target entropy for alpha.
      actor_update_freq: A frequency of the actor network updates.

    Returns:
      Actor and alpha losses.
    """
    states, actions, next_states, rewards, masks = next(replay_buffer_iter)[0]

    rewards = self.rewards_fn(states, actions, rewards)

    critic_loss = self.fit_critic(states, actions, next_states, rewards, masks,
                                  discount)

    self.avg_critic_loss(critic_loss)
    if tf.equal(self.critic_optimizer.iterations % self.log_interval, 0):
      tf.summary.scalar(
          'train sac/critic_loss',
          self.avg_critic_loss.result(),
          step=self.critic_optimizer.iterations)
      keras_utils.my_reset_states(self.avg_critic_loss)

    if tf.equal(self.critic_optimizer.iterations % actor_update_freq, 0):
      actor_loss, alpha_loss, entropy = self.fit_actor(states, target_entropy)
      soft_update(self.critic, self.critic_target, tau=tau)

      self.avg_actor_loss(actor_loss)
      self.avg_alpha_loss(alpha_loss)
      self.avg_actor_entropy(entropy)
      self.avg_alpha(self.alpha)
      if tf.equal(self.actor_optimizer.iterations % self.log_interval, 0):
        tf.summary.scalar(
            'train sac/actor_loss',
            self.avg_actor_loss.result(),
            step=self.actor_optimizer.iterations)
        keras_utils.my_reset_states(self.avg_actor_loss)

        tf.summary.scalar(
            'train sac/alpha_loss',
            self.avg_alpha_loss.result(),
            step=self.actor_optimizer.iterations)
        keras_utils.my_reset_states(self.avg_alpha_loss)

        tf.summary.scalar(
            'train sac/actor entropy',
            self.avg_actor_entropy.result(),
            step=self.actor_optimizer.iterations)
        keras_utils.my_reset_states(self.avg_actor_entropy)

        tf.summary.scalar(
            'train sac/alpha',
            self.avg_alpha.result(),
            step=self.actor_optimizer.iterations)
        keras_utils.my_reset_states(self.avg_alpha)

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

"""Implementation of AlgaeDICE.

Based on the publication "AlgaeDICE: Policy Gradient from Arbitrary Experience"
by Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, Dale Schuurmans.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import algae_dice.keras_utils as keras_utils

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def soft_update(net, target_net, tau=0.005):
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
    """Returns a tf.Distribution for given states modes of this distribution.

    Args:
      states: A batch of states.
    """
    out = self.trunk(states)
    mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
    mode = tf.nn.tanh(mu)

    log_std = tf.nn.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)

    dist = ds.TransformedDistribution(
        distribution=ds.Normal(loc=0., scale=1.),
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.Tanh(),
            tfp.bijectors.Affine(shift=mu, scale_diag=std),
        ]),
        event_shape=[mu.shape[-1]],
        batch_shape=[mu.shape[0]])
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
    self.critic = tf.keras.Sequential([
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

    q = self.critic(x)

    return q


class DoubleCritic(tf.keras.Model):
  """A critic network that estimates a dual Q-function."""

  def __init__(self, state_dim, action_dim):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
    """
    super(DoubleCritic, self).__init__()
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


class ALGAE(object):
  """Class performing algae training."""

  def __init__(self,
               state_dim,
               action_dim,
               log_interval,
               actor_lr=1e-3,
               critic_lr=1e-3,
               alpha_init=1.0,
               learn_alpha=True,
               algae_alpha=1.0,
               use_dqn=True,
               use_init_states=True,
               exponent=2.0):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      log_interval: Log losses every N steps.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      alpha_init: Initial temperature value for causal entropy regularization.
      learn_alpha: Whether to learn alpha or not.
      algae_alpha: Algae regularization weight.
      use_dqn: Whether to use double networks for target value.
      use_init_states: Whether to use initial states in objective.
      exponent: Exponent p of function f(x) = |x|^p / p.
    """
    self.actor = Actor(state_dim, action_dim)
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    self.avg_actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
    self.avg_alpha_loss = tf.keras.metrics.Mean('alpha_loss', dtype=tf.float32)
    self.avg_actor_entropy = tf.keras.metrics.Mean(
        'actor_entropy', dtype=tf.float32)
    self.avg_alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)
    self.avg_lambda = tf.keras.metrics.Mean('lambda', dtype=tf.float32)
    self.use_init_states = use_init_states

    if use_dqn:
      self.critic = DoubleCritic(state_dim, action_dim)
      self.critic_target = DoubleCritic(state_dim, action_dim)
    else:
      self.critic = Critic(state_dim, action_dim)
      self.critic_target = Critic(state_dim, action_dim)
    soft_update(self.critic, self.critic_target, tau=1.0)
    self._lambda = tf.Variable(0.0, trainable=True)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    self.avg_critic_loss = tf.keras.metrics.Mean(
        'critic_loss', dtype=tf.float32)

    self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
    self.learn_alpha = learn_alpha
    self.alpha_optimizer = tf.keras.optimizers.Adam()

    self.log_interval = log_interval

    self.algae_alpha = algae_alpha
    self.use_dqn = use_dqn
    self.exponent = exponent
    if self.exponent <= 1:
      raise ValueError('Exponent must be greather than 1, but received %f.' %
                       self.exponent)
    self.f = lambda resid: tf.pow(tf.abs(resid), self.exponent) / self.exponent
    clip_resid = lambda resid: tf.clip_by_value(resid, 0.0, 1e6)
    self.fgrad = lambda resid: tf.pow(clip_resid(resid), self.exponent - 1)

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  def critic_mix(self, s, a):
    if self.use_dqn:
      target_q1, target_q2 = self.critic_target(s, a)
      target_q = tf.minimum(target_q1, target_q2)
      q1, q2 = self.critic(s, a)
      return q1 * 0.05 + target_q * 0.95, q2 * 0.05 + target_q * 0.95,
    else:
      return self.critic(s, a) * 0.05 + self.critic_target(s, a) * 0.95

  def fit_critic(self, states, actions, next_states, rewards, masks, discount,
                 init_states):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      next_states: A batch of next states.
      rewards: A batch of rewards.
      masks: A batch of masks indicating the end of the episodes.
      discount: An MDP discount factor.
      init_states: A batch of init states from the MDP.

    Returns:
      Critic loss.
    """
    _, init_actions, _ = self.actor(init_states)
    _, next_actions, next_log_probs = self.actor(next_states)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.variables + [self._lambda])

      if self.use_dqn:
        target_q1, target_q2 = self.critic_mix(next_states, next_actions)

        target_q1 = target_q1 - self.alpha * next_log_probs
        target_q2 = target_q2 - self.alpha * next_log_probs

        target_q1 = rewards + discount * masks * target_q1
        target_q2 = rewards + discount * masks * target_q2

        q1, q2 = self.critic(states, actions)
        init_q1, init_q2 = self.critic(init_states, init_actions)

        if discount == 1:
          critic_loss1 = tf.reduce_mean(
              self.f(self._lambda + self.algae_alpha + target_q1 - q1) -
              self.algae_alpha * self._lambda)

          critic_loss2 = tf.reduce_mean(
              self.f(self._lambda + self.algae_alpha + target_q2 - q2) -
              self.algae_alpha * self._lambda)
        else:
          critic_loss1 = tf.reduce_mean(
              self.f(target_q1 - q1) +
              (1 - discount) * init_q1 * self.algae_alpha)

          critic_loss2 = tf.reduce_mean(
              self.f(target_q2 - q2) +
              (1 - discount) * init_q2 * self.algae_alpha)

        critic_loss = (critic_loss1 + critic_loss2)
      else:
        target_q = self.critic_mix(next_states, next_actions)
        target_q = target_q - self.alpha * next_log_probs
        target_q = rewards + discount * masks * target_q

        q = self.critic(states, actions)
        init_q = self.critic(init_states, init_actions)

        if discount == 1:
          critic_loss = tf.reduce_mean(
              self.f(self._lambda + self.algae_alpha + target_q - q) -
              self.algae_alpha * self._lambda)
        else:
          critic_loss = tf.reduce_mean(
              self.f(target_q - q) + (1 - discount) * init_q * self.algae_alpha)

    critic_grads = tape.gradient(critic_loss,
                                 self.critic.variables + [self._lambda])

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.variables + [self._lambda]))

    return critic_loss

  def fit_actor(self, states, actions, next_states, rewards, masks, discount,
                target_entropy, init_states):
    """Updates critic parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      next_states: A batch of next states.
      rewards: A batch of rewards.
      masks: A batch of masks indicating the end of the episodes.
      discount: An MDP discount factor.
      target_entropy: Target entropy value for alpha.
      init_states: A batch of init states from the MDP.

    Returns:
      Actor and alpha losses.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.variables)
      _, init_actions, _ = self.actor(init_states)
      _, next_actions, next_log_probs = self.actor(next_states)

      if self.use_dqn:
        target_q1, target_q2 = self.critic_mix(next_states, next_actions)
        target_q1 = target_q1 - self.alpha * next_log_probs
        target_q2 = target_q2 - self.alpha * next_log_probs
        target_q1 = rewards + discount * masks * target_q1
        target_q2 = rewards + discount * masks * target_q2

        q1, q2 = self.critic(states, actions)
        init_q1, init_q2 = self.critic(init_states, init_actions)

        if discount == 1:
          actor_loss1 = -tf.reduce_mean(
              tf.stop_gradient(
                  self.fgrad(self._lambda + self.algae_alpha + target_q1 - q1))
              * (target_q1 - q1))

          actor_loss2 = -tf.reduce_mean(
              tf.stop_gradient(
                  self.fgrad(self._lambda + self.algae_alpha + target_q2 - q2))
              * (target_q2 - q2))
        else:
          actor_loss1 = -tf.reduce_mean(
              tf.stop_gradient(self.fgrad(target_q1 - q1)) * (target_q1 - q1) +
              (1 - discount) * init_q1 * self.algae_alpha)

          actor_loss2 = -tf.reduce_mean(
              tf.stop_gradient(self.fgrad(target_q2 - q2)) * (target_q2 - q2) +
              (1 - discount) * init_q2 * self.algae_alpha)

        actor_loss = (actor_loss1 + actor_loss2) / 2.0
      else:
        target_q = self.critic_mix(next_states, next_actions)
        target_q = target_q - self.alpha * next_log_probs
        target_q = rewards + discount * masks * target_q

        q = self.critic(states, actions)
        init_q = self.critic(init_states, init_actions)

        if discount == 1:
          actor_loss = -tf.reduce_mean(
              tf.stop_gradient(
                  self.fgrad(self._lambda + self.algae_alpha + target_q - q)) *
              (target_q - q))
        else:
          actor_loss = -tf.reduce_mean(
              tf.stop_gradient(self.fgrad(target_q - q)) * (target_q - q) +
              (1 - discount) * init_q * self.algae_alpha)
      actor_loss += keras_utils.orthogonal_regularization(self.actor.trunk)

    actor_grads = tape.gradient(actor_loss, self.actor.variables)
    self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch([self.log_alpha])
      alpha_loss = tf.reduce_mean(self.alpha *
                                  (-next_log_probs - target_entropy))

    if self.learn_alpha:
      alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
      self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

    return actor_loss, alpha_loss, -next_log_probs

  @tf.function
  def train(self,
            replay_buffer_iter,
            init_replay_buffer,
            discount=0.99,
            tau=0.005,
            target_entropy=0,
            actor_update_freq=2):
    """Performs a single training step for critic and actor.

    Args:
      replay_buffer_iter: An tensorflow graph iteratable object for sampling
        transitions.
      init_replay_buffer: An tensorflow graph iteratable object for sampling
        init states.
      discount: A discount used to compute returns.
      tau: A soft updates discount.
      target_entropy: A target entropy for alpha.
      actor_update_freq: A frequency of the actor network updates.

    Returns:
      Actor and alpha losses.
    """
    states, actions, next_states, rewards, masks = next(replay_buffer_iter)[0]
    if self.use_init_states:
      init_states = next(init_replay_buffer)[0]
    else:
      init_states = states

    critic_loss = self.fit_critic(states, actions, next_states, rewards, masks,
                                  discount, init_states)
    step = 0
    self.avg_critic_loss(critic_loss)
    if tf.equal(self.critic_optimizer.iterations % self.log_interval, 0):
      train_measurements = [
          ('train/critic_loss', self.avg_critic_loss.result()),
      ]
      for (label, value) in train_measurements:
        tf.summary.scalar(label, value, step=step)

      keras_utils.my_reset_states(self.avg_critic_loss)

    if tf.equal(self.critic_optimizer.iterations % actor_update_freq, 0):
      actor_loss, alpha_loss, entropy = self.fit_actor(states, actions,
                                                       next_states, rewards,
                                                       masks, discount,
                                                       target_entropy,
                                                       init_states)
      soft_update(self.critic, self.critic_target, tau=tau)

      self.avg_actor_loss(actor_loss)
      self.avg_alpha_loss(alpha_loss)
      self.avg_actor_entropy(entropy)
      self.avg_alpha(self.alpha)
      self.avg_lambda(self._lambda)
      if tf.equal(self.actor_optimizer.iterations % self.log_interval, 0):
        train_measurements = [
            ('train/actor_loss', self.avg_actor_loss.result()),
            ('train/alpha_loss', self.avg_alpha_loss.result()),
            ('train/actor entropy', self.avg_actor_entropy.result()),
            ('train/alpha', self.avg_alpha.result()),
            ('train/lambda', self.avg_lambda.result()),
        ]
        for (label, value) in train_measurements:
          tf.summary.scalar(label, value, step=self.critic_optimizer.iterations)
        keras_utils.my_reset_states(self.avg_actor_loss)
        keras_utils.my_reset_states(self.avg_alpha_loss)
        keras_utils.my_reset_states(self.avg_actor_entropy)
        keras_utils.my_reset_states(self.avg_alpha)
        keras_utils.my_reset_states(self.avg_lambda)

  def evaluate(self, env, num_episodes=10, max_episode_steps=None):
    """Evaluates the policy.

    Args:
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
      max_episode_steps: Max steps in an episode.

    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = 0
    total_returns = 0

    for _ in range(num_episodes):
      state = env.reset()
      done = False
      episode_timesteps = 0
      while not done:
        action, _, _ = self.actor(np.array([state]))
        action = action[0].numpy()

        next_state, reward, done, _ = env.step(action)
        if (max_episode_steps is not None and
            episode_timesteps + 1 == max_episode_steps):
          done = True

        total_returns += reward
        total_timesteps += 1
        episode_timesteps += 1
        state = next_state

    return total_returns / num_episodes, total_timesteps / num_episodes

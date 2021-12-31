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

"""Critic module."""

import typing

from dm_env import specs as dm_env_specs
import tensorflow as tf

from rl_repr.batch_rl import keras_utils
from rl_repr.batch_rl import policies


def soft_update(net, target_net, tau=0.005):
  for var, target_var in zip(net.variables, target_net.variables):
    new_value = var * tau + target_var * (1 - tau)
    target_var.assign(new_value)


class CriticNet(tf.keras.Model):
  """A critic network."""

  def __init__(self,
               state_dim,
               action_dim = None,
               hidden_dims = (256, 256),
               cross_norm = False):
    """Creates a neural net.

    Args:
      state_dim: State size.
      action_dim: Action size.
      hidden_dims: List of hidden dimensions.
      cross_norm: Whether to use cross norm.
    """
    super().__init__()

    embed_dim = state_dim
    inputs = tf.keras.Input(shape=[None, state_dim])
    outputs = tf.keras.layers.LSTM(embed_dim)(inputs)
    if action_dim is None:
      inputs = tf.keras.Input(shape=(embed_dim,))
    else:
      inputs = tf.keras.Input(shape=(embed_dim + action_dim,))

    if cross_norm:
      layers = []
      relu_gain = tf.math.sqrt(2.0)
      relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
      near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)
      for hidden_dim in hidden_dims:
        layers += [
            tf.keras.layers.Dense(
                hidden_dim, use_bias=False, kernel_initializer=relu_orthogonal),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.ReLU()
        ]
      outputs = tf.keras.Sequential(
          layers + [tf.keras.layers.Dense(
              1, kernel_initializer=near_zero_orthogonal)]
          )(inputs)

      self.main = tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
      self.main = keras_utils.create_mlp(
          inputs.shape[-1], 1, hidden_dims=hidden_dims, activation=tf.nn.swish)

  @tf.function
  def call(self,
           states,
           actions = None,
           training = False):
    """Returns Q-value estimates for given states and actions.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      training: Whether to run in training mode.

    Returns:
      Two estimates of Q-values.
    """
    if actions is None:
      x = states
    else:
      x = tf.concat([states, actions], -1)

    return tf.squeeze(self.main(x, training), 1)


class SoftCriticNet(tf.keras.Model):
  """A soft critic network that estimates a dual Q-function."""

  def __init__(self,
               state_dim,
               action_spec,
               hidden_dims = (256, 256)):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action specification.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()
    self.value = CriticNet(state_dim, action_dim=None, hidden_dims=hidden_dims)

    self.advantage = policies.DiagGuassianPolicy(
        state_dim, action_spec, hidden_dims=hidden_dims)

    self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)

  @tf.function
  def call(self,
           states,
           actions):
    """Returns Q-value estimate for given states and actions.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      Estimate of Q-value.
    """
    value = self.value(states)
    advantage = self.advantage.log_probs(states, actions)
    alpha = tf.exp(self.log_alpha)
    return value + advantage * alpha


class SoftCritic(tf.keras.Model):
  """A critic network that estimates a dual Q-function."""

  def __init__(self,
               state_dim,
               action_spec,
               hidden_dims = (256, 256)):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action specification.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()
    self.critic1 = SoftCriticNet(
        state_dim, action_spec, hidden_dims=hidden_dims)
    self.critic2 = SoftCriticNet(
        state_dim, action_spec, hidden_dims=hidden_dims)

  @tf.function
  def call(self,
           states,
           actions):
    """Returns Q-value estimates for given states and actions.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      Two estimates of Q-values.
    """

    q1 = self.critic1(states, actions)
    q2 = self.critic2(states, actions)

    return q1, q2


class Critic(tf.keras.Model):
  """A critic network that estimates a dual Q-function."""

  def __init__(self,
               state_dim,
               action_dim,
               hidden_dims = (256, 256),
               cross_norm = False):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      hidden_dims: List of hidden dimensions.
      cross_norm: Whether to use cross norm.
    """
    super().__init__()
    self.critic1 = CriticNet(
        state_dim, action_dim, hidden_dims=hidden_dims, cross_norm=cross_norm)
    self.critic2 = CriticNet(
        state_dim, action_dim, hidden_dims=hidden_dims, cross_norm=cross_norm)

  @tf.function
  def call(self,
           states,
           actions,
           training = False):
    """Returns Q-value estimates for given states and actions.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      training: Whether to run in training mode.

    Returns:
      Two estimates of Q-values.
    """

    q1 = self.critic1(states, actions, training)
    q2 = self.critic2(states, actions, training)

    return q1, q2


class CrossNormCriticLearner(object):
  """Class performing cross norm critic fitting."""

  def __init__(self,
               state_dim,
               action_dim,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_update_period = 1):
    """Initializes critic learner.

    Args:
      state_dim: State size.
      action_dim: Action size.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_update_period: Target network update period.
    """
    self.discount = discount

    self.tau = tau
    self.target_update_period = target_update_period

    self.critic = Critic(state_dim, action_dim, cross_norm=True)
    self.critic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=critic_lr, beta_1=0.0)

  def fit_critic(self, get_inputs_fn,
                 extra_vars):
    """Updates critic parameters.

    Args:
      get_inputs_fn: Function for getting input states, actions,
        next_states, next_actions, rewards, and discounts
      extra_vars: Extra variable from embedder.

    Returns:
      Dictionary with information to track.
    """

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.variables + extra_vars)

      states, actions, next_states, next_actions, rewards, discounts = get_inputs_fn()

      all_states = tf.concat([states, next_states], axis=0)
      all_actions = tf.concat([actions, next_actions], axis=0)

      all_q1, all_q2 = self.critic(all_states, all_actions, training=True)

      q1, next_q1 = tf.split(all_q1, num_or_size_splits=2, axis=0)
      q2, next_q2 = tf.split(all_q2, num_or_size_splits=2, axis=0)

      next_q = tf.minimum(next_q1, next_q2)
      target_q = rewards + self.discount * discounts * tf.stop_gradient(next_q)

      critic_loss = tf.losses.mean_squared_error(target_q, q1) + \
          tf.losses.mean_squared_error(target_q, q2)

    critic_grads = tape.gradient(critic_loss, self.critic.variables + extra_vars)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.variables + extra_vars))

    return {
        'q1': tf.reduce_mean(q1),
        'q2': tf.reduce_mean(q2),
        'critic_loss': critic_loss
    }


class CriticLearner(object):
  """Class performing critic fitting."""

  def __init__(self,
               state_dim,
               action_dim,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_update_period = 1,
               hidden_dims = (256, 256)):
    """Initializes critic learner.

    Args:
      state_dim: State size.
      action_dim: Action size.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_update_period: Target network update period.
      hidden_dims: List of hidden dimensions.
    """
    self.discount = discount
    self.tau = tau
    self.target_update_period = target_update_period

    self.critic = Critic(state_dim, action_dim, hidden_dims=hidden_dims)
    self.critic_target = Critic(state_dim, action_dim, hidden_dims=hidden_dims)
    soft_update(self.critic, self.critic_target, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  def fit_critic(self, get_inputs_fn,
                 extra_vars):
    """Updates critic parameters.

    Args:
      get_inputs_fn: Function for getting input states, actions,
        next_states, next_actions, rewards, and discounts
      extra_vars: Extra variable from embedder.

    Returns:
      Dictionary with information to track.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables + extra_vars)

      states, actions, next_states, next_actions, rewards, discounts = get_inputs_fn()
      next_q1, next_q2 = self.critic_target(next_states, next_actions)
      target_q = rewards + self.discount * discounts * tf.minimum(
          next_q1, next_q2)
      target_q = tf.stop_gradient(target_q)

      q1, q2 = self.critic(states, actions)

      critic_loss = tf.losses.mean_squared_error(target_q, q1) + \
          tf.losses.mean_squared_error(target_q, q2)

    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables + extra_vars)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.trainable_variables + extra_vars))

    if self.critic_optimizer.iterations % self.target_update_period == 0:
      soft_update(self.critic, self.critic_target, tau=self.tau)

    return {'q1': tf.reduce_mean(q1), 'q2': tf.reduce_mean(q2),
            'critic_loss': critic_loss}

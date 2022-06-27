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

"""Implementation of critic modules."""

import typing

import tensorflow as tf


def soft_update(net, target_net, tau=0.005):
  for var, target_var in zip(net.variables, target_net.variables):
    new_value = var * tau + target_var * (1 - tau)
    target_var.assign(new_value)


class CriticNet(tf.keras.Model):
  """A critic network."""

  def __init__(self,
               state_dim,
               action_dim,
               hidden_dims = (256, 256)):
    """Creates a neural net.

    Args:
      state_dim: State size.
      action_dim: Action size.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()
    relu_gain = tf.math.sqrt(2.0)
    relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
    near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

    inputs = tf.keras.Input(shape=(state_dim + action_dim,))

    layers = []
    for hidden_dim in hidden_dims:
      layers.append(
          tf.keras.layers.Dense(
              hidden_dim,
              activation=tf.nn.relu,
              kernel_initializer=relu_orthogonal))
    outputs = tf.keras.Sequential(
        layers + [tf.keras.layers.Dense(
            1, kernel_initializer=near_zero_orthogonal)]
        )(inputs)

    self.main = tf.keras.Model(inputs=inputs, outputs=outputs)

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
    return tf.squeeze(self.main(x), 1)


class Critic(tf.keras.Model):
  """A critic network that estimates a dual Q-function."""

  def __init__(self,
               state_dim,
               action_dim,
               hidden_dims = (256, 256)):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()

    self.critic1 = CriticNet(state_dim, action_dim, hidden_dims=hidden_dims)
    self.critic2 = CriticNet(state_dim, action_dim, hidden_dims=hidden_dims)

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

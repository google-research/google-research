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

"""Implementation of common models for TD3, DDPG."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class Mask(Enum):
  ABSORBING = -1.0
  DONE = 0.0
  NOT_DONE = 1.0


class Actor(tf.keras.Model):
  """Implementation of a determistic policy."""

  def __init__(self, input_dim, action_dim):
    """Initializes a policy network.

    Args:
      input_dim: size of the input space
      action_dim: size of the action space
    """
    super(Actor, self).__init__()

    self.main = tf.keras.Sequential([
        tf.layers.Dense(
            units=400,
            activation='relu',
            kernel_initializer=tf.orthogonal_initializer(),
            input_shape=(input_dim,)),
        tf.layers.Dense(
            units=300,
            activation='relu',
            kernel_initializer=tf.orthogonal_initializer()),
        tf.layers.Dense(
            units=action_dim,
            activation='tanh',
            kernel_initializer=tf.orthogonal_initializer(0.01))
    ])

  def call(self, inputs):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).

    Returns:
      Actions produced by a policy.
    """
    return self.main(inputs)


class StochasticActor(tf.keras.Model):
  """Implements stochastic-actor."""

  def __init__(self, input_dim, action_dim):
    super(StochasticActor, self).__init__()

    self.mu = tf.keras.Sequential([
        tf.layers.Dense(
            units=64,
            activation='tanh',
            kernel_initializer=tf.orthogonal_initializer(),
            input_shape=(input_dim,)),
        tf.layers.Dense(
            units=64,
            activation='tanh',
            kernel_initializer=tf.orthogonal_initializer()),
        tf.layers.Dense(
            units=action_dim,
            activation=None,
            kernel_initializer=tf.orthogonal_initializer(0.01))
    ])

    # We exponentiate the logsig to get sig (hence we don't need softplus).
    self.logsig = tf.get_variable(
        name='logsig',
        shape=[1, action_dim],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)

  @property
  def variables(self):
    """Overrides the variables property of tf.keras.Model.

    Required to include variables defined through tf.get_variable().

    Returns:
      List of trainable variables.
    """
    mu_var = self.mu.variables
    sig_var = self.logsig
    return mu_var + [sig_var]

  def dist(self, mu, sig):
    return tfp.distributions.MultivariateNormalDiag(
        loc=mu,
        scale_diag=sig)

  def sample(self, mu, sig):
    return self.dist(mu, sig).sample()

  def call(self, inputs):
    """Returns action distribution, given a state."""
    act_mu = self.mu(inputs)
    act_sig = tf.exp(tf.tile(self.logsig, [tf.shape(act_mu)[0], 1]))
    tf.assert_equal(act_mu.shape, act_sig.shape)

    act_dist = self.dist(act_mu, act_sig)
    return act_dist


class Critic(tf.keras.Model):
  """Implementation of state-value function."""

  def __init__(self, input_dim):
    super(Critic, self).__init__()
    self.main = tf.keras.Sequential([
        tf.layers.Dense(
            units=64,
            input_shape=(input_dim,),
            activation='tanh',
            kernel_initializer=tf.orthogonal_initializer()),
        tf.layers.Dense(
            units=64,
            activation='tanh',
            kernel_initializer=tf.orthogonal_initializer()),
        tf.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.orthogonal_initializer())
    ])

  def call(self, inputs):
    return self.main(inputs)


class CriticDDPG(tf.keras.Model):
  """Implementation of a critic base network."""

  def __init__(self, input_dim):
    """Initializes a policy network.

    Args:
      input_dim: size of the input space
    """
    super(CriticDDPG, self).__init__()

    self.main = tf.keras.Sequential([
        tf.layers.Dense(
            units=400,
            input_shape=(input_dim,),
            activation='relu',
            kernel_initializer=tf.orthogonal_initializer()),
        tf.layers.Dense(
            units=300,
            activation='relu',
            kernel_initializer=tf.orthogonal_initializer()),
        tf.layers.Dense(
            units=1, kernel_initializer=tf.orthogonal_initializer())
    ])

  def call(self, inputs, actions):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.

    Returns:
      Values of observations.
    """
    x = tf.concat([inputs, actions], -1)
    return self.main(x)

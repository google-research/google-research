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

"""Actor implementation."""

import typing
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs import tensor_spec

tfd = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(tf.keras.Model):
  """Gaussian policy with TanH squashing."""

  def __init__(self, state_dim,
               action_spec):
    """Creates an actor.

    Args:
      state_dim: State size.
      action_spec: Action spec.
    """
    super().__init__()
    self.trunk = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256,
            input_shape=(state_dim,),
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            256, activation=tf.nn.relu, kernel_initializer='orthogonal'),
        tf.keras.layers.Dense(
            2 * action_spec.shape[0], kernel_initializer='orthogonal')
    ])
    self.action_mean = tf.constant(
        (action_spec.maximum + action_spec.minimum) / 2.0, dtype=tf.float32)
    self.action_scale = tf.constant(
        (action_spec.maximum - action_spec.minimum) / 2.0, dtype=tf.float32)

  def get_dist_and_mode(
      self,
      states,
      std = None
  ):
    """Returns a tf.Distribution for given states modes of this distribution.

    Args:
      states: A batch of states.
      std: A fixed std to use, if not provided use from the network.
    """
    out = self.trunk(states)
    mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
    mode = tf.nn.tanh(mu)

    log_std = tf.nn.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    if std is None:
      std = tf.exp(log_std)
    else:
      # Ugly hack
      std = tf.stop_gradient(mu) * 0.0 + std

    dist = tfd.TransformedDistribution(
        tfd.Sample(
            tfd.Normal(tf.zeros(mu.shape[:-1]), 1.0),
            sample_shape=mu.shape[-1:]),
        tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self.action_mean),
            tfp.bijectors.Scale(scale=self.action_scale),
            tfp.bijectors.Tanh(),
            tfp.bijectors.Shift(shift=mu),
            tfp.bijectors.ScaleMatvecDiag(scale_diag=std)
        ]))

    return dist, mode

  @tf.function
  def get_log_prob(self, states, actions,
                   std = None):
    """Evaluate log probs for actions conditined on states.

    Args:
      states: A batch of states.
      actions: A batch of actions to evaluate log probs on.
      std: A fixed std to use, if not provided use from the network.

    Returns:
      Log probabilities of actions.
    """
    dist, _ = self.get_dist_and_mode(states, std)
    log_probs = dist.log_prob(actions)
    return log_probs

  @tf.function
  def call(
      self,
      states,
      std = None
  ):
    """Computes actions for given inputs.

    Args:
      states: A batch of states.
      std: A fixed std to use, if not provided use from the network.
    Returns:
      A mode action, a sampled action and log probability of the sampled action.
    """
    dist, mode = self.get_dist_and_mode(states, std)
    samples = dist.sample()
    log_probs = dist.log_prob(samples)
    return mode, samples, log_probs

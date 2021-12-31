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

import typing

from dm_env import specs as dm_env_specs
import tensorflow as tf
import tensorflow_probability as tfp

from rl_repr.batch_rl import keras_utils

tfd = tfp.distributions


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class BasePolicy(tf.keras.Model):
  """Base class for policies."""

  def __init__(self,
               state_dim,
               action_dim,
               action_spec,
               hidden_dims = (256, 256),
               eps = 1e-3):
    """Creates an actor.

    Args:
      state_dim: State size.
      action_dim: Actiom size.
      action_spec: Action spec.
      hidden_dims: List of hidden dimensions.
      eps: Epsilon for numerical stability.
    """
    super().__init__()

    self.trunk = keras_utils.create_mlp(state_dim, action_dim,
                                        hidden_dims=hidden_dims)

    self.action_spec = action_spec
    self.action_mean = tf.constant(
        (action_spec.maximum + action_spec.minimum) / 2.0, dtype=tf.float32)
    self.action_scale = tf.constant(
        (action_spec.maximum - action_spec.minimum) / 2.0, dtype=tf.float32)
    self.eps = eps


class MixtureGuassianPolicy(BasePolicy):
  """Gaussian policy with TanH squashing."""

  def __init__(self, state_dim,
               action_spec,
               hidden_dims = (256, 256),
               num_components = 5):
    super().__init__(
        state_dim,
        num_components * action_spec.shape[0] * 3,
        action_spec,
        hidden_dims=hidden_dims)
    self._num_components = num_components

  def _get_dist_and_mode(
      self,
      states,
      out = None,
      stddev = 1.0):
    """Returns a tf.Distribution for given states modes of this distribution.

    Args:
      states: Batch of states.
      out: Batch of neural net outputs.
      stddev: Standard deviation of sampling distribution.
    """
    if out is None:
      out = self.trunk(states)
    logits, mu, log_std = tf.split(out, num_or_size_splits=3, axis=1)

    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.math.exp(log_std)

    shape = [std.shape[0], -1, self._num_components]
    logits = tf.reshape(logits, shape)
    mu = tf.reshape(mu, shape)
    std = tf.reshape(std, shape)

    components_distribution = tfd.Normal(loc=mu, scale=std)

    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_distribution)

    return tfd.TransformedDistribution(
        tfd.Independent(distribution),
        tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self.action_mean),
            tfp.bijectors.Scale(scale=self.action_scale),
            tfp.bijectors.Tanh(),
        ]))

  @tf.function
  def call(
      self,
      states,
      out = None,
      sample = False,
      with_log_probs = False
  ):
    """Computes actions for given inputs.

    Args:
      states: Batch of states.
      out: Batch of neural net outputs.
      sample: Whether to sample actions.
      with_log_probs: Whether to return log probability of sampled actions.

    Returns:
      Sampled actions.
    """
    if sample:
      dist = self._get_dist_and_mode(states, out)
    else:
      dist = self._get_dist_and_mode(states, out, stddev=0.0)
    actions = dist.sample()

    if with_log_probs:
      return actions, dist.log_prob(actions)
    else:
      return actions

  @tf.function
  def log_probs(self,
                states,
                actions,
                out = None):
    actions = tf.clip_by_value(actions, self.action_spec.minimum + self.eps,
                               self.action_spec.maximum - self.eps)
    dist = self._get_dist_and_mode(states, out)
    return dist.log_prob(actions)


class DiagGuassianPolicy(BasePolicy):
  """Gaussian policy with TanH squashing."""

  def __init__(self,
               state_dim,
               action_spec,
               hidden_dims = (256, 256),
               apply_tanh_squash = True):
    super().__init__(state_dim, action_spec.shape[0] * 2, action_spec,
                     hidden_dims=hidden_dims)
    self.apply_tanh_squash = apply_tanh_squash

  def _get_dist_and_mode(
      self,
      states,
      out = None,
      stddev = 1.0):
    """Returns a tf.Distribution for given states modes of this distribution.

    Args:
      states: Batch of states.
      out: Batch of neural net outputs.
      stddev: Standard deviation of sampling distribution.
    """
    if out is None:
      out = self.trunk(states)
    mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)

    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)

    dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std * stddev)
    if self.apply_tanh_squash:
      dist = tfd.TransformedDistribution(
          dist,
          tfp.bijectors.Chain([
              tfp.bijectors.Shift(shift=self.action_mean),
              tfp.bijectors.Scale(scale=self.action_scale),
              tfp.bijectors.Tanh(),
          ]))

    return dist

  @tf.function
  def call(
      self,
      states,
      out = None,
      sample = False,
      with_log_probs = False
  ):
    """Computes actions for given inputs.

    Args:
      states: Batch of states.
      out: Batch of neural net outputs.
      sample: Whether to sample actions.
      with_log_probs: Whether to return log probability of sampled actions.

    Returns:
      Sampled actions.
    """
    if sample:
      dist = self._get_dist_and_mode(states, out)
    else:
      dist = self._get_dist_and_mode(states, out, stddev=0.0)
    actions = dist.sample()

    if with_log_probs:
      return actions, dist.log_prob(actions)
    else:
      return actions

  @tf.function
  def log_probs(self,
                states,
                actions,
                out = None):
    actions = tf.clip_by_value(actions, self.action_spec.minimum + self.eps,
                               self.action_spec.maximum - self.eps)
    dist = self._get_dist_and_mode(states, out)
    return dist.log_prob(actions)


class DeterministicPolicy(BasePolicy):
  """Deterministic policy with TanH squashing."""

  def __init__(self, state_dim, action_spec,
               stddev):
    """Creates a deterministic policy.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      stddev: Noise scale.
    """
    super().__init__(state_dim, action_spec.shape[0], action_spec)
    self._noise = tfd.Normal(loc=0.0, scale=stddev)

  @tf.function
  def call(
      self,
      states,
      sample = False
  ):
    """Computes actions for given inputs.

    Args:
      states: Batch of states.
      sample: Whether to sample actions.

    Returns:
      Mode actions, sampled actions.
    """
    actions = tf.nn.tanh(self.trunk(states))
    if sample:
      actions = actions + self._noise.sample(actions.shape)
      actions = tf.clip_by_value(actions, -1.0, 1.0)

    return (actions + self.action_mean) * self.action_scale


class CVAEPolicy(BasePolicy):
  """Conditional variational autoencoder."""

  def __init__(self, state_dim, action_spec, latent_dim):
    """Creates an actor.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      latent_dim: Size of latent space.
    """
    action_dim = action_spec.shape[0]
    super().__init__(state_dim, action_dim, action_spec)
    del self.trunk
    del self.eps
    self.latent_dim = latent_dim

    relu_gain = tf.math.sqrt(2.0)
    relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)

    self.encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(
            750,
            input_dim=state_dim + action_dim,
            activation='relu',
            kernel_initializer=relu_orthogonal),
        tf.keras.layers.Dense(
            750, activation='relu', kernel_initializer=relu_orthogonal),
        tf.keras.layers.Dense(
            latent_dim + latent_dim, kernel_initializer='orthogonal'),
    ])

    self.decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(
            750,
            input_dim=state_dim + latent_dim,
            activation='relu',
            kernel_initializer=relu_orthogonal),
        tf.keras.layers.Dense(
            750, activation='relu', kernel_initializer=relu_orthogonal),
        tf.keras.layers.Dense(action_dim, kernel_initializer='orthogonal'),
    ])

  @tf.function
  def sample(self, states):
    eps = tf.random.normal(shape=(states.shape[0], self.latent_dim))
    return self.decode(states, eps)

  def encode(self, states, actions):
    inputs = tf.concat([states, actions], -1)
    mean, logvar = tf.split(self.encoder(inputs),
                            num_or_size_splits=2, axis=1)
    logvar = tf.clip_by_value(logvar, -4, 15)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean

  def decode(self, states, z):
    inputs = tf.concat([states, z], -1)
    outputs = self.decoder(inputs)
    outputs = tf.tanh(outputs)
    return (outputs + self.action_mean) * self.action_scale

  @tf.function
  def call(self,
           states,
           sample = True):
    """Computes actions for given inputs.

    Args:
      states: Batch of states.
      sample: Whether to sample actions.

    Returns:
      Mode actions, sampled actions.
    """
    assert sample, 'CVAE cannot be called without sampling'
    return self.sample(states)

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
"""Implementation of twin_sac, a mix of TD3 (https://arxiv.org/abs/1802.09477) and SAC (https://arxiv.org/abs/1801.01290, https://arxiv.org/abs/1812.05905).

Overall structure and hyperparameters are taken from TD3. However, the algorithm
itself represents a version of SAC.
"""

import typing

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs.tensor_spec import BoundedTensorSpec

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
               eps = 1e-6):
    """Creates an actor.

    Args:
      state_dim: State size.
      action_dim: Actiom size.
      action_spec: Action spec.
      hidden_dims: List of hidden dimensions.
      eps: Epsilon for numerical stability.
    """
    super().__init__()

    relu_gain = tf.math.sqrt(2.0)
    relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
    near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

    layers = []
    for hidden_dim in hidden_dims:
      layers.append(
          tf.keras.layers.Dense(
              hidden_dim,
              activation=tf.nn.relu,
              kernel_initializer=relu_orthogonal))

    inputs = tf.keras.Input(shape=(state_dim,))
    outputs = tf.keras.Sequential(
        layers + [tf.keras.layers.Dense(
            action_dim, kernel_initializer=near_zero_orthogonal)]
        )(inputs)

    self.trunk = tf.keras.Model(inputs=inputs, outputs=outputs)

    self.action_spec = action_spec
    self.action_mean = tf.constant(
        (action_spec.maximum + action_spec.minimum) / 2.0, dtype=tf.float32)
    self.action_scale = tf.constant(
        (action_spec.maximum - action_spec.minimum) / 2.0, dtype=tf.float32)
    self.eps = eps


class MixtureGuassianPolicy(BasePolicy):
  """Gaussian policy with TanH squashing."""

  def __init__(self,
               state_dim,
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
      stddev = 1.0):
    """Returns a tf.Distribution for given states modes of this distribution.

    Args:
      states: Batch of states.
      stddev: Standard deviation of sampling distribution.
    """
    out = self.trunk(states)
    logits, mu, log_std = tf.split(out, num_or_size_splits=3, axis=1)

    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)

    shape = [std.shape[0], -1, self._num_components]
    logits = tf.reshape(logits, shape)
    mu = tf.reshape(mu, shape)
    std = tf.reshape(std, shape)

    components_distribution = tfd.TransformedDistribution(
        tfd.Normal(loc=mu, scale=std),
        tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self.action_mean),
            tfp.bijectors.Scale(scale=self.action_scale),
            tfp.bijectors.Tanh(),
        ]))

    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_distribution)

    return tfd.Independent(distribution)

  @tf.function
  def call(
      self,
      states,
      sample = False,
      with_log_probs = False
  ):
    """Computes actions for given inputs.

    Args:
      states: Batch of states.
      sample: Whether to sample actions.
      with_log_probs: Whether to return log probability of sampled actions.

    Returns:
      Sampled actions.
    """
    if sample:
      dist = self._get_dist_and_mode(states)
    else:
      dist = self._get_dist_and_mode(states, stddev=0.0)
    actions = dist.sample()

    if with_log_probs:
      return actions, dist.log_prob(actions)
    else:
      return actions

  @tf.function
  def log_probs(
      self,
      states,
      actions,
      out = None,
      with_entropy = False
  ):
    actions = tf.clip_by_value(actions, self.action_spec.minimum + self.eps,
                               self.action_spec.maximum - self.eps)
    dist = self._get_dist_and_mode(states, out)

    sampled_actions = dist.sample()
    sampled_actions = tf.clip_by_value(sampled_actions,
                                       self.action_spec.minimum + self.eps,
                                       self.action_spec.maximum - self.eps)
    if with_entropy:
      return dist.log_prob(actions), -dist.log_prob(sampled_actions)
    else:
      return dist.log_prob(actions)


class DiagGuassianPolicy(BasePolicy):
  """Gaussian policy with TanH squashing."""

  def __init__(self,
               state_dim,
               action_spec,
               hidden_dims = (256, 256)):
    super().__init__(state_dim, action_spec.shape[0] * 2, action_spec,
                     hidden_dims=hidden_dims)

  def _get_dist_and_mode(
      self,
      states,
      stddev = 1.0):
    """Returns a tf.Distribution for given states modes of this distribution.

    Args:
      states: Batch of states.
      stddev: Standard deviation of sampling distribution.
    """
    out = self.trunk(states)
    mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)

    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)

    dist = tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=mu, scale_diag=std * stddev),
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
      sample = False,
      with_log_probs = False
  ):
    """Computes actions for given inputs.

    Args:
      states: Batch of states.
      sample: Whether to sample actions.
      with_log_probs: Whether to return log probability of sampled actions.

    Returns:
      Sampled actions.
    """
    if sample:
      dist = self._get_dist_and_mode(states)
    else:
      dist = self._get_dist_and_mode(states, stddev=0.0)
    actions = dist.sample()

    if with_log_probs:
      return actions, dist.log_prob(actions)
    else:
      return actions

  @tf.function
  def log_probs(
      self,
      states,
      actions,
      with_entropy = False
  ):
    actions = tf.clip_by_value(actions, self.action_spec.minimum + self.eps,
                               self.action_spec.maximum - self.eps)
    dist = self._get_dist_and_mode(states)

    sampled_actions = dist.sample()
    sampled_actions = tf.clip_by_value(sampled_actions,
                                       self.action_spec.minimum + self.eps,
                                       self.action_spec.maximum - self.eps)
    if with_entropy:
      return dist.log_prob(actions), -dist.log_prob(sampled_actions)
    else:
      return dist.log_prob(actions)

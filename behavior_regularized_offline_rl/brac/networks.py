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

"""Neural network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 0


def get_spec_means_mags(spec):
  means = (spec.maximum + spec.minimum) / 2.0
  mags = (spec.maximum - spec.minimum) / 2.0
  means = tf.constant(means, dtype=tf.float32)
  mags = tf.constant(mags, dtype=tf.float32)
  return means, mags


class ActorNetwork(tf.Module):
  """Actor network."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      ):
    super(ActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0] * 2,
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state):
    h = state
    for l in self._layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    a_distribution = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0.0, scale=1.0),
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.AffineScalar(shift=self._action_means,
                                       scale=self._action_mags),
            tfp.bijectors.Tanh(),
            tfp.bijectors.AffineScalar(shift=mean, scale=std),
        ]),
        event_shape=[mean.shape[-1]],
        batch_shape=[mean.shape[0]])
    return a_distribution, a_tanh_mode

  def get_log_density(self, state, action):
    a_dist, _ = self._get_outputs(state)
    log_density = a_dist.log_prob(action)
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample_n(self, state, n=1):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample(n)
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample(self, state):
    return self.sample_n(state, n=1)[1][0]


class CriticNetwork(tf.Module):
  """Critic Network."""

  def __init__(
      self,
      fc_layer_params=(),
      ):
    super(CriticNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class BCQActorNetwork(tf.Module):
  """Actor network for BCQ."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      max_perturbation=0.05,
      ):
    super(BCQActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._max_perturbation = max_perturbation
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0],
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags * self._max_perturbation + action
    a = tf.clip_by_value(
        a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state, action):
    return self._get_outputs(state, action)


class BCQVAENetwork(tf.Module):
  """VAE for learned behavior policy used by BCQ."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      latent_dim=None,
      ):
    super(BCQVAENetwork, self).__init__()
    if latent_dim is None:
      latent_dim = action_spec.shape[0] * 2
    self._action_spec = action_spec
    self._encoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._encoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        latent_dim * 2,
        activation=None)
    self._encoder_layers.append(output_layer)
    self._decoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._decoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        action_spec.shape[0],
        activation=None)
    self._decoder_layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)
    self._latent_dim = latent_dim

  @property
  def action_spec(self):
    return self._action_spec

  def forward(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._encoder_layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    std = tf.exp(tf.clip_by_value(log_std, -4, 15))
    z = mean + std * tf.random.normal(shape=std.shape)
    a = self.decode(state, z)
    return a, mean, std

  def decode(self, state, z):
    h = tf.concat([state, z], axis=-1)
    for l in self._decoder_layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags + self._action_means
    # a = tf.clip_by_value(
    #     a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  def sample(self, state):
    z = tf.random.normal(shape=[state.shape[0], self._latent_dim])
    z = tf.clip_by_value(z, -0.5, 0.5)
    return self.decode(state, z)

  def get_log_density(self, state, action):
    # variational lower bound
    a_recon, mean, std = self._p_fn.forward(state, action)
    log_2pi = tf.log(tf.constant(math.pi))
    recon = - 0.5 * tf.reduce_sum(
        tf.square(a_recon - action) + log_2pi, axis=-1)
    kl = 0.5 * tf.reduce_sum(
        - 1.0 - tf.log(tf.square(std)) + tf.square(mean) + tf.square(std),
        axis=-1)
    return recon - kl

  @property
  def weights(self):
    w_list = []
    for l in self._encoder_layers:
      w_list.append(l.weights[0])
    for l in self._decoder_layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state, action):
    return self._get_outputs(state, action)

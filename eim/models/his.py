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

"""Hamiltonian Importance Sampling distribution."""

from __future__ import absolute_import
import functools
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from eim.models import base

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class HIS(object):
  """Hamiltonian Importance Sampling distribution base class."""

  def __init__(self,
               T,
               data_dim,
               energy_fn,
               q_fn,
               proposal=None,
               init_alpha=1.,
               init_step_size=0.01,
               learn_temps=False,
               learn_stepsize=False,
               dtype=tf.float32,
               name="his"):
    self.timesteps = T
    self.data_dim = data_dim
    self.energy_fn = energy_fn
    self.q = q_fn

    init_alpha = -np.log(1. / init_alpha - 1. + 1e-4)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      self.raw_alphas = []
      for t in range(T):
        self.raw_alphas.append(tf.get_variable(
            name="raw_alpha_%d" % t,
            shape=[],
            dtype=dtype,
            initializer=tf.constant_initializer(init_alpha),
            trainable=learn_temps))
      self.log_alphas = [
          -tf.nn.softplus(-raw_alpha) for raw_alpha in self.raw_alphas
      ]
      self.log_alphas = [-tf.reduce_sum(self.log_alphas)] + self.log_alphas
      init_step_size = np.log(np.exp(init_step_size) - 1.)
      self.raw_step_size = tf.get_variable(
          name="raw_step_size",
          shape=data_dim,
          dtype=tf.float32,
          initializer=tf.constant_initializer(init_step_size),
          trainable=learn_stepsize)
      self.step_size = tf.math.softplus(self.raw_step_size)
      tf.summary.scalar("his_step_size", tf.reduce_mean(self.step_size))
      _ = [
          tf.summary.scalar("his_alpha/alpha_%d" % t,
                            tf.exp(self.log_alphas[t]))
          for t in range(len(self.log_alphas))
      ]

    if proposal is None:
      self.proposal = base.get_independent_normal(data_dim)
    else:
      self.proposal = proposal
    self.momentum_proposal = base.get_independent_normal(data_dim)

  def hamiltonian_potential(self, x):
    return tf.squeeze(self.energy_fn(x), axis=-1)

  def _grad_hamiltonian_potential(self, x):
    potential = self.hamiltonian_potential(x)
    return tf.gradients(potential, x)[0]

  def _hamiltonian_dynamics(self, x, momentum, alphas=None):
    """Deterministic leapfrog integrator."""
    if alphas is None:
      alphas = [tf.exp(log_alpha) for log_alpha in self.log_alphas]

    momentum *= alphas[0]
    grad_energy = self._grad_hamiltonian_potential(x)
    for t in range(1, self.timesteps + 1):
      momentum -= self.step_size / 2. * grad_energy
      x += self.step_size * momentum
      grad_energy = self._grad_hamiltonian_potential(x)
      momentum -= self.step_size / 2. * grad_energy
      momentum *= alphas[t]
    return x, momentum

  def _reverse_hamiltonian_dynamics(self, x, momentum):
    alphas = [tf.exp(-log_alpha) for log_alpha in self.log_alphas]
    alphas.reverse()
    x, momentum = self._hamiltonian_dynamics(x, -momentum, alphas)
    return x, -momentum

  def log_prob(self, x_final, num_samples=1):
    """Compute log probability lower bound on x_final.

    Args:
      x_final: [batch_size] + data_dim tensor.
      num_samples: Optional number of samples to compute bounds.
    Returns:
      log probability lower bound.
    """
    tiled_x_final = tf.tile(x_final, [num_samples] + [1] * len(self.data_dim))
    q = self.q(tiled_x_final)
    rho_final = q.sample()  # [num_samples * batch_size, data_dim]

    x_0, rho_0 = self._reverse_hamiltonian_dynamics(tiled_x_final, rho_final)
    elbo = (
        self.proposal.log_prob(x_0) + self.momentum_proposal.log_prob(rho_0) -
        q.log_prob(rho_final))
    iwae = (tf.reduce_logsumexp(tf.reshape(elbo, [num_samples, -1]), axis=0)
            - tf.log(tf.to_float(num_samples)))
    return iwae

  def sample(self, num_samples=1):
    """Draw a sample from the model."""
    x_0 = self.proposal.sample(num_samples)
    rho_0 = self.momentum_proposal.sample(num_samples)
    x_final, _ = self._hamiltonian_dynamics(x_0, rho_0)

    # Compute summaries
    initial_potential = self.hamiltonian_potential(x_0)
    final_potential = self.hamiltonian_potential(x_final)
    tf.summary.histogram("initial_potential", initial_potential)
    tf.summary.histogram("diff_potential", final_potential - initial_potential)

    return x_final


class FullyConnectedHIS(HIS):
  """HIS with fully connected networks."""

  def __init__(self,
               T,
               data_dim,
               energy_hidden_sizes,
               q_hidden_sizes,
               data_mean=None,
               proposal=None,
               init_alpha=1.,
               init_step_size=0.01,
               learn_temps=False,
               learn_stepsize=False,
               scale_min=1e-5,
               dtype=tf.float32,
               name="fully_connected_his"):
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    energy_fn = functools.partial(
        base.mlp,
        layer_sizes=energy_hidden_sizes + [1],
        final_activation=None,
        name="%s/energy_fn_mlp" % name)
    q_fn = functools.partial(
        base.conditional_normal,
        data_dim=data_dim,
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        bias_init=None,
        truncate=False,
        name="%s/q" % name)
    super(FullyConnectedHIS,
          self).__init__(T, data_dim, energy_fn, q_fn, proposal, init_alpha,
                         init_step_size, learn_temps, learn_stepsize, dtype,
                         name)


# class ConvHIS(HIS):
#   """HIS with conv networks."""
#
#   def __init__(self,
#                T,
#                data_dim,
#                q_hidden_sizes,
#                proposal=None,
#                init_alpha=1.,
#                init_step_size=0.01,
#                learn_temps=False,
#                learn_stepsize=False,
#                scale_min=1e-5,
#                dtype=tf.float32,
#                name="fully_connected_his"):
#
#     q_fn = functools.partial(
#         base.conditional_normal,
#         data_dim=data_dim,
#         hidden_sizes=q_hidden_sizes,
#         scale_min=scale_min,
#         bias_init=None,
#         truncate=False,
#         name="%s/q" % name)
#
#     conv = functools.partial(
#         tf.keras.layers.Conv2D, padding="SAME", activation=activation)
#     energy_fn = tf.keras.Sequential([
#         conv(base_depth, 5, 2),
#         conv(base_depth, 5, 2),
#         conv(2 * base_depth, 5, 2),
#         conv(2 * base_depth, 5, 2),
#         conv(4 * base_depth, 5, 2),
#         conv(4, 5, 2),
#         tfkl.Flatten(),
#         tfkl.Dense(1, activation=None),
#     ])
#
#     super(FullyConnectedHIS,
#           self).__init__(T, data_dim, energy_fn, q_fn, proposal, init_alpha,
#                          init_step_size, learn_temps, learn_stepsize, dtype,
#                          name)


# Copied implementation of HIS, will abstract out relevant components
# afterwards.
# class HISVAE(object):
#
#  def __init__(self,
#               T,
#               latent_dim,
#               data_dim,
#               energy_hidden_sizes,
#               q_hidden_sizes,
#               decoder_hidden_sizes,
#               proposal=None,
#               data_mean=None,
#               init_alpha=1.,
#               init_step_size=0.01,
#               learn_temps=False,
#               learn_stepsize=False,
#               scale_min=1e-5,
#               squash=False,
#               squash_eps=1e-6,
#               decoder_nn_scale=False,
#               dtype=tf.float32,
#               kl_weight=1.,
#               name="hisvae"):
#    self.kl_weight = kl_weight
#    if squash:
#      bijectors = [
#          tfp.bijectors.AffineScalar(scale=256.),
#          tfp.bijectors.AffineScalar(
#              shift=-squash_eps / 2., scale=(1. + squash_eps)),
#          tfp.bijectors.Sigmoid(),
#      ]
#      self.squash = tfp.bijectors.Chain(bijectors)
#    else:
#      self.squash = None
#
#    self.latent_dim = latent_dim
#    self.data_dim = data_dim
#    if data_mean is not None:
#      self.data_mean = data_mean
#
#      if squash:
#        self.unsquashed_data_mean = self.squash.inverse(data_mean)
#    else:
#      self.data_mean = tf.zeros((), dtype=dtype)
#    self.timesteps = T
#    self.energy_fn = functools.partial(
#        base.mlp,
#        layer_sizes=energy_hidden_sizes + [1],
#        final_activation=None,
#        name="%s/energy_fn_mlp" % name)
#    self.q_rho = functools.partial(
#        base.conditional_normal,
#        data_dim=data_dim,
#        hidden_sizes=q_hidden_sizes,
#        scale_min=scale_min,
#        bias_init=None,
#        truncate=False,
#        squash=False,
#        name="%s/q_rho" % name)
#    self.q_z = functools.partial(
#        base.conditional_normal,
#        data_dim=latent_dim,
#        hidden_sizes=q_hidden_sizes,
#        scale_min=scale_min,
#        bias_init=None,
#        truncate=False,
#        squash=False,
#        name="%s/q_z" % name)
#    self.decoder = functools.partial(
#        base.conditional_normal,
#        data_dim=data_dim,
#        hidden_sizes=decoder_hidden_sizes,
#        scale_min=scale_min,
#        nn_scale=decoder_nn_scale,
#        truncate=False,
#        squash=False,
#        name="%s/decoder" % name)
#
#    eps = 0.0001
#    init_alpha = -np.log(1. / init_alpha - 1. + eps)
#    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#      self.raw_alphas = [
#          tf.get_variable(
#              name="raw_alpha_%d" % t,
#              shape=[],
#              dtype=tf.float32,
#              initializer=tf.constant_initializer(init_alpha),
#              trainable=learn_temps) for t in range(T)
#      ]
#      self.log_alphas = [
#          -tf.nn.softplus(-raw_alpha) for raw_alpha in self.raw_alphas
#      ]
#      self.log_alphas = [-tf.reduce_sum(self.log_alphas)] + self.log_alphas
#      init_step_size = np.log(np.exp(init_step_size) - 1.)
#      self.raw_step_size = tf.get_variable(
#          name="raw_step_size",
#          shape=[data_dim],
#          dtype=tf.float32,
#          initializer=tf.constant_initializer(init_step_size),
#          trainable=learn_stepsize)
#      self.step_size = tf.math.softplus(self.raw_step_size)
#      tf.summary.scalar("his_step_size", tf.reduce_mean(self.step_size))
#      tf.summary.histogram("his_step_size", self.step_size)
#      [
#          tf.summary.scalar("his_alpha/alpha_%d" % t,
#                            tf.exp(self.log_alphas[t]))
#          for t in range(len(self.log_alphas))
#      ]
#
#    if proposal is None:
#      self.proposal = tfd.MultivariateNormalDiag(
#          loc=tf.zeros([latent_dim], dtype=dtype),
#          scale_diag=tf.ones([latent_dim], dtype=dtype))
#    else:
#      self.proposal = proposal
#    self.momentum_proposal = tfd.MultivariateNormalDiag(
#        loc=tf.zeros([data_dim], dtype=dtype),
#        scale_diag=tf.ones([data_dim], dtype=dtype))
#
#  def hamiltonian_potential(self, x):
#    return tf.squeeze(self.energy_fn(x), axis=-1)
#
#  def _grad_hamiltonian_potential(self, x):
#    potential = self.hamiltonian_potential(x)
#    return tf.gradients(potential, x)[0]
#
#  def _hamiltonian_dynamics(self, x, momentum, alphas=None):
#    if alphas is None:
#      alphas = [tf.exp(log_alpha) for log_alpha in self.log_alphas]
#
#    momentum *= alphas[0]
#    grad_energy = self._grad_hamiltonian_potential(x)
#    for t in range(1, self.timesteps + 1):
#      momentum -= self.step_size / 2. * grad_energy
#      x += self.step_size * momentum
#      grad_energy = self._grad_hamiltonian_potential(x)
#      momentum -= self.step_size / 2. * grad_energy
#      momentum *= alphas[t]
#    return x, momentum
#
#  def _reverse_hamiltonian_dynamics(self, x, momentum):
#    alphas = [tf.exp(-log_alpha) for log_alpha in self.log_alphas]
#    alphas.reverse()
#    x, momentum = self._hamiltonian_dynamics(x, -momentum, alphas)
#    return x, -momentum
#
#  def log_prob(self, data, num_samples=1):
#    batch_shape = tf.shape(data)[0:-1]
#    reshaped_data = tf.reshape(
#        data, [tf.math.reduce_prod(batch_shape), self.data_dim])
#    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
#    log_prob = tf.reshape(log_prob, batch_shape)
#    return log_prob
#
#  def _log_prob(self, data, num_samples=1):
#    if self.squash is not None:
#      x_final = self.squash.inverse(data)
#      x_final -= self.unsquashed_data_mean
#    else:
#      x_final = data
#
#    q_rho = self.q_rho(data)  # Maybe better to work w/ the untransformed data?
#    rho_final = q_rho.sample([num_samples])
#    x_final = tf.tile(x_final[tf.newaxis, :, :], [num_samples, 1, 1])
#    x_0, rho_0 = self._reverse_hamiltonian_dynamics(x_final, rho_final)
#    q_z = self.q_z(x_0)
#    z = q_z.sample()
#    p_x_given_z = self.decoder(z)
#
#    elbo = (
#        p_x_given_z.log_prob(x_0) + self.momentum_proposal.log_prob(rho_0) +
#        self.kl_weight * (self.proposal.log_prob(z) - q_z.log_prob(z)) -
#        q_rho.log_prob(rho_final))
#    if self.squash is not None:
#      elbo += tf.tile(
#          self.squash.inverse_log_det_jacobian(data, event_ndims=1)[None, :],
#          [num_samples, 1])
#    return tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(num_samples))
#
#  def sample(self, sample_shape=[1]):
#    z = self.proposal.sample(sample_shape)
#    p_x_given_z = self.decoder(z)
#    x_0 = p_x_given_z.sample()
#    rho_0 = self.momentum_proposal.sample(sample_shape=sample_shape)
#    x_final, _ = self._hamiltonian_dynamics(x_0, rho_0)
#
#    #    initial_potential = self.hamiltonian_potential(x_0, z, p_x_given_z)
#    #    final_potential = self.hamiltonian_potential(x_final, z, p_x_given_z)
#    #    tf.summary.histogram("initial_potential", initial_potential)
#    #    tf.summary.histogram("diff_potential",
# final_potential - initial_potential)
#    #    final_energy = tf.squeeze(
# self.energy_fn(tf.concat([x_final, z], axis=-1)), axis=-1)
#    #    tf.summary.histogram("final_energy", final_energy)
#
#    if self.squash is not None:
#      x_final += self.unsquashed_data_mean
#      x_final = self.squash.forward(x_final)
#    return x_final
#

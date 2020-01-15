# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Self-normalized Importance Sampling distribution."""

from __future__ import absolute_import
import functools
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from eim.models import base

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

deconv = functools.partial(tf.keras.layers.Conv2DTranspose, padding="SAME")
conv = functools.partial(tf.keras.layers.Conv2D, padding="SAME")


class AbstractNIS(base.ProbabilisticModel):
  """Self-normalized Importance Sampling distribution."""

  def __init__(self,  # pylint: disable=invalid-name
               K,
               data_dim,
               energy_fn,
               proposal=None,
               data_mean=None,
               reparameterize_proposal_samples=True,
               dtype=tf.float32):
    """Creates a NIS model.

    Args:
      K: The number of proposal samples to take.
      data_dim: The dimension of the data.
      energy_fn: Energy function.
      proposal: A distribution over the data space of this model. Must support
        sample() and log_prob() although log_prob only needs to return a lower
        bound on the true log probability. If not supplied, then defaults to
        Gaussian.
      data_mean: Mean of the data used to center the input.
      reparameterize_proposal_samples: Whether to allow gradients to pass
        through the proposal samples.
      dtype: Type of the tensors.
    """
    self.K = K   # pylint: disable=invalid-name
    self.data_dim = data_dim  # self.data_dim is always a list
    self.reparameterize_proposal_samples = reparameterize_proposal_samples
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.energy_fn = energy_fn
    if proposal is None:
      self.proposal = base.get_independent_normal(self.data_dim)
    else:
      self.proposal = proposal

  def _log_prob(self, data, num_samples=1):
    """Compute a lower bound on the log likelihood."""
    # Due to memory issues, we need to use num_samples=1 here
    num_samples, proposal_num_samples = 1, num_samples
    batch_size = tf.shape(data)[0]
    # Sample from the proposal and compute the weighs of the "unseen" samples.
    # We share these across the batch dimension.
    # [num_samples, K, data_size]
    proposal_samples = self.proposal.sample(num_samples * (self.K - 1))
    if not self.reparameterize_proposal_samples:
      proposal_samples = tf.stop_gradient(proposal_samples)

    # [num_samples, K]
    log_energy_proposal = tf.reshape(
        self.energy_fn(tf.reshape(proposal_samples, [-1] + self.data_dim)),
        [num_samples, self.K - 1])
    tf.summary.histogram("log_energy_proposal", log_energy_proposal)
    tf.summary.scalar("min_log_energy_proposal",
                      tf.reduce_min(log_energy_proposal))
    tf.summary.scalar("max_log_energy_proposal",
                      tf.reduce_max(log_energy_proposal))
    # [num_samples]
    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1)

    # [batch_size, num_samples]
    tiled_proposal_lse = tf.tile(proposal_lse[tf.newaxis, :], [batch_size, 1])

    # Compute the weights of the observed data.
    # [batch_size, 1]
    log_energy_data = tf.reshape(self.energy_fn(data), [batch_size])
    tf.summary.histogram("log_energy_data", log_energy_data)
    tf.summary.scalar("min_log_energy_data", tf.reduce_min(log_energy_data))
    tf.summary.scalar("max_log_energy_data", tf.reduce_max(log_energy_data))

    # [batch_size, num_samples]
    tiled_log_energy_data = tf.tile(log_energy_data[:, tf.newaxis],
                                    [1, num_samples])

    # Add the weights of the proposal samples with the true data weights.
    # [batch_size, num_samples]
    # pylint: disable=invalid-name
    Z_hat = tf.reduce_logsumexp(
        tf.stack([tiled_log_energy_data, tiled_proposal_lse], axis=-1), axis=-1)
    Z_hat -= tf.log(tf.to_float(self.K))
    # Perform the log-sum-exp reduction for IWAE
    # [batch_size]
    Z_hat = tf.reduce_logsumexp(
        Z_hat, axis=1) - tf.log(tf.to_float(num_samples))
    # pylint: enable=invalid-name

    try:
      # Try giving the proposal lower bound num_samples if it can use it.
      proposal_lp = self.proposal.log_prob(data,
                                           num_samples=proposal_num_samples)
    except TypeError:
      proposal_lp = self.proposal.log_prob(data)
    lower_bound = proposal_lp + log_energy_data - Z_hat
    return lower_bound

  def sample(self, num_samples=1):
    """Sample from the model."""
    flat_proposal_samples = self.proposal.sample(num_samples * self.K)
    proposal_samples = tf.reshape(flat_proposal_samples,
                                  [num_samples, self.K] + self.data_dim)
    log_energy = tf.reshape(
        tf.squeeze(self.energy_fn(flat_proposal_samples), axis=-1),
        [num_samples, self.K])
    indexes = tfd.Categorical(logits=log_energy).sample()  # [num_samples]
    samples = tf.batch_gather(proposal_samples,
                              tf.expand_dims(indexes, axis=-1))
    return tf.squeeze(samples, axis=1)  # Squeeze the selected dim


class NIS(AbstractNIS):
  """Self-normalized Importance Sampling distribution."""

  def __init__(self,
               K,
               data_dim,
               energy_hidden_sizes,
               proposal=None,
               data_mean=None,
               reparameterize_proposal_samples=True,
               dtype=tf.float32,
               name="nis"):
    """Creates a NIS model.

    Args:
      K: The number of proposal samples to take.
      data_dim: The dimension of the data.
      energy_hidden_sizes: The sizes of the hidden layers for the MLP that
        parameterizes the energy function.
      proposal: A distribution over the data space of this model. Must support
        sample() and log_prob() although log_prob only needs to return a lower
        bound on the true log probability. If not supplied, then defaults to
        Gaussian.
      data_mean: Mean of the data used to center the input.
      reparameterize_proposal_samples: Whether to allow gradients to pass
        through the proposal samples.
      dtype: Type of the tensors.
      name: Name to use for ops.
    """
    if data_mean is None:
      data_mean = tf.zeros((), dtype=dtype)
    energy_fn_helper = functools.partial(
        base.mlp,
        layer_sizes=energy_hidden_sizes + [1],
        final_activation=None,
        name="%s/energy_fn_mlp" % name)
    def energy_fn(x):
#      import ipdb
#      ipdb.set_trace()
      return energy_fn_helper(x - data_mean)
    super(NIS, self).__init__(K, data_dim, energy_fn, proposal, data_mean,
                              reparameterize_proposal_samples, dtype)


class ConvNIS(AbstractNIS):
  """Self-normalized Importance Sampling distribution with convs."""

  def __init__(self,
               K,
               data_dim,
               proposal=None,
               data_mean=None,
               reparameterize_proposal_samples=True,
               dtype=tf.float32,
               name="nis"):
    """Creates a NIS model.

    Args:
      K: The number of proposal samples to take.
      data_dim: The dimension of the data.
      proposal: A distribution over the data space of this model. Must support
        sample() and log_prob() although log_prob only needs to return a lower
        bound on the true log probability. If not supplied, then defaults to
        Gaussian.
      data_mean: Mean of the data used to center the input.
      reparameterize_proposal_samples: Whether to allow gradients to pass
        through the proposal samples.
      dtype: Type of the tensors.
      name: Name to use for ops.
    """
    if data_mean is None:
      data_mean = tf.zeros((), dtype=dtype)
    energy_fn = tf.keras.Sequential([
        tfkl.Lambda(lambda t: t - data_mean),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        conv(32, 4, 2, activation="relu"),
        tfkl.Flatten(),
        tfkl.Dense(1, activation=None),
    ])
    super(ConvNIS, self).__init__(K, data_dim, energy_fn, proposal, data_mean,
                                  reparameterize_proposal_samples, dtype)

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

# python3
"""Utilities for working with generative models."""
from typing import Callable, Tuple
from absl import logging

import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def _split_mu_log_sigma(params, slice_dim):
  """Splits `params` into `mu` and `log_sigma` along `slice_dim`."""
  params = tf.convert_to_tensor(params)
  size = params.get_shape()[slice_dim].value
  if size % 2 != 0:
    raise ValueError(
        '`params` must have an even size along dimension {}.'.format(slice_dim))
  half_size = size // 2
  mu = snt.SliceByDim(
      dims=[slice_dim], begin=[0], size=[half_size], name='mu')(
          params)
  log_sigma = snt.SliceByDim(
      dims=[slice_dim], begin=[half_size], size=[half_size], name='log_sigma')(
          params)
  return mu, log_sigma


class LogStddevNormal(tfp.distributions.Normal):
  """Diagonal Normal that accepts a concatenated[mu, log_sigma] tensor.

  It is safe to use this class with any `params` tensor as it expects log_sigma
  to be provided, which will then be exp'd and passed as sigma into the standard
  tf.distribution.Normal distribution.
  """

  def __init__(self, params, slice_dim=-1, name='Normal'):
    """Distribution constructor.

    Args:
      params: Tensor containing the concatenation of [mu, log_sigma] parameters.
        The shape of `params` must be known at graph construction (ie.
        params.get_shape() must work).
      slice_dim: Dimension along which params will be sliced to retrieve mu and
        log_sigma. Negative values index from the last dimension.
      name: Name of the distribution.

    Raises:
      ValueError: If the `params` tensor cannot be split evenly in two along the
        slicing dimension.
    """
    mu, self._log_sigma = _split_mu_log_sigma(params, slice_dim)
    sigma = tf.exp(self._log_sigma)
    super(LogStddevNormal, self).__init__(
        mu, sigma, name=name, validate_args=False)

  @property
  def log_scale(self):
    """Distribution parameter for log standard deviation."""
    return self._log_sigma


class QuantizedNormal(tfp.distributions.Normal):
  """Normal distribution with noise for quantized data."""

  def __init__(self,
               mu_log_sigma=None,
               mu=None,
               sigma=None,
               log_sigma=None,
               slice_dim=-1,
               bin_size=1 / 255,
               name='noisy_normal'):
    """Distribution constructor.

    Args:
      mu_log_sigma: Tensor, the concatenation along `slice_dim` or mu and
        log_sigma. Must not be specified if either `mu`, `sigma` or `log_sigma`
        is specified.
      mu: Tensor, the mean of the distribution. Must not be specified if
        `mu_log_sigma` is specified.
      sigma: Tensor, the standard deviation of the distribution. Must not be
        specified  if either `log_sigma` or `mu_log_sigma` is specified.
      log_sigma: Tensor, the log of the standard deviation of the distribution.
        Must not be specified if either `sigma` or `mu_log_sigma` is specified.
      slice_dim: Integer, specifies the dimension along which to split
        `mu_log_sigma`.
      bin_size: Number, specifies the width of the quantization bin.
      name: Name of the module.

    Raises:
      ValueError: if there is any redundancy when specifying the parameters
          of the distribution.
    """
    if mu_log_sigma is not None:
      if not all(v is None for v in (mu, sigma, log_sigma)):
        raise ValueError('If `mu_log_sigma` is provided, then `mu`, `sigma`, '
                         'and `log_sigma` should not be provided.')
      mu, self._log_sigma = _split_mu_log_sigma(mu_log_sigma, slice_dim)
      sigma = tf.exp(self._log_sigma)
    elif mu is not None:
      if sigma is None and log_sigma is None:
        raise ValueError('If `mu` is provided, then either `sigma` or '
                         '`log_sigma` need to be provided.')
      elif sigma is not None and log_sigma is not None:
        raise ValueError('`sigma` or `log_sigma` cannot be both provided.')
      elif log_sigma is not None:
        self._log_sigma = log_sigma
        sigma = tf.exp(log_sigma)
      else:
        self._log_sigma = tf.log(sigma)
    else:
      raise ValueError('Either `mu_log_sigma` or `mu` and either `sigma` or '
                       '`log_sigma` need to be provided.')

    self._bin_size = bin_size

    super(QuantizedNormal, self).__init__(
        mu, sigma, name=name, validate_args=False)

  def _log_prob(self, x):
    # Add quantization noise to the input.
    x += tf.random_uniform(
        tf.shape(x), -self._bin_size * 0.5, self._bin_size * 0.5, dtype=x.dtype)

    # Note: this relies on broadcasting. We know that `log_prob` has shape
    # of `x` for the Normal distribution, so we add tf.log(self._bin_size)
    # to each dimension of x, which when summed over non batch dimensions
    # will entailing adding tf.log(self._bin_size) * x.get_shape()[1:]
    return super(QuantizedNormal, self)._log_prob(x) + tf.cast(
        tf.log(self._bin_size), x.dtype)

  @property
  def log_sigma(self):
    """Distribution parameter for log standard deviation."""
    return self._log_sigma


def log_prob_elbo_components(encoder,
                             decoder,
                             prior,
                             x):
  """Computes ELBO terms for a Variational Autoencoder.

  Args:
    encoder: maps x to latent, q(z|x)
    decoder: maps z to distribution on x, p(x|z)
    prior: prior on z, p(z).
    x: input batch to compute terms over.

  Returns:
    log_p_x: log p(x|z) where z is a sample from the encoder
    kl: kl divergence between q(z|x) and p(z)
  """
  q = encoder(x)
  z = q.sample()

  try:
    kl = tfp.distributions.kl_divergence(q, prior)
  except NotImplementedError:
    logging.warn('Analytic KL divergence not available, using sampling KL'
                 'divergence instead')
    log_p_z = prior.log_prob(z, name='log_p_z')
    log_q_z = q.log_prob(z, name='log_q_z')

    # Reduce over all dimension except batch.
    sum_axis_p = list(range(1, log_p_z.get_shape().ndims))
    log_p_z = tf.reduce_sum(log_p_z, sum_axis_p)
    sum_axis_q = list(range(1, log_q_z.get_shape().ndims))
    log_q_z = tf.reduce_sum(log_q_z, sum_axis_q)

    kl = log_q_z - log_p_z

  # Reduce over all dimension except batch.
  sum_axis_kl = list(range(1, kl.get_shape().ndims))
  kl = tf.reduce_sum(kl, sum_axis_kl, name='kl')

  p = decoder(z)
  log_p_x = p.log_prob(x)
  # Reduce over all dimension except batch.
  sum_axis_logprob = list(range(1, log_p_x.get_shape().ndims))
  log_p_x = tf.reduce_sum(log_p_x, sum_axis_logprob, name='log_p_x')

  return log_p_x, kl

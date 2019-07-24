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

"""Implementation of Learned Acccept/Reject Sampling (Bauer & Mnih, 2018)."""

from __future__ import absolute_import
import functools
import tensorflow as tf
import tensorflow_probability as tfp

from eim.models import base
tfd = tfp.distributions


class SimpleLARS(object):
  """Learned Accept/Reject Sampling model."""

  def __init__(self,
               K,
               data_dim,
               accept_fn_layers,
               proposal=None,
               data_mean=None,
               ema_decay=0.99,
               dtype=tf.float32):
    self.k = K
    self.data_dim = data_dim
    self.ema_decay = ema_decay
    self.dtype = dtype
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.accept_fn = functools.partial(
        base.mlp,
        layer_sizes=accept_fn_layers + [1],
        final_activation=tf.math.log_sigmoid,
        name="a")
    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(
          loc=tf.zeros([data_dim], dtype=dtype),
          scale_diag=tf.ones([data_dim], dtype=dtype))
    else:
      self.proposal = proposal

  def log_prob(self, data):
    """Compute log likelihood estimate."""
    batch_size = tf.shape(data)[0]
    # Compute log a(z), log pi(z), and log q(z)
    log_a_z_r = tf.reshape(self.accept_fn(data - self.data_mean),
                           [batch_size])  # [batch_size]
    log_pi_z_r = self.proposal.log_prob(data)  # [batch_size]

    tf.summary.histogram("log_energy_data", log_a_z_r)

    # Sample zs from proposal to estimate Z
    z_s = self.proposal.sample(self.k)  # [K, data_dim]
    # Compute log a(z) for zs sampled from proposal
    log_a_z_s = tf.reshape(self.accept_fn(z_s - self.data_mean),
                           [self.k])  # [K]
    # pylint: disable=invalid-name
    log_ZS = tf.reduce_logsumexp(log_a_z_s)  # []
    log_Z_curr_avg = log_ZS - tf.log(tf.to_float(self.k))

    tf.summary.histogram("log_energy_proposal", log_a_z_s)

    # Set up EMA of Z
    Z_ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
    log_Z_curr_avg_sg = tf.stop_gradient(log_Z_curr_avg)
    Z_curr_avg_sg = tf.exp(log_Z_curr_avg_sg)
    maintain_Z_ema_op = Z_ema.apply([Z_curr_avg_sg])

    # In forward pass, log Z is log of the smoothed ema version of Z
    # In backward pass it is the current estimate of log Z, log_Z_curr_avg
    log_Z = log_Z_curr_avg  + tf.stop_gradient(
        tf.log(Z_ema.average(Z_curr_avg_sg)) - log_Z_curr_avg)
    # pylint: enable=invalid-name

    log_p = log_pi_z_r + log_a_z_r - log_Z[tf.newaxis]  # [batch_size]

    tf.summary.scalar("Z_ema", Z_ema.average(Z_curr_avg_sg))

    return log_p, maintain_Z_ema_op

  def sample(self, sample_shape=(1,)):
    """Sample from the model."""
    sample_shape = list(sample_shape)
    def while_body(z, accept):
      """Truncated rejection sampling."""
      new_z = self.proposal.sample(sample_shape)
      accept_prob = tf.reshape(
          tf.exp(self.accept_fn(new_z - self.data_mean)), sample_shape)
      new_accept = tf.math.less_equal(
          tf.random_uniform(shape=sample_shape, minval=0., maxval=1.),
          accept_prob)
      accepted = tf.logical_or(accept, new_accept)
      swap = tf.math.logical_and(tf.math.logical_not(accept), new_accept)
      z = tf.where(swap, new_z, z)
      return z, accepted

    def while_cond(_, accept):
      return tf.reduce_any(tf.logical_not(accept))

    shape = sample_shape + [self.data_dim]
    z0 = tf.zeros(shape, dtype=self.dtype)
    accept0 = tf.constant(False, shape=sample_shape)
    zs, _ = tf.while_loop(while_cond, while_body, loop_vars=(z0, accept0))
    return zs

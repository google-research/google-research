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
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from eim.models import base
tfd = tfp.distributions


class LARS(object):
  """Learned Accept/Reject Sampling model."""

  def __init__(self,
               K,
               T,
               data_dim,
               accept_fn_layers,
               proposal=None,
               data_mean=None,
               ema_decay=0.99,
               dtype=tf.float32,
               is_eval=False):
    self.k = K
    self.T = T  # pylint: disable=invalid-name
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
      self.proposal = base.get_independent_normal(data_dim)
    else:
      self.proposal = proposal
    self.is_eval = is_eval
    if is_eval:
      self.Z_estimate = tf.placeholder(tf.float32, shape=[])  # pylint: disable=invalid-name

    with tf.variable_scope("LARS_Z_ema", reuse=tf.AUTO_REUSE):
      self.Z_ema = tf.get_variable(  # pylint: disable=invalid-name
          name="LARS_Z_ema",
          shape=[],
          dtype=dtype,
          initializer=tf.constant_initializer(0.5),
          trainable=False)

  def log_prob(self, data, log_q_data=None, num_samples=1):
    """Compute log likelihood estimate."""
    # Compute log a(z), log pi(z), and log q(z)
    log_a_z_r = tf.squeeze(self.accept_fn(data - self.data_mean),
                           axis=-1)  # [batch_size]
    # [batch_size]
    try:
      # Try giving the proposal lower bound num_samples if it can use it.
      log_pi_z_r = self.proposal.log_prob(data, num_samples=num_samples)
    except TypeError:
      log_pi_z_r = self.proposal.log_prob(data)

    tf.summary.histogram("log_energy_data", log_a_z_r)
    if not self.is_eval:
      # Sample zs from proposal to estimate Z
      z_s = self.proposal.sample(self.k)  # [K, data_dim]
      # Compute log a(z) for zs sampled from proposal
      log_a_z_s = tf.squeeze(self.accept_fn(z_s - self.data_mean),
                             axis=-1)  # [K]
      tf.summary.histogram("log_energy_proposal", log_a_z_s)

      # pylint: disable=invalid-name
      log_ZS = tf.reduce_logsumexp(log_a_z_s)  # []
      log_Z_curr_avg = log_ZS - tf.log(tf.to_float(self.k))

      if log_q_data is not None:
        # This may only be valid when log pi is exact (i.e., not a lower bound).
        Z_curr_avg = (1. / (self.k + 1.)) * (
            tf.exp(log_ZS) +
            tf.exp(log_a_z_r + tf.stop_gradient(log_pi_z_r - log_q_data)))
      else:
        Z_curr_avg = tf.exp(log_Z_curr_avg)

      self.Z_smooth = (
          self.ema_decay * self.Z_ema + (1 - self.ema_decay) * Z_curr_avg)

      # In forward pass, log Z is log of the smoothed ema version of Z
      # In backward pass it is the current estimate of log Z, log_Z_curr_avg
      Z = Z_curr_avg + tf.stop_gradient(self.Z_smooth - Z_curr_avg)
      tf.summary.scalar("Z", tf.reduce_mean(Z))
    else:
      Z = self.Z_estimate  # pylint: disable=invalid-name

    # pylint: enable=invalid-name
    alpha = tf.pow(1. - Z, self.T - 1)
    log_prob = log_pi_z_r + tf.log(tf.exp(log_a_z_r) * (1. - alpha) / Z + alpha)
    return log_prob

  def post_train_op(self):
    # Set up EMA of Z (EMA is updated after gradient step).
    return tf.assign(self.Z_ema, tf.reduce_mean(self.Z_smooth))

  def compute_Z(self, num_samples):  # pylint: disable=invalid-name
    r"""Returns log(\sum_i a(z_i) / num_samples)."""
    z_s = self.proposal.sample(num_samples)  # [num_samples, data_dim]
    # Compute log a(z) for zs sampled from proposal
    log_a_z_s = tf.squeeze(self.accept_fn(z_s - self.data_mean),
                           axis=-1)  # [num_samples]
    log_Z = tf.reduce_logsumexp(log_a_z_s) - tf.log(  # pylint: disable=invalid-name
        tf.to_float(num_samples))  # []
    return log_Z

  def sample(self, num_samples=1):
    """Sample from the model."""

    def while_body(t, z, accept):
      """Truncated rejection sampling."""
      new_z = self.proposal.sample(num_samples)
      accept_prob = tf.squeeze(tf.exp(self.accept_fn(new_z - self.data_mean)),
                               axis=-1)
      new_accept = tf.math.less_equal(
          tf.random_uniform(shape=[num_samples], minval=0., maxval=1.),
          accept_prob)
      force_accept = tf.math.greater_equal(
          tf.to_float(t),
          tf.to_float(self.T) - 1.)
      new_accept = tf.math.logical_or(new_accept, force_accept)
      accepted = tf.logical_or(accept, new_accept)
      swap = tf.math.logical_and(tf.math.logical_not(accept), new_accept)
      z = tf.where(swap, new_z, z)
      return t + 1, z, accepted

    def while_cond(unused_t, unused_z, accept):
      return tf.reduce_any(tf.logical_not(accept))

    shape = [num_samples] + self.data_dim
    z0 = tf.zeros(shape, dtype=self.dtype)
    accept0 = tf.constant(False, shape=[num_samples])
    _, zs, _ = tf.while_loop(while_cond, while_body, loop_vars=(0, z0, accept0))
    return zs

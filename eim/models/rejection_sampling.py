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

"""Truncated Rejection Sampling distribution."""

from __future__ import absolute_import
import functools
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from eim.models import base
tfd = tfp.distributions


class AbstractRejectionSampling(base.ProbabilisticModel):
  """Truncated Rejection Sampling distribution."""

  def __init__(self,  # pylint: disable=invalid-name
               T,
               data_dim,
               logit_accept_fn,
               proposal=None,
               dtype=tf.float32,
               name="rejection_sampling"):
    """Creates a Rejection Sampling model.

    Args:
      T: The maximum number of proposals to sample in the rejection sampler.
      data_dim: The dimension of the data. Should be a list.
      logit_accept_fn: Accept function, takes [batch_size] + data_dim to [0, 1].
      proposal: A distribution over the data space of this model. Must support
        sample() and log_prob() although log_prob only needs to return a lower
        bound on the true log probability. If not supplied, then defaults to
        Gaussian.
      dtype: Type of data.
      name: Name to use in scopes.
    """
    self.T = T  # pylint: disable=invalid-name
    self.data_dim = data_dim
    self.logit_accept_fn = logit_accept_fn
    if proposal is None:
      self.proposal = base.get_independent_normal(data_dim)
    else:
      self.proposal = proposal
    self.name = name
    self.dtype = dtype
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      self.logit_Z = tf.get_variable(  # pylint: disable=invalid-name
          name="logit_Z",
          shape=[],
          dtype=dtype,
          initializer=tf.constant_initializer(0.),
          trainable=True)

    tf.summary.scalar("Expected_trials", tf.exp(-tf.log_sigmoid(self.logit_Z)))

  def trainable_variables(self):
    return [self.logit_Z] + self.logit_accept_fn.trainable_variables

  def _log_prob(self, data, num_samples=1):
    """Assumes data is [batch_size] + data_dim."""
    batch_size = tf.shape(data)[0]
    log_Z = tf.log_sigmoid(self.logit_Z)  # pylint: disable=invalid-name
    log_1mZ = -self.logit_Z + log_Z  # pylint: disable=invalid-name

    # [B]
    data_log_accept = tf.squeeze(
        tf.log_sigmoid(self.logit_accept_fn(data)), axis=-1)
    truncated_geometric_log_probs = tf.range(
        self.T - 1, dtype=self.dtype) * log_1mZ
    # [B, T-1]
    truncated_geometric_log_probs = (
        truncated_geometric_log_probs[None, :] + data_log_accept[:, None])
    # [B, T]
    truncated_geometric_log_probs = tf.concat([
        truncated_geometric_log_probs,
        tf.tile((self.T - 1) * log_1mZ[None, None], [batch_size, 1])
    ],
                                              axis=-1)
    truncated_geometric_log_probs -= tf.reduce_logsumexp(
        truncated_geometric_log_probs, axis=-1, keepdims=True)

    # [B]
    entropy = -tf.reduce_sum(
        tf.exp(truncated_geometric_log_probs) * truncated_geometric_log_probs,
        axis=-1)

    proposal_samples = self.proposal.sample([self.T])  # [T] + data_dim
    proposal_logit_accept = self.logit_accept_fn(proposal_samples)
    proposal_log_reject = tf.reduce_mean(-proposal_logit_accept +
                                         tf.log_sigmoid(proposal_logit_accept))

    # [B]
    noise_term = tf.reduce_sum(
        tf.exp(truncated_geometric_log_probs) *
        tf.range(self.T, dtype=self.dtype)[None, :] * proposal_log_reject,
        axis=-1)

    try:
      # Try giving the proposal lower bound num_samples if it can use it.
      log_prob_proposal = self.proposal.log_prob(data, num_samples=num_samples)
    except TypeError:
      log_prob_proposal = self.proposal.log_prob(data)
    elbo = log_prob_proposal + data_log_accept + noise_term + entropy

    return elbo

  def sample(self, num_samples=1):
    """Sample from the rejection sampling distribution.

    For ease of implementation, draw the maximum number of proposal samples.

    Args:
      num_samples: integer, number of samples to draw.

    Returns:
      samples: Tensor of samples from the distribution, [num_samples] + data_dim
    """
    flat_proposal_samples = self.proposal.sample(num_samples * self.T)
    proposal_samples = tf.reshape(flat_proposal_samples,
                                  [num_samples, self.T] + self.data_dim)
    flat_logit_accept = self.logit_accept_fn(flat_proposal_samples)
    logit_accept = tf.reshape(flat_logit_accept, [num_samples, self.T])
    accept_samples = tfd.Bernoulli(logits=logit_accept[:, :-1]).sample()

    # Add forced accept to last sample to ensure truncation
    accept_samples = tf.concat([
        accept_samples,
        tf.ones([num_samples, 1], dtype=accept_samples.dtype)
    ], axis=-1)

    # For each of sample_shape, find the first nonzero accept
    def get_first_nonzero_index(t):
      # t is batch_dims + [T], t is binary.
      _, indices = tf.math.top_k(t, k=1, sorted=False)
      return indices

    accept_indices = get_first_nonzero_index(accept_samples)  # sample_shape
    samples = tf.batch_gather(proposal_samples, accept_indices)
    return tf.squeeze(samples, axis=1)  # Squeeze the selected dim


class RejectionSampling(AbstractRejectionSampling):
  """Truncated Rejection Sampling distribution."""

  def __init__(self,  # pylint: disable=invalid-name
               T,
               data_dim,
               energy_hidden_sizes,
               proposal=None,
               data_mean=None,
               dtype=tf.float32,
               name="rejection_sampling"):
    if data_mean is None:
      data_mean = tf.zeros((), dtype=dtype)
    logit_accept_fn = functools.partial(
        base.mlp,
        layer_sizes=energy_hidden_sizes + [1],
        final_activation=None,
        name="rejection_sampling/energy_fn_mlp")
    super(RejectionSampling, self).__init__(
        T=T,
        data_dim=data_dim,
        proposal=proposal,
        logit_accept_fn=logit_accept_fn,
        dtype=dtype)

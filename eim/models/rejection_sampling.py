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
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class RejectionSampling(object):
  """Truncated Rejection Sampling distribution."""

  def __init__(self,
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
      self.proposal = tfd.MultivariateNormalDiag(
          loc=tf.zeros(data_dim, dtype=dtype),
          scale_diag=tf.ones(data_dim, dtype=dtype))
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

  def log_prob(self, data, num_samples=1):
    """Reshape data so that it is [batch_size] + data_dim."""
    batch_shape = tf.shape(data)[:-len(self.data_dim)]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape)] +
                               self.data_dim)
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

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

  def sample(self, sample_shape=(1,)):
    """Sample from the rejection sampling distribution.

    For ease of implementation, draw the maximum number of proposal samples.

    Args:
      sample_shape: Shape of samples to draw.

    Returns:
      samples: Tensor of samples from the distribution, sample_shape + data_dim
    """
    sample_shape = list(sample_shape)
    shape = sample_shape + [self.T]
    proposal_samples = self.proposal.sample(
        shape)  # sample_shape + [T] + data_dim

    # Work in the batched space
    batched_proposal_samples = tf.reshape(proposal_samples[:, :-1],
                                          [-1] + self.data_dim)
    logit_accept = self.logit_accept_fn(batched_proposal_samples)
    accept_samples = tfd.Bernoulli(logits=logit_accept).sample()

    # Reshape and add accept to last sample to ensure truncation
    accept_samples = tf.concat([
        tf.reshape(accept_samples, sample_shape + [self.T - 1]),
        tf.ones(sample_shape + [1], dtype=accept_samples.dtype)
    ],
                               axis=-1)

    # For each of sample_shape, find the first nonzero accept
    def get_first_nonzero_index(t):
      # t is batch_dims + [T], t is binary.
      _, indices = tf.math.top_k(t, k=1, sorted=False)
      return indices

    accept_indices = get_first_nonzero_index(accept_samples)  # sample_shape
    samples = tf.batch_gather(proposal_samples, accept_indices)
    return tf.squeeze(samples, axis=len(sample_shape))

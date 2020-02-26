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

# Lint as: python3
"""Shared test library between sgmcmc_test and diagnostics_test.

This library implements test classes used by both the SG-MCMC tests and the
diagnostics module test cases.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def sample_model(model,
                 optimizer,
                 nsamples,
                 batchsize=32,
                 transpose=True):
  """Sample from a given model using a SG-MCMC optimizer.

  Args:
    model: a tf.keras.Model that returns negative log-probability.
    optimizer: a tf.keras.optimizers.Optimizer from this file.
    nsamples: int, total number of samples to create.  This method will return
      at least as many samples as this value.
    batchsize: int, the number of unrolled sampling iterations.
    transpose: bool, whether to transpose the sampling result.

  Returns:
    samples: Tensor, if `transpose`, then (ndim, nsamples_out), tf.float32.
      If `transpose == False`, then (nsamples_out, ndim), tf.float32.
      The samples are ordered exactly as the MCMC sampler has produced them,
      so they can be used for e.g. autocorrelation estimation.
  """
  # Unroll sampling code for a given number of iterations
  @tf.function
  def sample_sgmcmc(count):
    """Method to sample from SG-MCMC."""
    samples = []
    for _ in range(count):
      with tf.GradientTape() as tape:
        nll = model(tf.zeros(1, 1), tf.zeros(1, 1))

      gradients = tape.gradient(nll, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      samples.append(tf.identity(model.trainable_variables[0]))

    return tf.stack(samples)

  # Sample in batches
  nloops = (nsamples // batchsize) + 1
  samples = []
  for _ in range(nloops):
    samples.append(sample_sgmcmc(batchsize))
  samples = tf.concat(samples, 0)
  if transpose:
    samples = tf.transpose(samples)

  return samples


def compute_ess(samples):
  """Compute an estimate of the effective sample size (ESS).

  See the [Stan
  manual](https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html)
  for a definition of the effective sample size in the context of MCMC.

  Args:
    samples: Tensor, vector (n,), float32 of n sequential observations.

  Returns:
    ess: float, effective sample size, >= 1, <= n.
    efficiency: float, >= 0.0, the relative efficiency obtained compared to
      the naive Monte Carlo estimate which has an efficiency of one.
  """
  ess, efficiency = compute_ess_multidimensional(
      tf.reshape(samples, (1, tf.size(samples))))
  ess = ess[0]
  efficiency = efficiency[0]

  return ess, efficiency


def compute_ess_multidimensional(samples):
  """Compute an estimate of the effective sample size (ESS) per dimension.

  See the [Stan
  manual](https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html)
  for a definition of the effective sample size in the context of MCMC.

  Args:
    samples: Tensor, (m,n), float32 of n sequential observations.

  Returns:
    ess: Tensor, (m,) float32, effective sample sizes per dimension, each
      element being >= 1, <= n.
    efficiency: Tensor, (m,) float32, each element >= 0.0, the relative
      efficiency obtained compared to the naive Monte Carlo estimate which has
      an efficiency of one.
  """
  autocorr = tfp.stats.auto_correlation(samples).numpy()
  first_neg = np.argmax(autocorr <= 0.0, axis=1)
  efficiency = []
  ess = []
  for d, fn in enumerate(first_neg):
    eff = 1.0 / (1.0 + np.sum(2.0*autocorr[d, 1:(fn-1)]))
    efficiency.append(eff)
    ess.append(autocorr.shape[1] * eff)

  return ess, efficiency


def flatten(samples):
  """Flatten the input tensor into a vector."""
  return tf.reshape(samples, (tf.size(samples),))


class StandardNormal1D(tf.keras.layers.Layer):
  """1D standard Normal model suitable for testing SG-MCMC procedures."""

  def build(self, input_shape):
    self.w = self.add_weight(shape=(1,), trainable=True)

  def call(self, inputs):
    return -tfp.distributions.Normal(loc=0.0, scale=1.0).log_prob(self.w)


@tf.custom_gradient
def _add_gradient_noise(x, noise_sigma):
  """Add gradient noise of a given standard deviation.

  This function behaves like identity on the forward path.  On the backward path
  Gaussian-distributed gradient noise is added.

  Args:
    x: Tensor, arbitrary shape, passed through.
    noise_sigma: float or scalar Tensor, the noise standard deviation.

  Returns:
    x: Tensor, the input tensor.
  """
  def grad(dy):
    return dy + noise_sigma*tf.random_normal(dy.shape), None
  return x, grad


class Normal2D(tf.keras.layers.Layer):
  """2D Normal model suitable for testing SG-MCMC procedures."""

  def __init__(self, stddev_x1=1.0, stddev_x2=1.0, correlation=0.99,
               noise_sigma=0.0, uniform_noise=True):
    """Create a bivariate Normal model with noisy log-likelihood gradient.

    Args:
      stddev_x1: float, >0, the standard deviation in the first coordinate.
      stddev_x2: float, >0, the standard deviation in the second coordinate.
      correlation: float, > -1.0, < 1.0, the correlation between the first and
        second coordinate.
      noise_sigma: float, >= 0.0, the noise level standard deviation of the
        2G Gaussian noise added to the log-likelihood gradient.
      uniform_noise: bool, if True the noise is added to the log-likelihood
        gradient everywhere.
        If False, it is added only in the lower-left diagonal of the 2D space.
    """
    super(Normal2D, self).__init__()

    cov = [[stddev_x1**2.0, correlation*stddev_x1*stddev_x2],
           [correlation*stddev_x1*stddev_x2, stddev_x2**2.0]]
    self.cov = tf.convert_to_tensor(cov)
    self.dist = tfp.distributions.MultivariateNormalTriL(
        loc=tf.zeros((2,)), scale_tril=tf.linalg.cholesky(self.cov))

    self.uniform_noise = uniform_noise
    self.noise_sigma = noise_sigma

  def build(self, input_shape):
    self.w = self.add_weight(shape=(2,), trainable=True)

  def call(self, inputs):
    if self.uniform_noise:
      w = _add_gradient_noise(self.w, self.noise_sigma)
    else:
      # Add noise only in lower left corner
      w = tf.cond(tf.reduce_sum(self.w) <= 0.0,
                  lambda: _add_gradient_noise(self.w, self.noise_sigma),
                  lambda: self.w)

    return -self.dist.log_prob(w)


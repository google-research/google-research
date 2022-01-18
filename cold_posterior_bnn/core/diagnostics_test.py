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

# Lint as: python3
"""Tests for google_research.google_research.cold_posterior_bnn.core.diagnostics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

from absl import logging
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from cold_posterior_bnn.core import diagnostics
from cold_posterior_bnn.core import sgmcmc_testlib


def _generate_symmetric_alpha_stable_variates(stability, shape):
  """Generate Symmetric-alpha-Stable variates.

  Args:
    stability: >0, <= 2.0, stability parameter.  We must have stability != 1.
    shape: shape of the Tensor to generate.

  Returns:
    sample: tf.Tensor of given shape containing SaS(stability,0,0,0) variates.
  """
  # Algorithm of (Chambers et al., 1976)
  # https://en.wikipedia.org/wiki/Stable_distribution#Simulation_of_stable_variables
  u = tf.random.uniform(shape, minval=-0.5*math.pi, maxval=0.5*math.pi)
  w = -tf.math.log(tf.random.uniform(shape))  # ~ Exponential(1)
  x1 = tf.math.sin(stability*u) / (tf.math.cos(u)**(1.0/stability))
  x2 = (tf.math.cos(u - stability*u)/w)**((1.0-stability)/stability)
  sample = x1*x2

  return sample


class DiagnosticsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      [0.25, 0.4, 0.6, 0.75, 0.9, 1.1, 1.25, 1.5, 1.75, 2.0])
  def test_stability_estimation(self, stability):
    nsamples = 1048576
    samples = _generate_symmetric_alpha_stable_variates(stability, (nsamples,))
    stability_estimate = 1.0 / (
        diagnostics.symmetric_alpha_stable_invstability_estimator(
            samples, 0, 16))
    self.assertAlmostEqual(stability, stability_estimate, delta=0.025,
                           msg='Inaccurate stability index estimate: '
                           'true stability %.3f, estimate %.3f.' % (
                               stability, stability_estimate))

  def test_variable_gradient_stability(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=100))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(10))

    nsamples = 131072
    data = tf.random.normal((nsamples, 100))
    labels = tf.reshape(tf.random.categorical(tf.zeros((nsamples, 10)), 1),
                        (nsamples,))

    with tf.GradientTape(persistent=True) as tape:
      logits = model(data, training=True)
      ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

    batchsize = 64

    # 1. Aggregate estimates
    stability_estimates = diagnostics.variable_gradient_stability_estimate(
        model, tape, ce, batchsize, nelem_per_piece=8)
    self.assertLen(stability_estimates, len(model.trainable_variables))
    for stability_est, parameter in zip(stability_estimates,
                                        model.trainable_variables):
      logging.info('Parameter "%s" has estimated stability %.5f',
                   parameter.name, float(stability_est))
      self.assertEqual(int(tf.size(stability_est)), 1,
                       msg='Stability estimate is not scalar.')
      self.assertGreaterEqual(float(stability_est), -0.1)
      self.assertLessEqual(float(stability_est), 2.1)

    # 2. Per-parameter estimates
    stability_estimates = diagnostics.variable_gradient_stability_estimate(
        model, tape, ce, batchsize, nelem_per_piece=8,
        aggregate_variable_estimates=False)
    for stability_est, parameter in zip(stability_estimates,
                                        model.trainable_variables):
      self.assertEqual(stability_est.shape, parameter.shape)
      self.assertAllInRange(stability_est, -1.5, 3.5)

  @parameterized.parameters([0.0, 0.25, 2.5])
  def test_gradient_noise_estimate(self, noise_sigma):
    pmodel = sgmcmc_testlib.Normal2D(noise_sigma=noise_sigma)
    model = tf.keras.Sequential([pmodel])

    grad_est = diagnostics.GradientNoiseEstimator()

    @tf.function
    def step_model(count):
      for _ in range(count):
        with tf.GradientTape() as tape:
          nll = model(tf.zeros(1, 1), tf.zeros(1, 1))

        gradients = tape.gradient(nll, model.trainable_variables)
        grad_est.apply_gradients(zip(gradients, model.trainable_variables))

    for _ in range(200):
      step_model(50)

    precond_dict = grad_est.estimate_fixed_preconditioner(
        model, scale_to_min=False)

    # Check that the estimated mass closely matches the noise stddev
    for name in precond_dict:
      mass = precond_dict[name]
      logging.info('Variable "%s" estimated mass %.5f, true stddev %.5f',
                   name, mass, noise_sigma)
      self.assertAlmostEqual(mass, noise_sigma, delta=0.02,
                             msg='Estimates mass %.5f differs from true '
                             'stddev %.5f' % (mass, noise_sigma))

  def _generate_autoregressive_data(self, ar_rate, shape, nitems):
    """Generate an example AR(1) data set with known autocorrelation."""
    data = list()
    data.append(tf.random.normal(shape))
    for t in range(1, nitems):
      data.append(ar_rate * data[t - 1] + tf.random.normal(shape))

    return data

  _AR_RATES = [0.75, 0.9, 0.99, 0.999]

  @parameterized.parameters(itertools.product(_AR_RATES))
  def test_autocorr_estimation(self, ar_rate):
    tf.set_random_seed(1)
    shape = (8, 16)
    data = self._generate_autoregressive_data(ar_rate, shape, 10000)

    acorr = diagnostics.AutoCorrelationEstimator(
        shape, nlevels=3, nsteps_per_level=16)
    for data_item in data:
      # TODO(nowozin): try to @tf.function this
      acorr.update(data_item)

    test_points = [1, 2, 3, 4, 10, 20, 30]
    for tp in test_points:
      autocorr_exact = ar_rate**tp
      autocorr_estimate = float(tf.reduce_mean(acorr(tp)))
      logging.info('AR @ %d (rate=%.4f), exact %.5f, estimate %.5f', tp,
                   ar_rate, autocorr_exact, autocorr_estimate)

      abs_diff = math.fabs(autocorr_exact - autocorr_estimate)
      ratio = autocorr_estimate / autocorr_exact
      self.assertTrue(
          abs_diff < 0.05 or (1.0 / ratio <= 1.4 and ratio <= 1.4),
          msg='Autocorrelation error, exact %.5f, estimate %.5f, '
          'abs_diff %.5f, ratio %.3f' %
          (autocorr_exact, autocorr_estimate, abs_diff, ratio))

    # Test the time-to-one-sample (TT1) estimates
    # TT1 is difficult to estimate and we only do a small number of samples
    # here, therefore the tolerances are quite generous
    tt1_exact = 1.0 / (1.0 - ar_rate)
    tt1_estimate = acorr.time_to_one_sample()
    logging.info('TT1 (rate=%.4f), exact %.5f, estimate %.5f', ar_rate,
                 tt1_exact, tt1_estimate)
    self.assertLess(
        tt1_estimate,
        4.0 * tt1_exact,
        msg='Estimated TT1 %.5f too large, true TT1 %.5f' %
        (tt1_estimate, tt1_exact))
    self.assertGreater(
        tt1_estimate,
        tt1_exact / 4.0,
        msg='Estimated TT1 %.5f too small, true TT1 %.5f' %
        (tt1_estimate, tt1_exact))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()

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

# Lint as: python3
"""Tests for google_research.google_research.cold_posterior_bnn.core.statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from cold_posterior_bnn.core import statistics as stats

tfd = tfp.distributions
TOL = 1e-7


class StatisticsTest(parameterized.TestCase, tf.test.TestCase):

  def test_classification_prob(self):
    cprob = stats.ClassificationLogProb()

    logits1 = tf.math.log([[0.3, 0.7], [0.6, 0.4]])
    logits2 = tf.math.log([[0.2, 0.8], [0.5, 0.5]])
    logits3 = tf.math.log([[0.4, 0.6], [0.4, 0.6]])

    cprob.reset()
    cprob.update(logits1)
    cprob.update(logits2)
    cprob.update(logits3)
    log_prob = cprob.result()

    self.assertAlmostEqual(math.log(0.3), float(log_prob[0, 0]), delta=TOL)
    self.assertAlmostEqual(math.log(0.7), float(log_prob[0, 1]), delta=TOL)
    self.assertAlmostEqual(math.log(0.5), float(log_prob[1, 0]), delta=TOL)
    self.assertAlmostEqual(math.log(0.5), float(log_prob[1, 1]), delta=TOL)

  def test_brier_score(self):
    logits1 = tf.math.log([[0.3, 0.7], [0.3, 0.7]])
    logits2 = tf.math.log([[0.2, 0.8], [0.6, 0.4]])
    logits3 = tf.math.log([[0.4, 0.6], [0.4, 0.6]])

    labels = tf.convert_to_tensor([0, 1], dtype=tf.int32)

    brier = stats.BrierScore()
    brier.reset()
    brier.update(logits1, labels)
    brier.update(logits2, labels)
    brier.update(logits3, labels)
    brier_score = brier.result()

    brier_score_true_0 = 0.3*0.3 + 0.7*0.7 - 2.0*0.3
    brier_score_true_1 = (1.3/3.0)**2.0 + (1.7/3.0)**2.0 - 2.0*(1.7/3.0)
    self.assertAlmostEqual(float(brier_score[0]), brier_score_true_0, delta=TOL)
    self.assertAlmostEqual(float(brier_score[1]), brier_score_true_1, delta=TOL)

  def _generate_perfect_calibration_logits(self, nsamples, nclasses,
                                           inv_temp=2.0):
    """Generate well distributed and well calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.
      inv_temp: float, >= 0.0, inverse temperature parameter.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    logits = inv_temp*tf.random.normal((nsamples, nclasses))
    logits = tf.math.log_softmax(logits)
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()

    return logits, labels

  def _generate_random_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and poorly calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    logits = 2.0*tf.random.normal((nsamples, nclasses))
    logits = tf.math.log_softmax(logits)
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()
    logits_other = 2.0*tf.random.normal((nsamples, nclasses))
    logits_other = tf.math.log_softmax(logits_other)

    return logits_other, labels

  @parameterized.parameters(
      (5, 3, 50000), (10, 5, 50000)
  )
  def test_ece_calibrated(self, num_bins, nclasses, nsamples):
    logits, labels = self._generate_perfect_calibration_logits(
        nsamples, nclasses)

    ece_stat = stats.ECE(num_bins)
    ece_stat.reset()
    ece_stat.update(logits, labels)
    ece = float(ece_stat.result())

    ece_tolerance = 0.01
    self.assertLess(ece, ece_tolerance, msg="ECE %.5f > %.2f for perfectly "
                    "calibrated logits" % (ece, ece_tolerance))

  @parameterized.parameters(
      (True, 3, 50000), (True, 5, 50000), (True, 10, 50000),
      (False, 3, 50000), (False, 5, 50000), (False, 10, 50000),
  )
  def test_brier_decomposition(self, well_calib, nclasses, nsamples):
    """Recompose the Brier decomposition and compare it to the Brier score."""
    if well_calib:
      logits, labels = self._generate_perfect_calibration_logits(
          nsamples, nclasses, inv_temp=0.25)
    else:
      logits, labels = self._generate_random_calibration_logits(
          nsamples, nclasses)

    score = stats.BrierScore()
    uncert = stats.BrierUncertainty()
    resol = stats.BrierResolution()
    reliab = stats.BrierReliability()

    for stat in [score, uncert, resol, reliab]:
      stat.reset()
      stat.update(logits, labels)

    score = float(tf.reduce_mean(score.result()))
    uncert = float(uncert.result())
    resol = float(resol.result())
    reliab = float(reliab.result())

    self.assertGreaterEqual(resol, 0.0, "Brier resolution is negative, this "
                            "should not happen.")
    self.assertGreaterEqual(reliab, 0.0, "Brier reliability is negative, this "
                            "should not happen.")

    score_from_decomposition = uncert - resol + reliab

    if well_calib:
      calib_str = "calibrated"
    else:
      calib_str = "uncalibrated"
    logging.info("Brier decomposition (%s) (n=%d, K=%d), "
                 "%.5f = %.5f - %.5f + %.5f (%.5f, diff %.5f)",
                 calib_str, nsamples, nclasses, score, uncert, resol, reliab,
                 score_from_decomposition, score - score_from_decomposition)

    self.assertAlmostEqual(score, score_from_decomposition, delta=0.025,
                           msg="Brier decomposition sums to %.5f which "
                           "deviates from Brier score %.5f" % (
                               score_from_decomposition, score))

  @parameterized.parameters(
      (3, 50000), (5, 50000)
  )
  def test_brierreliab_poorly_calibrated(self, nclasses, nsamples):
    logits, labels = self._generate_random_calibration_logits(
        nsamples, nclasses)

    brierreliab_stat = stats.BrierReliability()
    brierreliab_stat.reset()
    brierreliab_stat.update(logits, labels)
    reliab = float(brierreliab_stat.result())

    reliab_lower = 0.2
    self.assertGreater(reliab, reliab_lower,
                       msg="Brier reliability %.5f < %.2f for random "
                       "logits" % (reliab, reliab_lower))

  @parameterized.parameters(
      (3, 50000), (5, 50000)
  )
  def test_brierreliab_calibrated(self, nclasses, nsamples):
    logits, labels = self._generate_perfect_calibration_logits(
        nsamples, nclasses)

    brierreliab_stat = stats.BrierReliability()
    brierreliab_stat.reset()
    brierreliab_stat.update(logits, labels)
    reliab = float(brierreliab_stat.result())

    reliab_tolerance = 0.1
    self.assertLess(reliab, reliab_tolerance,
                    msg="Brier reliability %.5f > %.2f for perfectly "
                    "calibrated logits" % (reliab, reliab_tolerance))

  @parameterized.parameters(
      (5, 3, 50000), (10, 5, 50000)
  )
  def test_ece_poorly_calibrated(self, num_bins, nclasses, nsamples):
    logits, labels = self._generate_random_calibration_logits(
        nsamples, nclasses)

    ece_stat = stats.ECE(num_bins)
    ece_stat.reset()
    ece_stat.update(logits, labels)
    ece = float(ece_stat.result())

    ece_lower = 0.2
    self.assertGreater(ece, ece_lower, msg="ECE %.5f < %.2f for random "
                       "logits" % (ece, ece_lower))

  def test_standarddeviation(self):
    logits = tf.math.log([[0.3, 0.7], [0.3, 0.7]])
    labels = tf.convert_to_tensor([0, 1], dtype=tf.int32)

    caccuracy = stats.Accuracy()
    caccuracy.reset()
    caccuracy.update(logits, labels)
    accuracy = caccuracy.result()
    self.assertEqual(0.0, float(accuracy[0]))
    self.assertEqual(1.0, float(accuracy[1]))

    accstddev = stats.StandardDeviation(stats.Accuracy())
    accstddev.reset()
    accstddev.update(logits, labels)
    stddev = accstddev.result()

    self.assertAlmostEqual(0.5*math.sqrt(2.0), float(stddev), delta=TOL)

  def test_standarderror(self):
    logits = tf.math.log([[0.3, 0.7], [0.3, 0.7]])
    labels = tf.convert_to_tensor([0, 1], dtype=tf.int32)

    accsem = stats.StandardError(stats.Accuracy())
    accsem.reset()
    accsem.update(logits, labels)
    sem = accsem.result()

    self.assertAlmostEqual(0.5, float(sem), delta=TOL)

  def test_classification_accuracy(self):
    logits1 = tf.math.log([[0.3, 0.7], [0.3, 0.7]])
    logits2 = tf.math.log([[0.2, 0.8], [0.6, 0.4]])
    logits3 = tf.math.log([[0.4, 0.6], [0.4, 0.6]])

    labels = tf.convert_to_tensor([0, 1], dtype=tf.int32)

    caccuracy = stats.Accuracy()
    caccuracy.reset()
    caccuracy.update(logits1, labels)
    caccuracy.update(logits2, labels)
    caccuracy.update(logits3, labels)
    accuracy = caccuracy.result()

    self.assertEqual(0.0, float(accuracy[0]))
    self.assertEqual(1.0, float(accuracy[1]))

    gaccuracy = stats.GibbsAccuracy()
    gaccuracy.reset()
    gaccuracy.update(logits1, labels)
    gaccuracy.update(logits2, labels)
    gaccuracy.update(logits3, labels)
    accuracy = gaccuracy.result()

    self.assertEqual(0.0, float(accuracy[0]))
    self.assertAlmostEqual(0.666666667, float(accuracy[1]), delta=TOL)

  def test_classification_ce(self):
    cce = stats.ClassificationCrossEntropy()

    logits1 = tf.math.log([[0.3, 0.7], [0.6, 0.4]])
    logits2 = tf.math.log([[0.2, 0.8], [0.5, 0.5]])
    logits3 = tf.math.log([[0.4, 0.6], [0.4, 0.6]])

    labels = tf.convert_to_tensor([1, 0], dtype=tf.int32)

    cce.reset()
    cce.update(logits1, labels)
    cce.update(logits2, labels)
    cce.update(logits3, labels)
    ce = cce.result()

    self.assertAlmostEqual(-math.log(0.7), float(ce[0]), delta=TOL)
    self.assertAlmostEqual(-math.log(0.5), float(ce[1]), delta=TOL)

    ces = []
    gce = stats.ClassificationGibbsCrossEntropy()
    gce.reset()
    for logits in [logits1, logits2, logits3]:
      cce.reset()
      cce.update(logits, labels)
      ces.append(cce.result())

      gce.update(logits, labels)

    self.assertAllClose(
        tf.reduce_mean(tf.stack(ces, axis=0), axis=0),
        gce.result(),
        atol=TOL,
        msg="Gibbs cross entropy does not match mean CE.")


REGRESSION_MODEL_OUTPUT_TYPES = ["tensors", "dists"]


def NewRegressionModelOutputs(tensor_model_outputs, model_output_type="tensors",
                              outputs_with_log_stddevs=False, stddev=1.0):
  model_outputs = None
  if model_output_type == "tensors":
    model_outputs = tensor_model_outputs
  elif model_output_type == "dists":
    if outputs_with_log_stddevs:
      n_targets = tensor_model_outputs.shape[-1] // 2
      model_outputs = tfd.Normal(tensor_model_outputs[:, :, :n_targets],
                                 tf.exp(tensor_model_outputs[:, :, n_targets:]))
    else:
      model_outputs = tfd.Normal(tensor_model_outputs, stddev)
  else:
    raise Exception("Unknown model_output_type: {}".format(model_output_type))
  return model_outputs


class RegressionOutputsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(REGRESSION_MODEL_OUTPUT_TYPES))
  def test_regression_outputs_only_means_1d(self, model_output_type):
    tensor_model_outputs = tf.constant([
        [[0.3], [0.6]],  # Member 0, Example 0 and 1
        [[0.2], [0.5]],  # Member 1, Example 0 and 1
        [[0.4], [0.4]],  # Member 2, Example 0 and 1
    ])
    model_outputs = NewRegressionModelOutputs(tensor_model_outputs,
                                              model_output_type)

    ens_reg_outputs = stats.RegressionOutputs()
    ens_reg_outputs.update(model_outputs[0])
    ens_reg_outputs.update(model_outputs[1])
    ens_reg_outputs.update(model_outputs[2])
    means, variances = ens_reg_outputs.result()

    self.assertAlmostEqual(0.3, float(means[0][0]), delta=TOL)
    self.assertAlmostEqual(0.5, float(means[1][0]), delta=TOL)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.3, 0.2, 0.4],
        stddevs=[1.0, 1.0, 1.0])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[0][0]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.6, 0.5, 0.4],
        stddevs=[1.0, 1.0, 1.0])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[1][0]), delta=1e-5)

  @parameterized.parameters(itertools.product(REGRESSION_MODEL_OUTPUT_TYPES))
  def test_regression_outputs_only_means_2d_diff_stddev(self,
                                                        model_output_type):

    tensor_model_outputs = tf.constant([
        [[0.3, 0.4], [1.6, 0.6]],  # Member 0, Example 0 and 1
        [[0.2, 0.2], [0.8, 0.5]],  # Member 1, Example 0 and 1
        [[0.4, 0.6], [2.4, 0.4]],  # Member 2, Example 0 and 1
    ])

    model_outputs = NewRegressionModelOutputs(tensor_model_outputs,
                                              model_output_type,
                                              stddev=0.1)

    ens_reg_outputs = stats.RegressionOutputs(stddev=0.1)
    ens_reg_outputs.update(model_outputs[0])
    ens_reg_outputs.update(model_outputs[1])
    ens_reg_outputs.update(model_outputs[2])
    means, variances = ens_reg_outputs.result()

    self.assertAlmostEqual(0.3, float(means[0][0]), delta=TOL)
    self.assertAlmostEqual(0.4, float(means[0][1]), delta=TOL)
    self.assertAlmostEqual(1.6, float(means[1][0]), delta=TOL)
    self.assertAlmostEqual(0.5, float(means[1][1]), delta=TOL)

    # Expected mixture, does not have to use normal distributions
    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.3, 0.2, 0.4],
        stddevs=[0.1, 0.1, 0.1])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[0][0]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.4, 0.2, 0.6],
        stddevs=[0.1, 0.1, 0.1])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[0][1]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[1.6, 0.8, 2.4],
        stddevs=[0.1, 0.1, 0.1])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[1][0]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.6, 0.5, 0.4],
        stddevs=[0.1, 0.1, 0.1])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[1][1]), delta=1e-5)

  @parameterized.parameters(itertools.product(REGRESSION_MODEL_OUTPUT_TYPES))
  def test_regression_outputs_means_and_variances_2d(self, model_output_type):
    tensor_model_outputs = tf.constant([
        [  # member 0 tensor_model_outputs
            [0.3, 0.4, np.log(0.01), np.log(0.02)],  # Example 0
            [1.6, 0.6, np.log(2.0), np.log(0.01)],   # Example 1
        ],
        [  # member 1 tensor_model_outputs
            [0.2, 0.2, np.log(0.1), np.log(0.2)],  # Example 0
            [0.8, 0.5, np.log(0.5), np.log(0.2)],  # Example 1
        ],
        [  # member 2 tensor_model_outputs
            [0.4, 0.6, np.log(1.0), np.log(1.5)],  # Example 0
            [2.4, 0.4, np.log(0.05), np.log(0.1)],  # Example 1
        ]
    ])
    model_outputs = NewRegressionModelOutputs(tensor_model_outputs,
                                              model_output_type,
                                              outputs_with_log_stddevs=True)
    ens_reg_outputs = stats.RegressionOutputs(outputs_with_log_stddevs=True)
    ens_reg_outputs.update(model_outputs[0])  # Member 0 outputs
    ens_reg_outputs.update(model_outputs[1])  # Member 1 outputs
    ens_reg_outputs.update(model_outputs[2])  # Member 2 outputs
    means, variances = ens_reg_outputs.result()

    self.assertAlmostEqual(0.3, float(means[0][0]), delta=TOL)
    self.assertAlmostEqual(0.4, float(means[0][1]), delta=TOL)
    self.assertAlmostEqual(1.6, float(means[1][0]), delta=TOL)
    self.assertAlmostEqual(0.5, float(means[1][1]), delta=TOL)

    # Expected mixture, does not have to use normal distributions
    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.3, 0.2, 0.4],
        stddevs=[0.01, 0.1, 1.0])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[0][0]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.4, 0.2, 0.6],
        stddevs=[0.02, 0.2, 1.5])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[0][1]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[1.6, 0.8, 2.4],
        stddevs=[2.0, 0.5, 0.05])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[1][0]), delta=1e-5)

    expected_variance = self._get_mixture_variance(
        probs=[1 / 3, 1 / 3, 1 / 3],
        means=[0.6, 0.5, 0.4],
        stddevs=[0.01, 0.2, 0.1])
    self.assertAlmostEqual(
        float(expected_variance), float(variances[1][1]), delta=1e-5)

  @staticmethod
  def _get_mixture_variance(probs, means, stddevs):
    assert len(probs) == len(means) == len(stddevs)
    n = len(probs)
    components = []
    for i in range(n):
      components.append(tfd.Normal(loc=means[i], scale=stddevs[i]))
    mixture = tfd.Mixture(
        cat=tfd.Categorical(probs=probs), components=components)

    variance = mixture.variance()
    return variance


class RegressionNormalLogProbTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(REGRESSION_MODEL_OUTPUT_TYPES))
  def test_regression_normal_log_prob_means_and_stddevs_2d(self,
                                                           model_output_type):
    tensor_model_outputs = tf.constant([
        [[0.3, 0.4, np.log(0.01), np.log(0.02)],
         [1.6, 0.6, np.log(2.0), np.log(0.01)]],
        [[0.2, 0.2, np.log(0.1), np.log(0.2)],
         [0.8, 0.5, np.log(0.5), np.log(0.2)]],
        [[0.4, 0.6, np.log(1.0), np.log(1.5)],
         [2.4, 0.4, np.log(0.05), np.log(0.1)]],
    ])
    labels = tf.constant([[0.2, 0.4], [1.4, 1.0]])

    model_outputs = NewRegressionModelOutputs(tensor_model_outputs,
                                              model_output_type,
                                              outputs_with_log_stddevs=True)

    ens_reg_outputs = stats.RegressionOutputs(outputs_with_log_stddevs=True)
    ens_reg_outputs.update(model_outputs[0])
    ens_reg_outputs.update(model_outputs[1])
    ens_reg_outputs.update(model_outputs[2])
    means, variances = ens_reg_outputs.result()
    expected_nll = -tfd.Normal(means, variances**0.5).log_prob(labels)

    rnlls = stats.RegressionNormalLogProb(outputs_with_log_stddevs=True)
    rnlls.update(model_outputs[0], labels)
    rnlls.update(model_outputs[1], labels)
    rnlls.update(model_outputs[2], labels)
    nlls = rnlls.result()
    self.assertAllClose(expected_nll, nlls, atol=TOL)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()

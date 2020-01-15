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

# Lint as: python2, python3
"""Tests for metrics_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from uq_benchmark_2019 import metrics_lib


class MetricsLibTest(tf.test.TestCase, parameterized.TestCase):

  def test_bin_predictions_and_accuracies(self):
    num_samples = int(1e5)
    num_bins = 7
    probabilities = np.linspace(0, 1, num_samples)
    labels = np.random.random(num_samples) < probabilities

    bin_edges, accuracies, counts = metrics_lib.bin_predictions_and_accuracies(
        probabilities, labels, num_bins)

    bin_centers = metrics_lib.bin_centers_of_mass(probabilities, bin_edges)
    self.assertTrue((bin_centers > bin_edges[:-1]).all())
    self.assertTrue((bin_centers < bin_edges[1:]).all())
    self.assertAllClose(accuracies, bin_centers, atol=0.05)
    self.assertAllClose(np.ones(num_bins), num_bins * counts / num_samples,
                        atol=0.05)

  #
  # expected_calibration_error
  #

  def test_expected_calibration_error(self):
    np.random.seed(1)
    nsamples = 100
    probs = np.linspace(0, 1, nsamples)
    labels = np.random.rand(nsamples) < probs
    ece = metrics_lib.expected_calibration_error(probs, labels)
    bad_ece = metrics_lib.expected_calibration_error(probs / 2, labels)

    self.assertBetween(ece, 0, 1)
    self.assertBetween(bad_ece, 0, 1)
    self.assertLess(ece, bad_ece)

    bins = metrics_lib.get_quantile_bins(10, probs)
    quantile_ece = metrics_lib.expected_calibration_error(probs, labels, bins)
    bad_quantile_ece = metrics_lib.expected_calibration_error(
        probs / 2, labels, bins)

    self.assertBetween(quantile_ece, 0, 1)
    self.assertBetween(bad_quantile_ece, 0, 1)
    self.assertLess(quantile_ece, bad_quantile_ece)

  def test_expected_calibration_error_all_wrong(self):
    num_bins = 90
    ece = metrics_lib.expected_calibration_error(
        np.zeros(10), np.ones(10), bins=num_bins)
    self.assertAlmostEqual(ece, 1.)

    ece = metrics_lib.expected_calibration_error(
        np.ones(10), np.zeros(10), bins=num_bins)
    self.assertAlmostEqual(ece, 1.)

  def test_expected_calibration_error_all_right(self):
    num_bins = 90
    ece = metrics_lib.expected_calibration_error(
        np.ones(10), np.ones(10), bins=num_bins)
    self.assertAlmostEqual(ece, 0.)
    ece = metrics_lib.expected_calibration_error(
        np.zeros(10), np.zeros(10), bins=num_bins)
    self.assertAlmostEqual(ece, 0.)

  def test_expected_calibration_error_bad_input(self):
    with self.assertRaises(ValueError):
      metrics_lib.expected_calibration_error(np.ones(1), np.ones(1))
    with self.assertRaises(ValueError):
      metrics_lib.expected_calibration_error(np.ones(100), np.ones(1))
    with self.assertRaises(ValueError):
      metrics_lib.expected_calibration_error(np.ones(100), np.ones(100) * 0.5)

  #
  # Tests for multiclass functions.
  #

  def test_get_multiclass_predictions_and_correctness(self):
    multiclass_probs = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2],
                                 [0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
    labels = np.array([2, 0, 1, 0])
    (argmax_probs,
     is_correct) = metrics_lib.get_multiclass_predictions_and_correctness(
         multiclass_probs, labels)
    self.assertAllEqual(argmax_probs, [0.7, 0.5, 0.7, 0.5])
    self.assertAllEqual(is_correct, [True, True, False, False])

  def test_get_multiclass_predictions_and_correctness_error_cases(self):
    multiclass_probs = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2],
                                 [0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
    labels = np.array([2, 0, 1, 0])
    with self.assertRaises(ValueError):
      bad_multiclass_probs = multiclass_probs - 0.01
      metrics_lib.get_multiclass_predictions_and_correctness(
          bad_multiclass_probs, labels)
    with self.assertRaises(ValueError):
      metrics_lib.get_multiclass_predictions_and_correctness(
          bad_multiclass_probs[Ellipsis, None], labels)
    with self.assertRaises(ValueError):
      metrics_lib.get_multiclass_predictions_and_correctness(
          bad_multiclass_probs, labels[Ellipsis, None])

  def test_expected_calibration_error_multiclass(self):
    num_samples = int(1e4)
    num_classes = 5
    probabilities, labels = _make_perfectly_calibrated_multiclass(
        num_samples, num_classes)
    good_ece = metrics_lib.expected_calibration_error_multiclass(
        probabilities, labels)
    bad_ece = metrics_lib.expected_calibration_error_multiclass(
        np.fliplr(probabilities), labels)
    self.assertAllClose(good_ece, 0, atol=0.05)
    self.assertAllClose(bad_ece, 0.5, atol=0.05)

    good_ece_topk = metrics_lib.expected_calibration_error_multiclass(
        probabilities, labels, top_k=3)
    self.assertAllClose(good_ece_topk, 0, atol=0.05)

  @parameterized.parameters(1, 2, None)
  def test_expected_calibration_error_quantile_multiclass(self, top_k):
    bad_quantile_eces = {1: .5, 2: .25, None: .2}
    num_samples = int(1e4)
    num_classes = 5
    probabilities, labels = _make_perfectly_calibrated_multiclass(
        num_samples, num_classes)

    bins = metrics_lib.get_quantile_bins(10, probabilities, top_k=top_k)
    good_quantile_ece = metrics_lib.expected_calibration_error_multiclass(
        probabilities, labels, bins, top_k)
    bad_quantile_ece = metrics_lib.expected_calibration_error_multiclass(
        np.fliplr(probabilities), labels, bins, top_k)
    self.assertAllClose(good_quantile_ece, 0, atol=0.05)
    self.assertAllClose(bad_quantile_ece, bad_quantile_eces[top_k], atol=0.05)

  def test_accuracy_top_k(self):
    num_samples = 20
    num_classes = 10
    probs = np.random.rand(num_samples, num_classes)
    probs /= np.expand_dims(probs.sum(axis=1), axis=-1)
    probs = np.apply_along_axis(sorted, 1, probs)
    labels = np.tile(np.arange(num_classes), 2)
    top_2_accuracy = metrics_lib.accuracy_top_k(probs, labels, 2)
    top_5_accuracy = metrics_lib.accuracy_top_k(probs, labels, 5)
    self.assertEqual(top_2_accuracy, .2)
    self.assertEqual(top_5_accuracy, .5)

  #
  # Tests for Brier score, deomposition
  #

  def test_brier_scores(self):
    batch_shape = (2, 3)
    num_samples, num_classes = 99, 9
    logits = tf.random.uniform(batch_shape + (num_samples, num_classes))
    dist = tfp.distributions.Categorical(logits=logits)
    labels = dist.sample().numpy()
    probs = dist.probs_parameter().numpy()

    scores = metrics_lib.brier_scores(labels, probs=probs)
    # Check that computing from logits returns the same result.
    self.assertAllClose(scores, metrics_lib.brier_scores(labels, logits=logits))

    self.assertEqual(scores.shape, batch_shape + (num_samples,))

    def compute_brier(labels_, logits_):
      probs_ = tf.math.softmax(logits_, axis=1)
      _, nlabels = probs_.shape
      plabel = tf.reduce_sum(tf.one_hot(labels_, nlabels) * probs_, axis=1)
      brier = tf.reduce_sum(tf.square(probs_), axis=1) - 2.0 * plabel
      return tf.reduce_mean(brier)

    scores_avg = scores.mean(-1)
    for indices in np.ndindex(*batch_shape):
      score_i = compute_brier(labels[indices], logits[indices])
      self.assertAlmostEqual(score_i.numpy(), scores_avg[indices])

  def test_brier_decompositions(self):
    batch_shape = (2, 3)
    num_samples, num_classes = 99, 9
    logits = tf.random.uniform(batch_shape + (num_samples, num_classes))
    dist = tfp.distributions.Categorical(logits=logits)
    labels = dist.sample().numpy()
    probs = dist.probs_parameter().numpy()

    all_decomps = metrics_lib.brier_decompositions(labels, probs)
    self.assertEqual(all_decomps.shape, batch_shape + (3,))
    for indices in np.ndindex(*batch_shape):
      decomp_i = metrics_lib.brier_decomposition(labels[indices],
                                                 logits[indices])
      decomp_i = tf.stack(decomp_i, axis=-1).numpy()
      self.assertAllClose(decomp_i, all_decomps[indices])


def _make_perfectly_calibrated_multiclass(num_samples, num_classes):
  argmax_probabilities = np.linspace(1/num_classes, 1, num_samples)
  # Probs have uniform probability among non-selected class.
  probabilities = (1 - argmax_probabilities) / (num_classes - 1)
  probabilities = np.tile(probabilities[:, None], [1, num_classes])
  probabilities[:, 0] = argmax_probabilities
  labels = np.stack([np.random.choice(num_classes, p=p) for p in probabilities])
  return probabilities, labels

if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

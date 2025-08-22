# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for transferability metrics."""

from parameterized import parameterized
import tensorflow as tf

from stable_transfer.transferability import features_targets_utils
from stable_transfer.transferability import gbc
from stable_transfer.transferability import hscore
from stable_transfer.transferability import leep
from stable_transfer.transferability import logme
from stable_transfer.transferability import nleep


class LeepTest(tf.test.TestCase):
  """Test the transferability metric LEEP."""

  @parameterized.expand([(5,), (10,), (15,)])
  def testEdgeCasesLeepScore(self, target_classes):
    """Test the edge cases of the LEEP metric."""
    points = 3000
    source_classes = 30
    targets = tf.random.uniform(
        shape=(points,), maxval=target_classes, dtype=tf.int32, seed=123)
    targets = features_targets_utils.shift_target_labels(targets)
    perfect_transfer_predictions = tf.one_hot(targets, source_classes)
    random_predictions = tf.nn.softmax(
        tf.random.normal([points, source_classes]), axis=-1)

    perfect_leep_score = leep.get_leep_score(
        perfect_transfer_predictions, targets)
    random_leep_score = leep.get_leep_score(random_predictions, targets)

    # Highest possible leep score is ~ log(1) -> 0
    self.assertAlmostEqual(perfect_leep_score, 0)
    # Leep has a lower bound when predictions are random (uniform predictions)
    lower_bound_leep = tf.math.log(1/target_classes)  # Uniform predictions
    # Test that for random predictions leep is close to the lower bound
    self.assertAlmostEqual(random_leep_score, lower_bound_leep, delta=0.01)


class NLeepTest(tf.test.TestCase):
  """Test the transferability metric NLEEP."""

  @parameterized.expand([(5, None), (5, 0.8), (15, None), (15, 0.8),])
  def testNLeepScore(self, target_classes, variance_pca):
    """Test the NLEEP metric for different target classes and PCA reductions."""
    points = 3000
    features_dim = 15
    targets = tf.random.uniform(
        shape=(points,), maxval=target_classes, dtype=tf.int32, seed=123)
    targets = features_targets_utils.shift_target_labels(targets)
    perfect_transfer_features = tf.one_hot(targets, features_dim)
    random_features = tf.nn.softmax(
        tf.random.normal([points, features_dim]), axis=-1)

    if variance_pca:
      perfect_transfer_features = features_targets_utils.pca_reduction(
          perfect_transfer_features, n_components=variance_pca)
      random_features = features_targets_utils.pca_reduction(
          random_features, n_components=variance_pca)

    perfect_nleep_score = nleep.get_nleep_score(
        perfect_transfer_features, targets, target_classes * 5)
    random_nleep_score = nleep.get_nleep_score(
        random_features, targets, target_classes * 5)

    # Highest possible nleep score is ~ log(1) -> 0
    self.assertAlmostEqual(perfect_nleep_score.numpy(), 0)
    # Test that for random predictions nleep is < 0
    self.assertLess(random_nleep_score.numpy(), 0)


class HscoreTest(tf.test.TestCase):
  """Test the transferability metric H-score."""

  @parameterized.expand([(5, 16, None), (5, 32, 16), (10, 16, 0.8)])
  def testHscore(self, target_classes, features_dim, n_components):
    """Test the edge cases of the H-score metric."""
    points = 3000
    targets = tf.random.uniform(
        shape=(points,), maxval=target_classes, dtype=tf.int32, seed=123)
    random_features = tf.random.normal([points, features_dim])
    perfect_features = tf.one_hot(targets, features_dim)

    if n_components:
      perfect_features = features_targets_utils.pca_reduction(
          perfect_features, n_components=n_components)
      random_features = features_targets_utils.pca_reduction(
          random_features, n_components=n_components)

    high_hscore = hscore.get_hscore(perfect_features, targets)
    random_hscore = hscore.get_hscore(random_features, targets)

    # H-score has an upper bound given by the features_dim (see paper)
    self.assertLess(high_hscore, features_dim)
    # For random predictions we expect an h-score value of almost 0
    self.assertAlmostEqual(random_hscore, 0, delta=0.1)


class LogmeTest(tf.test.TestCase):
  """Test the transferability metric LogME."""

  @parameterized.expand([(5, 16, 3000), (15, 32, 3000), (15, 64, 60)])
  def testLogme(self, target_classes, features_dim, points):
    """Test the transferability metric LogME."""
    targets = tf.random.uniform(
        shape=(points,), maxval=target_classes, dtype=tf.int32, seed=123)
    targets = features_targets_utils.shift_target_labels(targets)

    random_features = tf.random.normal([points, features_dim])
    good_features = tf.one_hot(targets, features_dim) + random_features * 1e-3

    high_logme = logme.get_logme_score(good_features, targets)
    random_logme = logme.get_logme_score(random_features, targets)

    # Test that logme is higher for non-random features
    self.assertGreater(high_logme, random_logme)


class GbcTest(tf.test.TestCase):
  """Test the transferability metric GBC."""

  @parameterized.expand([
      (5, 16, None, 'spherical'),
      (5, 32, 16, 'spherical'),
      (5, 32, 16, 'diagonal'),
      (10, 16, 0.8, 'diagonal')])
  def testSimilarVSRandomFeatures(
      self, target_classes, features_dim, n_components, gaussian_type):
    points = 3000
    targets = tf.random.uniform(
        shape=(points,), maxval=target_classes, dtype=tf.int32, seed=123)
    random_features = tf.random.normal([points, features_dim])
    perfect_features = tf.one_hot(targets, features_dim)

    if n_components:
      perfect_features = features_targets_utils.pca_reduction(
          perfect_features, n_components=n_components)
      random_features = features_targets_utils.pca_reduction(
          random_features, n_components=n_components)

    high_gbc = gbc.get_gbc_score(perfect_features, targets, gaussian_type)
    random_gbc = gbc.get_gbc_score(random_features, targets, gaussian_type)

    # Test that GBC is higher for non-random features
    self.assertGreater(high_gbc, random_gbc)

  @parameterized.expand([
      (5, 16, None, 'spherical'),
      (5, 32, 16, 'spherical'),
      (5, 32, 16, 'diagonal'),
      (10, 16, 0.8, 'diagonal')])
  def testMinimumEdgeCase(
      self, target_classes, features_dim, n_components, gaussian_type):
    """Test the minimum edge case of the transferability metric."""
    num_points_per_class = 500
    targets = [tf.ones(num_points_per_class) * c for c in range(target_classes)]
    overlapping_features = [
        tf.random.normal(
            shape=[num_points_per_class, features_dim], mean=0, stddev=0.0001)
        for c in range(target_classes)]
    overlapping_features = tf.concat(overlapping_features, axis=0)
    if n_components:
      overlapping_features = features_targets_utils.pca_reduction(
          overlapping_features, n_components=n_components)
    targets = tf.concat(targets, axis=0)

    min_gbc_score = gbc.get_gbc_score(
        overlapping_features, targets, gaussian_type)
    # If the inter class features completely overlap, then the Bhattacharyya
    # distance should be almost 0 and the metric has a lower bound of:
    #  -(target_classes) * (target_classes - 1).
    # Proof: -(sum(exp^-d(c_i, c_j)) over every combo of (c_i, c_j) with i!=j
    #  if d(c_i, c_j) -> 0, then -(sum(exp^-d(c_i, c_j)) ->
    #  -> -(sum(1)) -> -(target_classes) * (target_classes - 1)
    # In practice the distance will be a small value but not completely zero,
    # therefore the metric will be slighlty higher than the lower bound.
    lower_bound_metric = -target_classes * (target_classes - 1)
    self.assertGreater(min_gbc_score, lower_bound_metric)
    # We test if we are close to the lower bound, given a threshold to balance
    # the fact that the Bhattacharyya distance is in practice higher than 0
    threshold = 0.2
    self.assertLess(
        (min_gbc_score - lower_bound_metric) / -lower_bound_metric, threshold)

  @parameterized.expand([
      (5, 16, None, 'spherical'),
      (5, 32, 16, 'spherical'),
      (5, 32, 16, 'diagonal'),
      (10, 16, 0.8, 'diagonal')])
  def testMaximumEdgeCase(
      self, target_classes, features_dim, n_components, gaussian_type):
    """Test the maximum edge case of the transferability metric."""
    num_points_per_class = 500
    targets = [tf.ones(num_points_per_class) * c for c in range(target_classes)]
    separable_features = [
        tf.random.normal(
            shape=[num_points_per_class, features_dim], mean=c**2, stddev=0.01)
        for c in range(target_classes)]
    separable_features = tf.concat(separable_features, axis=0)
    if n_components:
      separable_features = features_targets_utils.pca_reduction(
          separable_features, n_components=n_components)
    targets = tf.concat(targets, axis=0)

    max_gbc_score = gbc.get_gbc_score(
        separable_features, targets, gaussian_type)
    # If the inter class features are completely separable the Bhattacharyya
    # distance should go to infinite and the metric should go to 0 (exp^-dist).
    self.assertAlmostEqual(max_gbc_score, 0)


if __name__ == '__main__':
  tf.test.main()

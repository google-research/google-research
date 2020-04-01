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

"""Tests for the losses based on soft sorting operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from soft_sort import losses


class LossesTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(LossesTest, self).setUp()
    tf.random.set_seed(0)

  def test_accuracy(self):
    y_pred = tf.constant([[1.0, 2.0, 3.0, 4.0, 6.0, 5.0]])
    loss_fn = losses.SoftErrorLoss(topk=1.0, power=1.0)
    acc = loss_fn.accuracy([2], y_pred)
    self.assertGreaterEqual(acc, 0.0)
    self.assertLessEqual(acc, 1.0)
    acc2 = loss_fn.accuracy([4], y_pred)
    self.assertGreater(acc2, acc)

    # Test that topk gives the same accuracy for all top 3.
    loss_fn = losses.SoftErrorLoss(topk=3, power=1.0)
    acc = loss_fn.accuracy([3], y_pred)
    acc2 = loss_fn.accuracy([4], y_pred)
    self.assertAllClose(acc, acc2)
    self.assertAllClose(acc, tf.constant([1.0]))

    acc3 = loss_fn.accuracy([0], y_pred)
    self.assertLess(acc3, 1.0)

  def test_softerror(self):
    y_pred = tf.constant([[1.0, 2.0, 3.0, 4.0, 6.0, 5.0]])
    loss_fn = losses.SoftErrorLoss(topk=1.0, power=1.0)

    loss = loss_fn([2], y_pred)
    self.assertGreaterEqual(loss, 0.0)
    self.assertLessEqual(loss, 1.0)

    # A better prediction should lead to a lower loss.
    loss2 = loss_fn([4], y_pred)
    self.assertLessEqual(loss2, loss)


class RegressionLossesTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for the regression specific losses from the losses module."""

  def setUp(self):
    super(RegressionLossesTest, self).setUp()
    tf.random.set_seed(0)
    self._num_points = 20
    self._values = tf.range(0, self._num_points, dtype=tf.float32)
    self._y_pred = tf.random.shuffle(tf.reshape(self._values, (-1, 1)))
    self._y_true = tf.zeros((self._num_points,), dtype=tf.float32)

  @parameterized.named_parameters(
      ('quantile_10', 0.1, 1.0),
      ('quantile_50', 0.5, 2.0))
  def test_lqr(self, quantile, power):
    loss_fn = losses.LeastQuantileRegressionLoss(quantile=quantile, power=power)
    loss = loss_fn(self._y_true, self._y_pred)
    expected_loss = math.pow(
        self._values[int(quantile * self._num_points)], power)
    self.assertAllClose(loss, expected_loss, 0.2, 0.2)

  @parameterized.named_parameters(
      ('quantile_10', 0.1, 1.0),
      ('quantile_50', 0.5, 2.0))
  def test_soft_lqr(self, quantile, power):
    loss_fn = losses.SoftLeastQuantileRegressionLoss(
        quantile=quantile, power=power)
    loss = loss_fn(self._y_true, self._y_pred)
    expected_loss = math.pow(
        self._values[int(quantile * self._num_points)], power)
    self.assertAllClose(loss, expected_loss, 0.2, 0.2)

  @parameterized.named_parameters(
      ('hard_1_4', 0.1, 0.4, 1.0),
      ('hard_2_4', 0.2, 0.4, 2.0))
  def test_trimmed(self, start, end, power):
    loss_fn = losses.TrimmedRegressionLoss(
        start_quantile=start, end_quantile=end, power=power)
    loss = loss_fn(self._y_true, self._y_pred)
    start_index = int(start * self._num_points)
    end_index = int(end * self._num_points)
    selected = tf.pow(self._values[start_index:end_index], power)
    expected_loss = tf.math.reduce_mean(selected)
    self.assertAllClose(loss, expected_loss, 0.2, 0.2)

  @parameterized.named_parameters(
      ('soft_1_4', 0.1, 0.4, 2.0),
      ('soft_1_6', 0.1, 0.6, 1.0))
  def test_soft_trimmed(self, start, end, power):
    loss_fn = losses.SoftTrimmedRegressionLoss(
        start_quantile=start, end_quantile=end, power=power)
    loss = loss_fn(self._y_true, self._y_pred)
    start_index = int(start * self._num_points)
    end_index = int(end * self._num_points)
    selected = tf.pow(self._values[start_index:end_index], power)
    expected_loss = tf.math.reduce_mean(selected)
    self.assertAllClose(loss, expected_loss, 0.2, 0.2)

  @parameterized.named_parameters(
      ('soft_1_3', 0.1, 0.3),
      ('soft_0_2', 0.0, 0.2),
      ('soft_4_1', 0.4, 1.0),
      ('soft_0_1', 0.0, 1.0))
  def test_soft_trimmed_degenerated(self, start, end):
    """Tests several possible degenerated cases."""
    loss_fn = losses.SoftTrimmedRegressionLoss(
        start_quantile=start, end_quantile=end)
    try:
      loss_fn(self._y_true, self._y_pred)
    except ValueError:
      self.fail('SoftTrimmedRegressionLoss raised ValueError unexpectedly!')

if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

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

"""Tests for metrics."""

import numpy as np
import tensorflow as tf

from simulation_research.next_day_wildfire_spread.models import metrics


class MetricsTest(tf.test.TestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()
    self.y_true = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
    self.y_true_masked = np.array([[0., 1.], [-1., -1.]], dtype=np.float32)
    self.y_pred = np.array([[0., 1.], [1., 0.]], dtype=np.float32)

  def testAUC(self):
    """Checks that AUC is computed correctly."""
    metric = metrics.AUCWithMaskedClass()
    metric.update_state(self.y_true, self.y_pred)
    result = metric.result()
    self.assertEqual(result.shape, ())
    self.assertAllClose(result.numpy(), 0.5)

  def testAUCWithMask(self):
    """Checks that AUC is computed correctly on masked inputs."""
    metric = metrics.AUCWithMaskedClass()
    metric.update_state(self.y_true_masked, self.y_pred)
    result = metric.result()
    self.assertEqual(result.shape, ())
    self.assertAllClose(result.numpy(), 1.0)

  def testRecall(self):
    """Checks that recall is computed correctly."""
    metric = metrics.RecallWithMaskedClass()
    metric.update_state(self.y_true, self.y_pred)
    result = metric.result()
    self.assertEqual(result.shape, ())
    self.assertAllClose(result.numpy(), 0.5)

  def testRecallWithMask(self):
    """Checks that recall is computed correctly on masked inputs."""
    metric = metrics.RecallWithMaskedClass()
    metric.update_state(self.y_true_masked, self.y_pred)
    result = metric.result()
    self.assertEqual(result.shape, ())
    self.assertAllClose(result.numpy(), 1.0)

  def testPrecision(self):
    """Checks that precision is computed correctly."""
    metric = metrics.PrecisionWithMaskedClass()
    metric.update_state(self.y_true, self.y_pred)
    result = metric.result()
    self.assertEqual(result.shape, ())
    self.assertAllClose(result.numpy(), 0.5)

  def testPrecisionWithMask(self):
    """Checks that precision is computed correctly."""
    metric = metrics.PrecisionWithMaskedClass()
    metric.update_state(self.y_true_masked, self.y_pred)
    result = metric.result()
    self.assertEqual(result.shape, ())
    self.assertAllClose(result.numpy(), 1.0)


if __name__ == '__main__':
  tf.test.main()

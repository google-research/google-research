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

"""Tests for the metrics related to soft sorting operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from soft_sort import metrics


class MetricsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()
    tf.random.set_seed(0)

  def test_quantile_error(self):
    y_pred = tf.random.shuffle(
        tf.reshape(tf.range(0, 21, dtype=tf.float32), (-1, 1)))
    y_true = tf.zeros((21,), dtype=tf.float32)
    error = metrics.quantile_error(y_true, y_pred, 0.50)
    self.assertAllClose(error, 10.0)
    error = metrics.quantile_error(y_true, y_pred, 0.10)
    self.assertAllClose(error, 2.0)

  def test_trimmed_error(self):
    y_pred = tf.random.shuffle(
        tf.reshape(tf.range(0, 21, dtype=tf.float32), (-1, 1)))
    y_true = tf.zeros((21,), dtype=tf.float32)
    error = metrics.trimmed_error(y_true, y_pred, 0.10, 0.40)
    selected = list(range(2, 8))
    self.assertAllClose(error, np.mean(selected))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

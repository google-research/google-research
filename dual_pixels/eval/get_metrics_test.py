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

"""Tests for get_metrics."""
import numpy as np
import tensorflow.compat.v2 as tf

from dual_pixels.eval import get_metrics


class GetMetricsTest(tf.test.TestCase):

  def test_import(self):
    self.assertIsNotNone(get_metrics)

  def test_metrics(self):
    gt_depth = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 7.0, 8.0, 9.0]])
    gt_conf = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    a = -3.0
    b = 2.0
    pred = a * gt_depth + b
    # Perturb last element which should not contribute to the loss as it's
    # zero confidence.
    pred[-1, -1] = 124.0
    metrics = get_metrics.metrics(
        pred, gt_depth, gt_conf, crop_height=2, crop_width=4)
    # All losses should be zero.
    for loss_name in ['wmae', 'wrmse', 'spearman']:
      self.assertAllClose(metrics[loss_name], 0.0)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

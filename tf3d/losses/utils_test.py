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

"""Tests for ...tf3d.losses.utils."""

import math
import numpy as np
import tensorflow as tf
from tf3d.losses import utils


class UtilsTest(tf.test.TestCase):

  def test_sample_from_labels_balanced(self):
    labels = tf.constant([
        1, 3, 3, 3, 3, 3, 1, 2, 2, 2, 4, 5, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    ], dtype=tf.int32)
    samples = utils.sample_from_labels_balanced(labels=labels, num_samples=10)
    self.assertAllEqual(samples.shape, [10])

  def test_get_normalized_center_distances(self):
    predicted_boxes_center = tf.constant([[0.0, 0.0, 0.0],
                                          [1.0, 1.0, 1.0]], dtype=tf.float32)
    gt_boxes_center = tf.constant([[2.0, 2.0, 2.0],
                                   [5.0, -5.0, -5.0]], dtype=tf.float32)
    gt_boxes_length = tf.constant([1.0, 5.0], dtype=tf.float32)
    gt_boxes_height = tf.constant([2.0, 10.0], dtype=tf.float32)
    gt_boxes_width = tf.constant([2.0, 10.0], dtype=tf.float32)
    normalized_center_distances = utils.get_normalized_center_distances(
        predicted_boxes_center=predicted_boxes_center,
        gt_boxes_center=gt_boxes_center,
        gt_boxes_length=gt_boxes_length,
        gt_boxes_height=gt_boxes_height,
        gt_boxes_width=gt_boxes_width)
    self.assertAllClose(
        normalized_center_distances.numpy(),
        np.array([math.sqrt(12.0) / 3.0,
                  math.sqrt(88.0) / 15.0]))

  def test_get_balanced_loss_weights_foreground_background(self):
    labels = tf.constant([1, 1, 1, 1, 0, 0], dtype=tf.int32)
    weights = utils.get_balanced_loss_weights_foreground_background(
        labels=labels)
    self.assertAllClose(np.sum(weights.numpy()), np.array(6.0))
    self.assertAllClose(weights.numpy()[0] * 2.0, weights.numpy()[4])

  def test_get_balanced_loss_weights_multiclass(self):
    labels = tf.constant([1, 1, 1, 1, 2, 2, 3], dtype=tf.int32)
    weights = utils.get_balanced_loss_weights_multiclass(labels=labels)
    self.assertAllClose(np.sum(weights.numpy()), np.array(7.0))
    self.assertAllClose(weights.numpy()[0] * 2.0, weights.numpy()[4])
    self.assertAllClose(weights.numpy()[4] * 2.0, weights.numpy()[6])


if __name__ == '__main__':
  tf.test.main()

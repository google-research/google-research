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

"""Tests for metrics."""

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np

from imp.max.evaluation import metrics


class MetricsTest(absltest.TestCase):

  def test_accuracy(self):
    metric = metrics.Accuracy(top=1, average_logits=False)
    true = np.array([[0, 1], [1, 0]])
    pred = np.array([[0, 1], [0, 1]])
    accuracy = metric(pred, true)
    self.assertEqual(accuracy['top_1'], 0.5)

  def test_accuracy_jax(self):
    metric = metrics.Accuracy(top=1, average_logits=False).enable_jax_mode()
    true = np.array([[0, 1], [1, 0]])
    pred = np.array([[0, 1], [0, 1]])
    accuracy = metric(pred, true)
    self.assertEqual(accuracy['top_1'], 0.5)

  def test_accuracy_sequence(self):
    metric = metrics.Accuracy(top=1, average_logits=False)
    true = np.array([[[0, 1], [1, 0]]])
    pred = np.array([[[0, 1], [0, 1]]])
    accuracy = metric(pred, true)
    self.assertEqual(accuracy['top_1'], 0.5)

  def test_accuracy_with_mask(self):
    metric = metrics.Accuracy(top=1, average_logits=False)
    true = np.array([[[0, 1], [1, 0]]])
    pred = np.array([[[0, 1], [0, 1]]])
    mask = np.array([[1, 0]])
    accuracy = metric(pred, true, mask=mask)
    self.assertAlmostEqual(accuracy['top_1'], 1.)

  def test_accuracy_with_zero_mask(self):
    metric = metrics.Accuracy(top=1, average_logits=False)
    true = np.array([[[0, 1], [1, 0]]])
    pred = np.array([[[0, 1], [0, 1]]])
    mask = np.array([[0, 0]])
    accuracy = metric(pred, true, mask=mask)
    self.assertAlmostEqual(accuracy['top_1'], 0.)

  def test_accuracy_average_logits(self):
    metric = metrics.Accuracy(top=1, average_logits=True)
    true = np.array([[[0, 1]], [[1, 0]]])
    pred = np.array([[[0, 1], [0, 1]], [[1, 0], [1, 0]]])
    accuracy = metric(pred, true)
    self.assertEqual(accuracy['top_1'], 1.)

  def test_retrieval_recall(self):
    modality_1 = jnp.array(
        [[[0.396484, 0.761719, 0.761719, 0.158203]],
         [[0.192383, 0.0466309, 0.828125, 0.859375]],
         [[0.400391, 0.792969, 0.847656, 0.597656]],
         [[0.886719, 0.78125, 0.11377, 0.196289]],
         [[0.8125, 0.953125, 0.0810547, 0.0197754]],
         [[0.0154419, 0.402344, 0.96875, 0.15918]],
         [[0.847656, 0.503906, 0.902344, 0.396484]],
         [[0.259766, 0.632812, 0.503906, 0.333984]],
         [[0.361328, 0.617188, 0.835938, 0.523438]],
         [[0.242188, 0.490234, 0.984375, 0.804688]],
         [[0.792969, 0.933594, 0.322266, 0.416016]],
         [[0.392578, 0.839844, 0.515625, 0.388672]],
         [[0.0888672, 0.00750732, 0.285156, 0.523438]],
         [[0.625, 0.129883, 0.972656, 0.0957031]],
         [[0.578125, 0.796875, 0.714844, 0.223633]],
         [[0.382812, 0.769531, 0.785156, 0.652344]]], dtype=jnp.bfloat16)
    modality_2 = jnp.array(
        [[[2.76562, 0.863281, -0.400391, -2.26562],
          [2.84375, 0.375, -0.53125, -1.1875]],
         [[-2.40625, -1.21094, 2.09375, 2.54688],
          [0.691406, -0.0805664, 1.82812, 3.53125]],
         [[0.376953, -1.53125, 2.04688, 0.0473633],
          [-0.671875, 3.29688, 2.70312, 2.375]],
         [[-0.917969, 1.55469, 0.178711, 1.28125],
          [-0.957031, -3.89062, -0.155273, -0.777344]],
         [[-0.445312, 0.941406, 2.28125, -1.92969],
          [1.26562, 1.46875, 0.109863, -1.02344]],
         [[-1.36719, -1.9375, -0.478516, 0.382812],
          [0.628906, 0.804688, 0.824219, -1.92188]],
         [[2.0625, -1.29688, -0.460938, -1.44531],
          [2.6875, -1.85938, 3.26562, 2.26562]],
         [[-1.46094, -1.67969, -1.74219, -0.265625],
          [2.51562, -0.984375, -0.550781, 2.6875]],
         [[-1.05469, 1.26562, 0.138672, 1.0625],
          [6.46875, 3.95312, -2.46875, -0.474609]],
         [[1.33594, 1.42188, 0.714844, -0.546875],
          [1.59375, 0.074707, 1.67188, 0.871094]],
         [[2.375, 0.910156, 0.181641, -2.34375],
          [-1.95312, 2.03125, -2.15625, 0.203125]],
         [[0.644531, 0.116211, -0.308594, 1.57031],
          [-0.251953, -1.05469, 0.371094, -0.353516]],
         [[2.28125, 0.535156, -0.220703, 2.8125],
          [-0.417969, 0.363281, 1.91406, 0.800781]],
         [[1.54688, 0.550781, 0.867188, -3.98438],
          [0.910156, -0.988281, -1.53906, 0.605469]],
         [[-3.28125, 5.21875, 2.90625, 1.03906],
          [-1.66406, 1.57031, 0.800781, 0.726562]],
         [[1.0625, -3.82812, -1.125, 0.361328],
          [0.875, 0.341797, 0.378906, 2.14062]]], dtype=jnp.bfloat16)

    # assert all 'best'-based metrics are correct
    ret_best = metrics.RetrievalRecall(
        at=(1, 5, 10), instance_selection_method='best')
    ret_metrics_best = ret_best(modality_1, modality_2)

    self.assertAlmostEqual(ret_metrics_best['m1_vs_m2']['R1'], 0.125)
    self.assertAlmostEqual(ret_metrics_best['m2_vs_m1']['R1'], 0.0625)
    self.assertAlmostEqual(ret_metrics_best['m1_vs_m2']['R5'], 0.5625)
    self.assertAlmostEqual(ret_metrics_best['m2_vs_m1']['R5'], 0.4375)
    self.assertAlmostEqual(ret_metrics_best['m1_vs_m2']['R10'], 0.9375)
    self.assertAlmostEqual(ret_metrics_best['m2_vs_m1']['R10'], 0.75)
    self.assertAlmostEqual(ret_metrics_best['m1_vs_m2']['MedianRank'], 5.0)
    self.assertAlmostEqual(ret_metrics_best['m2_vs_m1']['MedianRank'], 7.0)

    # assert all 'boundary'-based metrics are correct
    ret_boundary = metrics.RetrievalRecall(
        at=(1, 5, 10), instance_selection_method='boundary')
    ret_metrics_boundary = ret_boundary(modality_1, modality_2)

    self.assertAlmostEqual(ret_metrics_boundary['m1_vs_m2']['R1'], 0.125)
    self.assertAlmostEqual(ret_metrics_boundary['m2_vs_m1']['R1'], 0.0625)
    self.assertAlmostEqual(ret_metrics_boundary['m1_vs_m2']['R5'], 0.4375)
    self.assertAlmostEqual(ret_metrics_boundary['m2_vs_m1']['R5'], 0.4375)
    self.assertAlmostEqual(ret_metrics_boundary['m1_vs_m2']['R10'], 0.5)
    self.assertAlmostEqual(ret_metrics_boundary['m2_vs_m1']['R10'], 0.75)
    self.assertAlmostEqual(ret_metrics_boundary['m1_vs_m2']['MedianRank'], 9.0)
    self.assertAlmostEqual(ret_metrics_boundary['m2_vs_m1']['MedianRank'], 7.0)


if __name__ == '__main__':
  absltest.main()

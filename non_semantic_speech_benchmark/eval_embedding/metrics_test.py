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

"""Tests for utilities defined in metrics.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from non_semantic_speech_benchmark.eval_embedding import metrics


class MetricsTest(parameterized.TestCase):

  def testCalculateEer(self):
    self.assertEqual(
        metrics.calculate_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0)
    self.assertEqual(
        metrics.calculate_eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 1)
    self.assertEqual(
        metrics.calculate_eer([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9]), 0.5)

  @parameterized.named_parameters(
      ('Perfect scores', [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], [1, 0.5, 0, 0, 0],
       [0, 0, 0, 0.5, 1.0]),
      ('Perfectly wrong', [0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1], [1, 1, 1, 0.5, 0],
       [0, 0.5, 1, 1, 1]),
      ('Fifty-fifty', [0.1, 0.8, 0.2, 0.9], [0, 0, 1, 1], [1, 0.5, 0.5, 0, 0],
       [0, 0, 0.5, 0.5, 1]),
  )
  def testCalculateDetCurve(self, scores, labels, expected_fpr, expected_fnr):
    fpr, fnr = metrics.calculate_det_curve(labels, scores)
    fpr = list(fpr)
    fnr = list(fnr)
    self.assertEqual(fpr, expected_fpr)
    self.assertEqual(fnr, expected_fnr)

  def testAUCSanity(self):
    metrics.calculate_auc([0, 0, 1, 1],
                          np.array([[0.1, 0.2, 0.8, 0.9],
                                    [0.9, 0.8, 0.2, 0.1]]).transpose())
    metrics.calculate_auc([0, 0, 1, 1],
                          np.array([[0.9, 0.8, 0.2, 0.1],
                                    [0.1, 0.2, 0.8, 0.9]]).transpose())
    metrics.calculate_auc([0, 1, 1, 2],
                          np.array([[0.1, 0.2, 0.7, 0.8],
                                    [0.5, 0.6, 0.1, 0.1],
                                    [0.4, 0.2, 0.2, 0.1]]).transpose(),
                          binary_classification=False)
    metrics.calculate_auc([0, 1, 1, 2],
                          np.array([[0.8, 0.7, 0.2, 0.1],
                                    [0.1, 0.1, 0.6, 0.5],
                                    [0.1, 0.2, 0.2, 0.4]]).transpose(),
                          binary_classification=False)
    metrics.calculate_auc([0, 1, 1, 2],
                          np.array([[0.1, 0.2, 0.7, 0.8],
                                    [0.5, 0.6, 0.1, 0.1],
                                    [0.4, 0.2, 0.2, 0.1]]).transpose(),
                          binary_classification=False,
                          multi_class='ovo')
    metrics.calculate_auc([0, 1, 1, 2],
                          np.array([[0.8, 0.7, 0.2, 0.1],
                                    [0.1, 0.1, 0.6, 0.5],
                                    [0.1, 0.2, 0.2, 0.4]]).transpose(),
                          binary_classification=False,
                          multi_class='ovo')

  def testDPrimeSanity(self):
    auc = metrics.calculate_auc([0, 0, 1, 1],
                                np.array([[0.1, 0.2, 0.8, 0.9],
                                          [0.9, 0.8, 0.2, 0.1]]).transpose())
    metrics.dprime_from_auc(auc)
    auc = metrics.calculate_auc([0, 0, 1, 1],
                                np.array([[0.9, 0.8, 0.2, 0.1],
                                          [0.1, 0.2, 0.8, 0.9]]).transpose())
    metrics.dprime_from_auc(auc)

  def testBalancedAccuracySanity(self):
    metrics.balanced_accuracy([0, 0, 1, 1, 0], [0, 1, 0, 1, 1])


if __name__ == '__main__':
  absltest.main()

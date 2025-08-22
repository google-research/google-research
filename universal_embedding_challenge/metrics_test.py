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

"""Tests for Universal Embedding Challenge metric computation."""

from absl.testing import absltest

from universal_embedding_challenge import metrics


def _CreateRetrievalSolution():
  """Creates retrieval solution to be used in tests.

  Returns:
    solution: Dict mapping test image ID to list of ground-truth image IDs.
  """
  return {
      '0123456789abcdef': ['fedcba9876543210', 'fedcba9876543220'],
      '0223456789abcdef': ['fedcba9876543210'],
      '0323456789abcdef': [
          'fedcba9876543230', 'fedcba9876543240', 'fedcba9876543250'
      ],
      '8e4b41721b2b9956': [
          '8d64bb75e71528e6', 'a067e09dd98af225', '9b22c5cc0891834e',
          '13c548a94626f68a', '2499d93fe22bcbd8', 'f55c2c3550b3ef46'
      ],
      '0423456789abcdef': ['fedcba9876543230'],
  }


def _CreateRetrievalPredictions():
  """Creates retrieval predictions to be used in tests.

  Returns:
    predictions: Dict mapping test image ID to a list with predicted index image
    ids.
  """
  return {
      '0223456789abcdef': ['fedcba9876543200', 'fedcba9876543210'],
      '0323456789abcdef': ['fedcba9876543240'],
      '8e4b41721b2b9956': [
          '8d64bb75e71528e6', 'a067e09dd98af225', '9b22c5cc0891834e',
          '13c548a94626f68a', 'fedcba9876543230', 'f55c2c3550b3ef46'
      ],
      '0423456789abcdef': ['fedcba9876543230', 'fedcba9876543240'],
  }


class MetricsTest(absltest.TestCase):

  def testCalibratedPrecisionTriggerUnknownQueryImage(self):
    # Define input.
    predictions = {'0223456789abcdej': ['fedcba9876543200', 'fedcba9876543210']}
    solution = _CreateRetrievalSolution()

    with self.assertRaisesRegex(
        ValueError,
        'Test image 0223456789abcdej is not part of retrieval_solution.'):
      metrics.CalibratedPrecision(predictions, solution)

  def testCalibratedPrecisionWorks(self):
    # Define input.
    predictions = _CreateRetrievalPredictions()
    solution = _CreateRetrievalSolution()

    # Run tested function.
    calibrated_precision_at_5 = metrics.CalibratedPrecision(
        predictions, solution)

    # Define expected results. The following is the calibrated_precision for
    # each query image:
    # 0223456789abcdef has 1 positive, but it is retrieved at position 2 -> 0
    # 0323456789abcdef has 3 positives, only 1 of them is retrieved at position
    # 1 -> 1/3
    # 8e4b41721b2b9956 has 6 positives, 4 of them is retrieved at rank 5 -> 4/5
    # 0423456789abcdef has 1 positive, and it is retrieved at position 1 -> 1
    # 0123456789abcdef has 2 positive, but it is not in the prediction output
    # The expected_calibrated_precision_at_5 = (0+1/3+4/5+1)/5
    expected_calibrated_precision_at_5 = 0.426667

    # Compare actual and expected results.
    self.assertAlmostEqual(calibrated_precision_at_5,
                           expected_calibrated_precision_at_5, 6)


if __name__ == '__main__':
  absltest.main()

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

from absl.testing import absltest
from cisc.src.post_processing import metrics_calculators


class MetricsCalculatorsTest(absltest.TestCase):

  def test_calculate_metrics_at_position_same_values(self):
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    evaluated = [0.1, 0.2, 0.3, 0.4, 0.5]
    evaluated_position = 3
    metrics = metrics_calculators.calculate_metrics_at_position(
        baseline, evaluated, evaluated_position
    )
    self.assertEqual(metrics.evaluated_position, 3)
    self.assertAlmostEqual(metrics.baseline_accuracy, 30)
    self.assertAlmostEqual(metrics.evaluated_accuracy, 30)
    self.assertEqual(metrics.comparable_baseline_traces_needed, 3)

  def test_calculate_metrics_at_position_different_values(self):
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    evaluated = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]
    evaluated_position = 3
    metrics = metrics_calculators.calculate_metrics_at_position(
        baseline, evaluated, evaluated_position
    )
    self.assertEqual(metrics.evaluated_position, 3)
    self.assertAlmostEqual(metrics.baseline_accuracy, 30)
    self.assertAlmostEqual(metrics.evaluated_accuracy, 60)
    self.assertEqual(metrics.comparable_baseline_traces_needed, 6)

  def test_calculate_metrics_at_position_above_all_baseline_traces(self):
    baseline = [0.36, 0.38, 0.43, 0.46, 0.48, 0.49]
    evaluated = [0.36, 0.44, 0.48, 0.5, 0.52, 0.54]
    evaluated_position = 5
    metrics = metrics_calculators.calculate_metrics_at_position(
        baseline, evaluated, evaluated_position
    )
    self.assertEqual(metrics.evaluated_position, 5)
    self.assertAlmostEqual(metrics.baseline_accuracy, 48)
    self.assertAlmostEqual(metrics.evaluated_accuracy, 52)
    self.assertEqual(metrics.comparable_baseline_traces_needed, 7)  # max + 1

  def test_calculate_metrics_at_position_baseline_is_better(self):
    baseline = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]
    evaluated = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    evaluated_position = 3
    metrics = metrics_calculators.calculate_metrics_at_position(
        baseline, evaluated, evaluated_position
    )
    self.assertEqual(metrics.evaluated_position, 3)
    self.assertAlmostEqual(metrics.baseline_accuracy, 60)
    self.assertAlmostEqual(metrics.evaluated_accuracy, 30)
    # 1 trace is enough.
    self.assertEqual(metrics.comparable_baseline_traces_needed, 1)

  def test_find_first_improvment_above_threshold(self):
    accuracies = [40, 60, 70, 71, 72]
    self.assertEqual(
        metrics_calculators.find_first_improvment_above_threshold(
            accuracies, 0.1
        ),
        2,
    )
    self.assertEqual(
        metrics_calculators.find_first_improvment_above_threshold(
            accuracies, 0.9
        ),
        3,
    )
    self.assertEqual(
        metrics_calculators.find_first_improvment_above_threshold(
            accuracies, 0.99
        ),
        5,
    )


if __name__ == "__main__":
  absltest.main()

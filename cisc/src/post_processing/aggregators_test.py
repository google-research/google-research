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
from cisc.src.post_processing import aggregators


class AggregatorsTest(absltest.TestCase):

  def test_majority(self):
    answers = ["A", "B", "C", "A", "A", "C"]
    ans, conf = aggregators.majority(answers)
    self.assertEqual(ans, "A")
    self.assertAlmostEqual(conf, 0.5)

  def test_majority_filters_empty_answers(self):
    answers = ["", "B", "C", "", "", "C"]
    ans, conf = aggregators.majority(answers)
    self.assertEqual(ans, "C")
    self.assertAlmostEqual(conf, 2 / 3)

  def test_majority_filters_all_empty_answers(self):
    answers = ["", ""]
    ans, conf = aggregators.majority(answers)
    self.assertEqual(ans, "")
    self.assertAlmostEqual(conf, 0)

  def test_majority_with_conf(self):
    answers = ["A", "A", "A", "B", "B", "C"]
    confidences = [6, 6, 6, 8, 8, 9]  # normalization -> [1, 1, 1, 3, 3, 4]
    ans, conf = aggregators.majority_with_conf(
        answers,
        confidences,
        norm_type=aggregators.NormalizationType.LINEAR,
    )
    self.assertEqual(ans, "B")
    self.assertAlmostEqual(conf, 6 / 13)

    # Softmax puts more emphasis on the higher confidences.
    ans, _ = aggregators.majority_with_conf(
        answers,
        confidences,
        norm_type=aggregators.NormalizationType.SOFTMAX,
        temp=1.0,
    )
    self.assertEqual(ans, "C")

    # Higher temperature puts more weight on repitions.
    ans, _ = aggregators.majority_with_conf(
        answers,
        confidences,
        norm_type=aggregators.NormalizationType.SOFTMAX,
        temp=2.0,
    )
    self.assertEqual(ans, "B")

  def test_majority_with_conf_zero(self):
    answers = ["A", "B"]
    confidences = [0, 0]  # normalization -> [1, 1]
    _, conf = aggregators.majority_with_conf(answers, confidences)
    self.assertAlmostEqual(conf, 1 / 2)

  def test_majority_with_conf_nan_conf(self):
    answers = ["A", "B"]
    confidences = [None, 0]
    ans, conf = aggregators.majority_with_conf(answers, confidences)
    self.assertEqual(ans, "B")
    self.assertAlmostEqual(conf, 1)

  def test_majority_with_conf_all_nan_conf(self):
    answers = ["A", "B"]
    confidences = [None, None]
    ans, conf = aggregators.majority_with_conf(answers, confidences)
    self.assertEqual(ans, "")
    self.assertAlmostEqual(conf, 0)

  def test_majority_with_conf_nan_ans(self):
    answers = [None, "B"]
    confidences = [2, 1]
    ans, conf = aggregators.majority_with_conf(answers, confidences)
    self.assertEqual(ans, "B")
    self.assertAlmostEqual(conf, 1)

  def test_majority_with_conf_all_nan_ans(self):
    answers = [None, None]
    confidences = [2, 1]
    ans, conf = aggregators.majority_with_conf(answers, confidences)
    self.assertEqual(ans, "")
    self.assertAlmostEqual(conf, 0)

  def test_max_confidence(self):
    answers = ["A", "B", "C", "A", "B", "C"]
    confidences = [1, 9, 3, 4, 5, 6]
    self.assertEqual(aggregators.max_confidence(answers, confidences), "B")

  def test_max_confidence_nan(self):
    answers = ["A", None, "C"]
    confidences = [1, 9, 3]
    self.assertEqual(aggregators.max_confidence(answers, confidences), "C")

  def test_remove_tile(self):
    answers = ["A", "A", "A", "B", "B", "C"]
    confidences = [1, 1, 1, 4, 5, 6]
    self.assertEqual(aggregators.remove_tile(answers, confidences, 50), "B")

  def test_sort_a_by_b(self):
    self.assertEqual(
        aggregators.sort_a_by_b(["A", "B", "C"], [1, 3, 2.2]), ["A", "C", "B"]
    )

  def test_is_in_any(self):
    answers = ["A", "B", "C", "A", "B", "C"]
    confidences = [1, 9, 3, 4, 5, 6]
    golden_label = "B"
    self.assertIsNone(
        aggregators.is_in_any(answers, confidences, golden_label, num_traces=1)
    )
    self.assertEqual(
        aggregators.is_in_any(answers, confidences, golden_label, num_traces=2),
        "B",
    )
    self.assertEqual(
        aggregators.is_in_any(
            answers,
            confidences,
            golden_label,
            num_traces=1,
            sort_by_confidence=True,
        ),
        "B",
    )

  def test_only_tie_break(self):
    answers = ["A", "A", "B"]
    confidences = [1, 1, 9]
    ans, conf = aggregators.majority_with_conf(
        answers,
        confidences,
        norm_type=aggregators.NormalizationType.LINEAR,
        only_tie_break=True,
    )
    self.assertEqual(ans, "A")
    self.assertAlmostEqual(conf, 2 / 3)


if __name__ == "__main__":
  absltest.main()

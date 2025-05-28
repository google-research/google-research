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

"""Tests for eval_lib."""

from absl.testing import absltest
import numpy as np

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import eval_lib
from assessment_plan_modeling.ap_parsing import tokenizer_lib


class SegmentationEvalLibTest(absltest.TestCase):

  def test_calc_metrics_usage(self):
    # example usage
    self.assertTrue(
        np.allclose(
            np.array(list(eval_lib.calc_metrics(1, 10, 20, "test").values())),
            np.array([0.05, 0.1, 1 / 15, 1 / 29, 1.0, 10.0, 20.0]),
            equal_nan=True))

  def test_calc_metrics_tp_zero(self):
    # zero tp
    self.assertTrue(
        np.allclose(
            np.array(list(eval_lib.calc_metrics(0, 10, 20, "test").values())),
            np.array([0.0, 0.0, np.nan, 0.0, 0.0, 10, 20]),
            equal_nan=True))

  def test_calc_metrics_truth_zero(self):
    # zero total true
    self.assertTrue(
        np.allclose(
            np.array(list(eval_lib.calc_metrics(1, 0, 20, "test").values())),
            np.array([0.05, np.nan, 0.0, 1 / 19, 1, 0, 20]),
            equal_nan=True))

  def test_span_level_metrics(self):
    # only truth, complete, partial, only pred
    truth_token_overlaps = [
        eval_lib.TokenOverlap(0),
        eval_lib.TokenOverlap(2),
        eval_lib.TokenOverlap(3),
    ]  # total tokens: 5

    truth_token_span_sizes = [2, 2, 4]  # total tokens: 8
    predicted_token_span_sizes = [2, 4, 1, 3]  # total tokens: 10

    # span_relaxed
    span_relaxed_metrics = [1 / 2, 2 / 3, 4 / 7, 0.4, 2, 3, 4]

    # example usage
    self.assertSequenceAlmostEqual(
        eval_lib.span_level_metrics(
            truth_token_overlaps, truth_token_span_sizes,
            predicted_token_span_sizes,
            ap_parsing_lib.LabeledSpanType.UNKNOWN_TYPE).values(),
        span_relaxed_metrics)

  def test_token_level_metrics(self):
    # only truth, complete, partial, only pred
    truth_token_labels = [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    pred_token_labels = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1]
    token_mask = [True] * len(truth_token_labels)

    # token_relaxed
    token_relaxed_metrics = [7 / 10, 7 / 9, 14 / 19, 7 / 12, 7, 9, 10]

    # example usage
    self.assertSequenceAlmostEqual(
        eval_lib.token_level_metrics(
            np.array(truth_token_labels), np.array(pred_token_labels),
            np.array(token_mask),
            ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE).values(),
        token_relaxed_metrics)

  def test_evaluate(self):
    # Char space:
    #       0         1          2          3
    #       01234567890123456 78901234 56789012345
    text = "# DM2: on insulin\n # COPD\n- nebs prn"
    # Token:012 3456 78       90123    4567   89
    #       0                  1
    tokens = tokenizer_lib.tokenize(text)

    truth_token_spans = [
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_token=2,
            end_token=4),  # span_text="DM2"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
            start_token=6,
            end_token=9),  # span_text="on insulin"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_token=13,
            end_token=14),  # span_text="COPD"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
            start_token=17,
            end_token=20),  # span_text="nebs prn"
    ]

    predicted_token_spans = [
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_token=2,
            end_token=9),  # span_text="DM2: on insulin"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_token=13,
            end_token=14),  # span_text="COPD"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
            start_token=17,
            end_token=18),  # span_text="- nebs"
        ap_parsing_lib.LabeledTokenSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
            start_token=18,
            end_token=20),  # span_text="prn"
    ]

    conf_mat = np.zeros((9, 9))
    conf_mat[1, 1] = 1

    expected_metrics = {
        # span_relaxed/PROBLEM_TITLE
        "span_relaxed/PROBLEM_TITLE/precision": 1,
        "span_relaxed/PROBLEM_TITLE/recall": 1,
        "span_relaxed/PROBLEM_TITLE/f1": 1,
        "span_relaxed/PROBLEM_TITLE/jaccard": 1,
        "span_relaxed/PROBLEM_TITLE/tp": 2,
        "span_relaxed/PROBLEM_TITLE/total_true": 2,
        "span_relaxed/PROBLEM_TITLE/total_pred": 2,
        # token_relaxed/PROBLEM_TITLE
        "token_relaxed/PROBLEM_TITLE/precision": 0.5,
        "token_relaxed/PROBLEM_TITLE/recall": 1,
        "token_relaxed/PROBLEM_TITLE/f1": 2 / 3,
        "token_relaxed/PROBLEM_TITLE/jaccard": 0.5,
        "token_relaxed/PROBLEM_TITLE/tp": 3,
        "token_relaxed/PROBLEM_TITLE/total_true": 3,
        "token_relaxed/PROBLEM_TITLE/total_pred": 6,
        # span_relaxed/PROBLEM_DESCRIPTION
        "span_relaxed/PROBLEM_DESCRIPTION/precision": np.nan,
        "span_relaxed/PROBLEM_DESCRIPTION/recall": 0,
        "span_relaxed/PROBLEM_DESCRIPTION/f1": np.nan,
        "span_relaxed/PROBLEM_DESCRIPTION/jaccard": 0,
        "span_relaxed/PROBLEM_DESCRIPTION/tp": 0,
        "span_relaxed/PROBLEM_DESCRIPTION/total_true": 1,
        "span_relaxed/PROBLEM_DESCRIPTION/total_pred": 0,
        # token_relaxed/PROBLEM_DESCRIPTION
        "token_relaxed/PROBLEM_DESCRIPTION/precision": np.nan,
        "token_relaxed/PROBLEM_DESCRIPTION/recall": 0,
        "token_relaxed/PROBLEM_DESCRIPTION/f1": np.nan,
        "token_relaxed/PROBLEM_DESCRIPTION/jaccard": 0,
        "token_relaxed/PROBLEM_DESCRIPTION/tp": 0,
        "token_relaxed/PROBLEM_DESCRIPTION/total_true": 2,
        "token_relaxed/PROBLEM_DESCRIPTION/total_pred": 0,
        # span_relaxed/ACTION_ITEM
        "span_relaxed/ACTION_ITEM/precision": 0.5,
        "span_relaxed/ACTION_ITEM/recall": 1,
        "span_relaxed/ACTION_ITEM/f1": 2 / 3,
        "span_relaxed/ACTION_ITEM/jaccard": 0.5,
        "span_relaxed/ACTION_ITEM/tp": 1,
        "span_relaxed/ACTION_ITEM/total_true": 1,
        "span_relaxed/ACTION_ITEM/total_pred": 2,
        # token_relaxed/ACTION_ITEM
        "token_relaxed/ACTION_ITEM/precision": 1,
        "token_relaxed/ACTION_ITEM/recall": 1,
        "token_relaxed/ACTION_ITEM/f1": 1,
        "token_relaxed/ACTION_ITEM/jaccard": 1,
        "token_relaxed/ACTION_ITEM/tp": 2,
        "token_relaxed/ACTION_ITEM/total_true": 2,
        "token_relaxed/ACTION_ITEM/total_pred": 2,
        # action_item_type/MEDICATIONS
        "action_item_type/MEDICATIONS/f1": 2 / 3,
        "action_item_type/MEDICATIONS/jaccard": 0.5,
        "action_item_type/MEDICATIONS/precision": 0.5,
        "action_item_type/MEDICATIONS/recall": 1,
        "action_item_type/MEDICATIONS/tp": 1,
        "action_item_type/MEDICATIONS/total_true": 1,
        "action_item_type/MEDICATIONS/total_pred": 2,
        "action_item_type/ALL/confusion_matrix": conf_mat,
    }
    calculated_metrics = eval_lib.evaluate_from_labeled_token_spans(
        truth_token_spans, predicted_token_spans, tokens=tokens)

    for k, v in expected_metrics.items():
      self.assertIn(k, calculated_metrics)
      self.assertTrue(
          np.allclose(v, calculated_metrics[k], equal_nan=True),
          msg=f"{k} - expected: {v} got {calculated_metrics[k]}")


if __name__ == "__main__":
  absltest.main()

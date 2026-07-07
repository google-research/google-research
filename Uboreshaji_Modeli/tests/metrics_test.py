# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import collections
from unittest import mock

from absl.testing import absltest
import numpy as np

from Uboreshaji_Modeli.common import metrics

EvalPrediction = collections.namedtuple(
    "EvalPrediction", ["predictions", "label_ids"]
)


class MetricsTest(absltest.TestCase):

  def _create_eval_prediction(
      self,
      logits,
      boxes,
      num_gt_boxes,
      gt_boxes,
      class_labels,
  ):
    """Helper to create EvalPrediction objects with correct numpy shapes."""
    boxes_arr = np.array([boxes], dtype=np.float32)

    if num_gt_boxes == 0:
      gt_boxes_arr = np.array([[]], dtype=np.float32).reshape(1, 0, 4)
      gt_classes_arr = np.array([[]], dtype=np.int64).reshape(1, 0)
    else:
      gt_boxes_arr = np.array([gt_boxes], dtype=np.float32)
      gt_classes_arr = np.array([class_labels], dtype=np.int64)

    return EvalPrediction(
        predictions=(np.array([logits], dtype=np.float32), boxes_arr),
        label_ids={
            "num_boxes": np.array([num_gt_boxes]),
            "boxes": gt_boxes_arr,
            "class_labels": gt_classes_arr,
        },
    )

  def test_perfect_match_single_box(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=1,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2]],
        class_labels=[0],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 1.0)
    self.assertAlmostEqual(result["map"], 1.0)
    self.assertAlmostEqual(result["mar_100"], 1.0)

  def test_perfect_match_multiple_boxes(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0], [-10.0, 10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]],
        num_gt_boxes=2,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]],
        class_labels=[0, 1],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 1.0)
    self.assertAlmostEqual(result["map"], 1.0)
    self.assertAlmostEqual(result["mar_100"], 1.0)

  def test_zero_match_wrong_class(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=1,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2]],
        class_labels=[1],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 0.0)

  def test_low_confidence_predictions_filtered_out(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.9
    )
    eval_pred = self._create_eval_prediction(
        logits=[[1.0, -1.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=1,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2]],
        class_labels=[0],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 0.0)

  def test_empty_ground_truth_returns_undefined(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=0,
        gt_boxes=[],
        class_labels=[],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], -1.0)

  @mock.patch("evaluate.load")
  def test_asr_wer(self, mock_load):
    tokenizer = mock.Mock()
    tokenizer.batch_decode.side_effect = [
        ["predicted string"],
        ["reference string"],
    ]
    tokenizer.pad_token_id = 0

    mock_wer = mock.Mock()
    mock_wer.compute.return_value = 0.5
    mock_load.return_value = mock_wer

    compute_metrics = metrics.create_asr_compute_metrics_fn(tokenizer)

    eval_pred = EvalPrediction(
        predictions=np.array([[1, 2]]), label_ids=np.array([[3, 4]])
    )

    result = compute_metrics(eval_pred)

    self.assertEqual(result["wer"], 0.5)
    mock_wer.compute.assert_called_once_with(
        predictions=["predicted string"], references=["reference string"]
    )

    # Verify skip_special_tokens=True was passed to batch_decode
    self.assertEqual(tokenizer.batch_decode.call_count, 2)
    for _, kwargs in tokenizer.batch_decode.call_args_list:
      self.assertTrue(kwargs.get("skip_special_tokens"))

  @mock.patch("evaluate.load")
  def test_ast_bleu(self, mock_load):
    tokenizer = mock.Mock()
    tokenizer.batch_decode.side_effect = [
        ["predicted string"],
        ["reference string"],
    ]
    tokenizer.pad_token_id = 0

    mock_bleu = mock.Mock()
    mock_bleu.compute.return_value = {"score": 42.0}
    mock_load.return_value = mock_bleu

    compute_metrics = metrics.create_ast_compute_metrics_fn(tokenizer)

    eval_pred = EvalPrediction(
        predictions=np.array([[1, 2]]), label_ids=np.array([[3, 4]])
    )

    result = compute_metrics(eval_pred)

    self.assertEqual(result["bleu"], 42.0)
    mock_bleu.compute.assert_called_once_with(
        predictions=["predicted string"], references=[["reference string"]]
    )

    # Verify skip_special_tokens=True was passed to batch_decode
    self.assertEqual(tokenizer.batch_decode.call_count, 2)
    for _, kwargs in tokenizer.batch_decode.call_args_list:
      self.assertTrue(kwargs.get("skip_special_tokens"))


class FormatForPublisherTest(absltest.TestCase):
  """Tests for format_for_publisher."""

  def test_overall_uses_prefix(self):
    result = metrics.format_for_publisher(
        eval_results={
            "best_eval_map": 0.5,
            "best_eval_map_50": 0.7,
        },
        label_names=["a"],
        model_label2id={"a": 0},
        prefix="best_eval",
    )
    results = result["evaluation_metrics"]["detection_metrics"]

    self.assertAlmostEqual(
        results["overall_metrics"]["ap_overall"], 0.5)
    self.assertAlmostEqual(
        results["overall_metrics"]["ap_50"], 0.7)

  def test_overall_skips_missing_keys(self):
    result = metrics.format_for_publisher(
        eval_results={"eval_map_50": 0.7},
        label_names=["a"],
        model_label2id={"a": 0},
        prefix="eval",
    )
    results = result["evaluation_metrics"]["detection_metrics"]

    self.assertIn("ap_50", results["overall_metrics"])
    self.assertNotIn("ap_overall", results["overall_metrics"])

  def test_per_label_extraction(self):
    result = metrics.format_for_publisher(
        eval_results={
            "eval_map_50_class_0": 0.8,
            "eval_map_50_class_1": 0.6,
        },
        label_names=["a", "b"],
        model_label2id={"a": 0, "b": 1},
        prefix="eval",
    )
    results = result["evaluation_metrics"]["detection_metrics"]

    self.assertAlmostEqual(
        results["per_label_metrics"]["a"]["ap_50"], 0.8)
    self.assertAlmostEqual(
        results["per_label_metrics"]["b"]["ap_50"], 0.6)

  def test_per_label_skips_missing_classes(self):
    result = metrics.format_for_publisher(
        eval_results={"eval_map_50_class_0": 0.9},
        label_names=["a", "b"],
        model_label2id={"a": 0, "b": 1},
        prefix="eval",
    )
    results = result["evaluation_metrics"]["detection_metrics"]

    self.assertIn("a", results["per_label_metrics"])
    self.assertNotIn("b", results["per_label_metrics"])

  def test_train_metrics_merged(self):
    result = metrics.format_for_publisher(
        eval_results={"eval_map": 0.5},
        label_names=["a"],
        model_label2id={"a": 0},
        train_metrics={
            "status": "COMPLETED",
            "total_steps": 100,
        },
    )

    self.assertEqual(result["status"], "COMPLETED")
    self.assertEqual(result["total_steps"], 100)

  def test_train_metrics_none(self):
    result = metrics.format_for_publisher(
        eval_results={"eval_map": 0.5},
        label_names=["a"],
        model_label2id={"a": 0},
        train_metrics=None,
    )

    self.assertNotIn("status", result)

  def test_flat_numeric_results_forwarded(self):
    result = metrics.format_for_publisher(
        eval_results={
            "eval_map": 0.5,
            "eval_runtime": 12.3,
        },
        label_names=["a"],
        model_label2id={"a": 0},
    )

    self.assertAlmostEqual(result["eval_map"], 0.5)
    self.assertAlmostEqual(result["eval_runtime"], 12.3)

  def test_non_numeric_results_excluded(self):
    result = metrics.format_for_publisher(
        eval_results={  # pyrefly: ignore[bad-argument-type]
            "eval_map": 0.5,
            "eval_note": "some string",
        },
        label_names=["a"],
        model_label2id={"a": 0},
    )

    self.assertIn("eval_map", result)
    self.assertNotIn("eval_note", result)

  def test_label_names_forwarded(self):
    result = metrics.format_for_publisher(
        eval_results={},
        label_names=["a", "b", "c"],
        model_label2id={
            "a": 0, "b": 1, "c": 2,
        },
    )

    self.assertEqual(result["label_names"], ["a", "b", "c"])


if __name__ == "__main__":
  absltest.main()

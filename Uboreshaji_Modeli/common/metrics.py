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

"""Metrics for object detection."""

from typing import Any, Callable, Mapping, MutableMapping, Sequence

import evaluate
import immutabledict
import torch
from torchmetrics.detection import mean_ap

from Uboreshaji_Modeli.common import box_utils

_TORCHMETRICS_TO_PUBLISHER = immutabledict.immutabledict({
    "map": "ap_overall",
    "map_50": "ap_50",
    "map_75": "ap_75",
    "map_small": "ap_s",
    "map_medium": "ap_m",
    "map_large": "ap_l",
    "mar_100": "ar_100",
    "mar_small": "ar_s",
    "mar_medium": "ar_m",
    "mar_large": "ar_l",
})


def create_compute_metrics_fn(
    resize_to, score_threshold = 0.0
):
  """Creates compute_metrics function for HF Trainer."""
  map_metric = mean_ap.MeanAveragePrecision(
      box_format="xyxy",
      max_detection_thresholds=[10, 100, 1000],
      class_metrics=True,
  )

  def compute_metrics(eval_pred):
    logits_arr, boxes_arr = eval_pred.predictions
    labels_dict = eval_pred.label_ids
    map_metric.reset()

    all_logits = torch.from_numpy(logits_arr).float()
    all_boxes = torch.from_numpy(boxes_arr).float()

    preds, targets = [], []

    for i, logits in enumerate(all_logits):
      n_tgt = int(labels_dict["num_boxes"][i].item())

      probs = logits.sigmoid()
      scores, pred_labels = probs.max(-1)

      keep = scores > score_threshold

      pred_boxes = box_utils.box_cxcywh_to_xyxy(all_boxes[i][keep]) * resize_to

      preds.append({
          "boxes": pred_boxes,
          "scores": scores[keep],
          "labels": pred_labels[keep],
      })

      tgt_boxes_raw = torch.from_numpy(labels_dict["boxes"][i, :n_tgt]).float()
      tgt_boxes = box_utils.box_cxcywh_to_xyxy(tgt_boxes_raw) * resize_to

      targets.append({
          "boxes": tgt_boxes,
          "labels": (
              torch.from_numpy(labels_dict["class_labels"][i, :n_tgt]).long()
          ),
      })

    map_metric.update(preds, targets)
    results = map_metric.compute()

    computed_metrics = {
        "map": results["map"].item(),
        "map_50": results["map_50"].item(),
        "map_75": results["map_75"].item(),
        "map_small": results["map_small"].item(),
        "mar_100": results["mar_100"].item(),
        "mar_small": results["mar_small"].item(),
        "mar_medium": results["mar_medium"].item(),
        "mar_large": results["mar_large"].item(),
    }

    return computed_metrics
  return compute_metrics


def create_asr_compute_metrics_fn(
    tokenizer,
):
  """Creates compute_metrics function for ASR (WER)."""
  wer_metric = evaluate.load("wer")

  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
      predictions = predictions[0]

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Handle label masking
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

  return compute_metrics


def create_ast_compute_metrics_fn(
    tokenizer,
):
  """Creates compute_metrics function for AST (BLEU)."""
  bleu_metric = evaluate.load("sacrebleu")

  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
      predictions = predictions[0]

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Handle label masking
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    references = [[ref] for ref in label_str]

    results = bleu_metric.compute(predictions=pred_str, references=references)

    return {"bleu": results["score"]}  # pyrefly: ignore[unsupported-operation]

  return compute_metrics


def format_for_publisher(
    *,
    eval_results,
    label_names,
    model_label2id,
    train_metrics = None,
    prefix = "eval",
    metrics_category = "detection_metrics",
):
  """Builds a unified evaluation dict for both publisher and result collection.

  Args:
    eval_results: Flat dict from compute_metrics (keys prefixed with `prefix`).
    label_names: Ordered list of class names used during training.
    model_label2id: Mapping from label name to integer class id.
    train_metrics: Optional dict with training metadata (e.g. train_loss,
      total_steps, wall_clock_seconds, status).
    prefix: Prefix used for evaluation keys (e.g., "eval" or "best_eval").
    metrics_category: Category used for nesting evaluation block (e.g.
      "detection_metrics", "text_sft_metrics", or "speech_metrics").

  Returns:
    Dict containing flat metrics at the top level (for collect_results.py)
    and the nested evaluation_metrics block (for parse_evaluation).
  """
  if metrics_category == "detection_metrics":
    overall = {
        pub_key: eval_results[f"{prefix}_{tm_key}"]
        for tm_key, pub_key in _TORCHMETRICS_TO_PUBLISHER.items()
        if f"{prefix}_{tm_key}" in eval_results
    }
    per_label = {
        name: {"ap_50": eval_results[f"{prefix}_map_50_class_{class_idx}"]}
        for name, class_idx in model_label2id.items()
        if f"{prefix}_map_50_class_{class_idx}" in eval_results
    }
    category_block = {
        metrics_category: {
            "overall_metrics": overall,
            "per_label_metrics": per_label,
        }
    }
  else:
    overall = {
        k.replace(f"{prefix}_", ""): v
        for k, v in eval_results.items()
        if isinstance(v, (int, float))
    }
    category_block = {
        metrics_category: {
            "overall_metrics": overall,
        }
    }

  return {
      **(train_metrics if train_metrics else {}),
      **{k: v for k, v in eval_results.items() if isinstance(v, (int, float))},
      "label_names": label_names,
      "evaluation_metrics": category_block,
  }

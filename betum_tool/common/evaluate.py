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

"""Standardized evaluation utility for YOLO/COCO predictions."""

import json
import os
import pathlib
from typing import Any

import numpy as np
from pycocotools import coco
from pycocotools import cocoeval

COCOeval = cocoeval.COCOeval
COCO = coco.COCO
Path = pathlib.Path


def _load_coco_gt(gt_json):
  """Loads ground-truth COCO object, checking type validity."""
  if isinstance(gt_json, (str, Path)):
    return COCO(str(gt_json))
  elif isinstance(gt_json, COCO):
    return gt_json
  else:
    raise TypeError("gt_json must be a file path or a COCO instance.")


def _load_predictions_list(
    predictions,
):
  """Parses predictions parameter into a list of prediction dictionaries."""
  if isinstance(predictions, (str, Path)):
    if not os.path.exists(predictions) or os.path.getsize(predictions) == 0:
      raise FileNotFoundError(
          f"Predictions file not found or empty: {predictions}"
      )
    with open(predictions, "r") as f:
      return json.load(f)
  elif isinstance(predictions, list):
    return predictions
  elif isinstance(predictions, COCO):
    return list(predictions.anns.values())
  else:
    raise TypeError(
        "predictions must be a file path, a list of dictionaries, or a COCO"
        " instance."
    )


def _slice_precision_matrix(
    precision_matrix,
    cat_idx,
    max_dets_idx,
):
  """Slices the COCOeval precision tensor to retrieve precision scores.

  The precision matrix computed by COCOeval is a 5D numpy array with the shape
  [T, R, K, A, M], representing:
  - T: IoU thresholds [0.50:0.95:0.05]. Index 0 references IoU=0.50 (AP@50).
  - R: Recall thresholds (typically 101 levels from 0.0 to 1.0).
  - K: Class/category index inside the COCO evaluation list.
  - A: Area ranges [all, small, medium, large], where:
       * Index 0 ('all'): all object sizes.
       * Index 1 ('small'): area < 32^2 square pixels.
       * Index 2 ('medium'): 32^2 <= area <= 96^2 square pixels.
       * Index 3 ('large'): area > 96^2 square pixels.
  - M: Max detections per image index (e.g. index of maxDets=100).

  Args:
    precision_matrix: A 5-dimensional numpy array of evaluations.
    cat_idx: The 0-based category index (K) in the evaluations list.
    max_dets_idx: The index (M) corresponding to the max detections count.

  Returns:
    A 1-dimensional numpy array of shape (101,) containing precision values
    across different recall levels.
  """
  return precision_matrix[0, :, cat_idx, 0, max_dets_idx]


def _compute_per_class_ap50(
    coco_eval,
    class_mapping_int,
):
  """Slices COCOeval precision tensor to calculate per-class AP@50."""
  precision_matrix = coco_eval.eval["precision"]
  cat_ids = coco_eval.params.catIds

  per_class_ap50 = {}
  for cat_id in cat_ids:
    class_name = class_mapping_int.get(cat_id, f"Category {cat_id}")
    k = cat_ids.index(cat_id)

    # Get index of max detections (usually [1, 10, 100] -> index 2)
    m_idx = len(coco_eval.params.maxDets) - 1
    prec = _slice_precision_matrix(precision_matrix, k, m_idx)

    if np.all(prec == -1):
      ap50 = -1.0
    else:
      ap50 = float(np.mean(prec[prec > -1]))

    per_class_ap50[class_name] = round(ap50, 4)
  return per_class_ap50


def _print_summary_report(
    agg_ap,
    agg_ap50,
    agg_ap75,
    cat_ids,
    class_mapping_int,
    per_class_ap50,
):
  """Prints a beautiful, formatted console report of the evaluation."""
  print("\n" + "=" * 65)
  print(f" {'UNIFIED COCO EVALUATION REPORT':^63}")
  print("=" * 65)
  print(f"  Aggregate AP @ [IoU=0.50:0.95] : {agg_ap:.4f}")
  print(f"  Aggregate AP@50 (IoU=0.50)     : {agg_ap50:.4f}")
  print(f"  Aggregate AP@75 (IoU=0.75)     : {agg_ap75:.4f}")
  print("-" * 65)
  print(f" | {'Category ID':<13} | {'Category Name':<24} | {'AP@50':<15} |")
  print("-" * 65)
  for cat_id in sorted(cat_ids):
    class_name = class_mapping_int.get(cat_id, f"Category {cat_id}")
    ap50_val = per_class_ap50.get(class_name, -1.0)
    ap50_str = f"{ap50_val:.4f}" if ap50_val >= 0 else "N/A (No GT)"
    print(f" | {cat_id:<13} | {class_name:<24} | {ap50_str:<15} |")
  print("=" * 65 + "\n")


def evaluate_coco(
    gt_json,
    predictions,
    score_threshold = None,
    class_mapping = None,
    verbose = True,
):
  """Runs standardized COCO evaluation and computes both aggregate and per-class AP@50.

  Args:
      gt_json: Path to the ground-truth COCO JSON file, or an initialized COCO
        instance.
      predictions: Path to predictions JSON file, a list of prediction dicts, or
        an initialized COCO instance.
      score_threshold: Optional threshold to filter out predictions with low
        confidence score.
      class_mapping: Optional mapping from category integer ID to class name
        string. If not provided, it will be automatically parsed from `gt_json`.
      verbose: If True, prints a beautifully formatted evaluation report.

  Returns:
      A dictionary containing the overall AP, AP@50, AP@75, and a nested
      dictionary of per-class AP@50:
      {
          "AP": float,
          "AP@50": float,
          "AP@75": float,
          "per_class_ap50": {
              "class_name_or_id": float,
              ...
          }
      }
  """
  # 1. Load Ground Truth and Predictions
  coco_gt = _load_coco_gt(gt_json)
  pred_list = _load_predictions_list(predictions)

  # 2. Filter predictions by score threshold if provided
  if score_threshold is not None:
    pred_list = [p for p in pred_list if p.get("score", 1.0) >= score_threshold]

  # 3. If no predictions left, return empty metrics
  if not pred_list:
    if verbose:
      print("\n" + "!" * 55)
      print(
          "Warning: No predictions found above threshold"
          f" {score_threshold or 0.0}!"
      )
      print("!" * 55 + "\n")

    if class_mapping is None:
      class_mapping = {cat["id"]: cat["name"] for cat in coco_gt.cats.values()}
    per_class = {name: 0.0 for name in class_mapping.values()}

    return {
        "AP": 0.0,
        "AP@50": 0.0,
        "AP@75": 0.0,
        "per_class_ap50": per_class,
    }

  # 4. Initialize COCO DT representation
  if isinstance(predictions, (str, Path)):
    coco_dt = coco_gt.loadRes(str(predictions))
  else:
    coco_dt = coco_gt.loadRes(pred_list)

  # 5. Run COCO evaluation
  coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

  # Extract aggregate stats
  agg_ap = round(float(coco_eval.stats[0]), 2)
  agg_ap50 = round(float(coco_eval.stats[1]), 2)
  agg_ap75 = round(float(coco_eval.stats[2]), 2)

  # 6. Parse category / class mapping
  if class_mapping is None:
    class_mapping = {cat["id"]: cat["name"] for cat in coco_gt.cats.values()}
  class_mapping_int = {int(k): str(v) for k, v in class_mapping.items()}

  # 7. Compute per-class AP@50
  per_class_ap50 = _compute_per_class_ap50(coco_eval, class_mapping_int)

  # 8. Print beautiful summary report if verbose
  if verbose:
    _print_summary_report(
        agg_ap,
        agg_ap50,
        agg_ap75,
        coco_eval.params.catIds,
        class_mapping_int,
        per_class_ap50,
    )

  return {
      "AP": agg_ap,
      "AP@50": agg_ap50,
      "AP@75": agg_ap75,
      "per_class_ap50": per_class_ap50,
  }

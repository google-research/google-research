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

r"""Converts raw OWL-v2 outputs to COCO predictions JSON format.

OWL-v2 outputs bounding boxes in normalised [cx, cy, w, h] format
relative to the longest image edge. This script converts them to
COCO absolute [x_min, y_min, width, height] format and maps text
query indices to COCO category IDs.

Can be used as a library (import the functions) or as a CLI:

    python parse_output.py \
        --input raw_predictions.json \
        --output coco_predictions.json
"""

import argparse
import json
from typing import Any


def owlv2_box_to_coco(
    cx,
    cy,
    w,
    h,
    img_width,
    img_height,
):
  """Converts a single OWL-v2 normalised box to COCO absolute format.

  OWL-v2 normalises coordinates by the longest edge of the image
  (not independently by width/height). This function reverses that.

  Args:
    cx: Normalised center-x (0–1, relative to max(H, W)).
    cy: Normalised center-y (0–1, relative to max(H, W)).
    w: Normalised width (0–1, relative to max(H, W)).
    h: Normalised height (0–1, relative to max(H, W)).
    img_width: Original image width in pixels.
    img_height: Original image height in pixels.

  Returns:
    COCO-format bbox: [x_min, y_min, width, height] in absolute pixels.
  """
  max_side = max(img_width, img_height)
  abs_cx = cx * max_side
  abs_cy = cy * max_side
  abs_w = w * max_side
  abs_h = h * max_side

  x_min = max(0.0, abs_cx - abs_w / 2)
  y_min = max(0.0, abs_cy - abs_h / 2)
  # Clamp to image bounds
  x_min = min(x_min, float(img_width))
  y_min = min(y_min, float(img_height))
  abs_w = min(abs_w, float(img_width) - x_min)
  abs_h = min(abs_h, float(img_height) - y_min)

  return [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)]


def format_predictions_as_coco(
    predictions,
    query_index_to_category_id,
):
  """Converts a list of per-image OWL-v2 predictions to COCO format.

  Args:
    predictions: List of dicts, one per image, each containing:
      - image_id (int): COCO image ID.
      - img_width (int): Original image width.
      - img_height (int): Original image height.
      - boxes: List of [cx, cy, w, h] normalised boxes.
      - scores: List of float confidence scores.
      - labels: List of int text query indices.
    query_index_to_category_id: Mapping from text query index to COCO
      category_id.

  Returns:
    List of COCO prediction dicts:
      [{"image_id": int, "category_id": int,
        "bbox": [x, y, w, h], "score": float}, ...]
  """
  coco_preds = []
  for pred in predictions:
    image_id = pred["image_id"]
    img_w = pred["img_width"]
    img_h = pred["img_height"]

    for box, score, label in zip(
        pred["boxes"], pred["scores"], pred["labels"]
    ):
      category_id = query_index_to_category_id.get(label)
      if category_id is None:
        continue

      coco_box = owlv2_box_to_coco(
          box[0], box[1], box[2], box[3], img_w, img_h
      )

      coco_preds.append({
          "image_id": image_id,
          "category_id": category_id,
          "bbox": coco_box,
          "score": round(float(score), 4),
      })

  return coco_preds


def main():
  """CLI entry point."""
  parser = argparse.ArgumentParser(
      description="Convert OWL-v2 raw predictions to COCO format."
  )
  parser.add_argument(
      "--input", required=True, help="Path to raw predictions JSON."
  )
  parser.add_argument(
      "--output", required=True, help="Path to output COCO predictions JSON."
  )
  args = parser.parse_args()

  with open(args.input, "r") as f:
    raw = json.load(f)

  predictions = raw["predictions"]
  query_map = {
      int(k): v for k, v in raw["query_index_to_category_id"].items()
  }

  coco_preds = format_predictions_as_coco(predictions, query_map)

  with open(args.output, "w") as f:
    json.dump(coco_preds, f, indent=2)

  print(f"Wrote {len(coco_preds)} predictions to {args.output}")


if __name__ == "__main__":
  main()

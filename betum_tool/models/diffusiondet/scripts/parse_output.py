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

"""Converts Detectron2/DiffusionDet prediction outputs to standard COCO predictions JSON format.

Can be imported and used inside a notebook, or run as a CLI to parse saved raw
predictions.
"""

import argparse
import json
import pathlib
import pickle
from typing import TYPE_CHECKING, TypedDict, Union

if TYPE_CHECKING:
  from detectron2 import structures

Path = pathlib.Path


class CocoPrediction(TypedDict):
  image_id: int
  category_id: int
  bbox: list[float]
  score: float


class Detectron2ImagePrediction(TypedDict):
  image_id: int
  instances: Union["structures.Instances", object]


def instances_to_coco_predictions(
    instances, image_id
):
  """Converts a single Detectron2 `Instances` object to COCO prediction dicts.

  Args:
      instances: Detectron2 Instances object containing: - pred_boxes: Boxes
        object containing bounding boxes in [x1, y1, x2, y2] format. - scores:
        Tensor of float confidence scores. - pred_classes: Tensor of integer
        category IDs.
      image_id: The COCO image ID for these predictions.

  Returns:
      List of COCO format prediction dicts:
          [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score":
          float}, ...]
  """
  predictions: list[CocoPrediction] = []
  if not hasattr(instances, "pred_boxes") or len(instances) == 0:  # type: ignore
    return predictions

  # Extract tensors and move to CPU numpy arrays
  boxes = instances.pred_boxes.tensor.cpu().numpy()  # type: ignore [union-attr] # [x1, y1, x2, y2]
  scores = instances.scores.cpu().numpy()  # type: ignore [union-attr]
  classes = instances.pred_classes.cpu().numpy()  # type: ignore [union-attr]

  for box, score, cls in zip(boxes, scores, classes):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    predictions.append({
        "image_id": int(image_id),
        "category_id": int(cls),
        "bbox": [
            round(float(x1), 2),
            round(float(y1), 2),
            round(float(w), 2),
            round(float(h), 2),
        ],
        "score": round(float(score), 4),
    })
  return predictions


def detectron2_predictions_to_coco(
    predictions_list,
):
  """Converts a list of per-image prediction dicts to standard COCO predictions list.

  Each item in predictions_list should be a dictionary containing:
      - "image_id": int
      - "instances": Detectron2 Instances object (or a mock object with
      equivalent attributes)

  Args:
      predictions_list: List of dicts containing image_id and instances.

  Returns:
      A flat list of standard COCO format prediction dictionaries.
  """
  coco_preds: list[CocoPrediction] = []
  for pred in predictions_list:
    img_id = pred["image_id"]
    instances = pred["instances"]
    coco_preds.extend(instances_to_coco_predictions(instances, img_id))
  return coco_preds


def main():
  """CLI entry point to parse saved raw predictions."""
  parser = argparse.ArgumentParser(
      description=(
          "Convert saved Detectron2/DiffusionDet predictions to COCO JSON."
      )
  )
  parser.add_argument(
      "--input",
      required=True,
      help="Path to saved raw predictions file (.pkl or .json)",
  )
  parser.add_argument(
      "--output",
      required=True,
      help="Path to save the output COCO predictions JSON.",
  )
  args = parser.parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)

  if not input_path.exists():
    raise FileNotFoundError(f"Input predictions file not found: {input_path}")

  print(f"Loading predictions from {input_path}...")
  if input_path.suffix == ".pkl":
    with open(input_path, "rb") as f:
      raw_preds = pickle.load(f)
  elif input_path.suffix == ".json":
    with open(input_path, "r") as f:
      raw_preds = json.load(f)
  else:
    raise ValueError("Unsupported file format. Input must end in .pkl or .json")

  coco_preds = []
  # If raw_preds is a list of dicts
  if isinstance(raw_preds, list):
    # Check if it's already formatted as COCO format or needs conversion
    if (
        len(raw_preds) > 0
        and "bbox" in raw_preds[0]
        and "score" in raw_preds[0]
    ):
      print("Input is already in COCO format. Copying to output...")
      coco_preds = raw_preds
    else:
      print("Converting list of Detectron2 predictions...")
      coco_preds = detectron2_predictions_to_coco(raw_preds)
  # If raw_preds is a dictionary of {image_id: instances}
  elif isinstance(raw_preds, dict):
    print("Converting dictionary of image_id -> instances predictions...")
    for img_id, instances in raw_preds.items():
      coco_preds.extend(instances_to_coco_predictions(instances, img_id))
  else:
    raise ValueError(
        "Invalid raw predictions structure. Expected a list or dictionary."
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w") as f:
    json.dump(coco_preds, f, indent=2)

  print(f"Successfully parsed {len(coco_preds)} prediction entries.")
  print(f"Wrote output to {output_path.resolve()}")


if __name__ == "__main__":
  main()

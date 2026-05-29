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

r"""Parses YOLO prediction outputs and formats them into standard COCO predictions JSON.

Supports both memory-level parsing of Ultralytics `Results` objects and parsing of saved YOLO txt label files.

Example usage (CLI):
    python3 parse_output.py \
        --labels_dir runs/detect/predict/labels \
        --coco_val data/coco/cashew_val.json \
        --output outputs/yolo26_predictions.json
"""

import argparse
import json
import pathlib
from typing import Any

Path = pathlib.Path


def parse_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Convert YOLO prediction outputs to COCO predictions JSON format."
      )
  )
  parser.add_argument(
      "--labels_dir",
      type=str,
      help="Path to directory containing YOLO prediction .txt label files",
  )
  parser.add_argument(
      "--coco_val",
      type=str,
      required=True,
      help="Path to ground-truth COCO JSON file",
  )
  parser.add_argument(
      "--output",
      type=str,
      required=True,
      help="Path where formatted COCO predictions JSON will be written",
  )
  return parser.parse_args()


def load_coco_mapping(
    coco_path,
):
  """Loads the COCO JSON file and builds two indexing mappings.

  Args:
    coco_path: Path to the ground-truth COCO JSON file.

  Returns:
    A tuple containing two dictionaries:
    1. name_map: maps exact image filename -> image metadata (id, width, height)
    2. stem_map: maps image filename stem -> image metadata (id, width, height)
  """
  print(f"Loading ground-truth COCO mapping from {coco_path}...")
  with open(coco_path, "r") as f:
    coco_data = json.load(f)

  stem_map = {}
  name_map = {}

  for img in coco_data.get("images", []):
    file_name = img["file_name"]
    img_meta = {"id": img["id"], "width": img["width"], "height": img["height"]}
    name_map[file_name] = img_meta
    name_map[Path(file_name).name] = img_meta
    stem_map[Path(file_name).stem] = img_meta
    # Also add stripped stem for robust matching with weird spacing names
    stem_map[Path(file_name).stem.strip()] = img_meta

  return name_map, stem_map


def yolo_results_to_coco(
    results, coco_val_path
):
  """Converts a list of Ultralytics Results objects directly to COCO predictions list."""
  name_map, _ = load_coco_mapping(Path(coco_val_path))
  predictions = []

  print(f"Processing {len(results)} Results objects...")
  for result in results:
    # Get filename from the prediction path
    img_path = Path(result.path)
    img_name = img_path.name

    # Lookup in COCO mapping
    img_meta = name_map.get(img_name)
    if not img_meta:
      # Try fallback matching
      img_meta = name_map.get(img_name.strip())

    if not img_meta:
      print(
          "Warning: Could not find COCO image metadata for filename:"
          f" '{img_name}'"
      )
      continue

    img_id = img_meta["id"]

    # Iterate over detected boxes
    if result.boxes is not None:
      for box in result.boxes:
        cat_id = int(box.cls.item())
        conf = float(box.conf.item())
        # xyxy format: [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Convert to COCO format [x_min, y_min, width, height]
        w = x2 - x1
        h = y2 - y1

        predictions.append({
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(conf, 4),
        })

  print(f"Formulated {len(predictions)} predictions.")
  return predictions


def parse_yolo_txt_files(
    labels_dir, coco_val_path
):
  """Parses YOLO prediction text files to compile a COCO predictions list."""
  _, stem_map = load_coco_mapping(coco_val_path)
  predictions = []

  txt_files = list(labels_dir.glob("*.txt"))
  print(f"Found {len(txt_files)} prediction .txt files in {labels_dir}...")

  for txt_path in txt_files:
    stem = txt_path.stem

    # Lookup in COCO mapping
    img_meta = stem_map.get(stem)
    if not img_meta:
      img_meta = stem_map.get(stem.strip())

    if not img_meta:
      print(f"Warning: Could not find COCO image metadata for stem: '{stem}'")
      continue

    img_id = img_meta["id"]
    img_w = img_meta["width"]
    img_h = img_meta["height"]

    with open(txt_path, "r") as f:
      lines = f.readlines()

    for line in lines:
      parts = line.strip().split()
      if len(parts) < 6:
        # If confidence score is missing, default to 1.0
        cat_id, cx_norm, cy_norm, w_norm, h_norm = map(float, parts[:5])
        conf = 1.0
      else:
        cat_id, cx_norm, cy_norm, w_norm, h_norm, conf = map(float, parts[:6])

      cat_id = int(cat_id)

      # Convert normalized YOLO to absolute COCO [x_min, y_min, w, h]
      w = w_norm * img_w
      h = h_norm * img_h
      x_min = (cx_norm - w_norm / 2.0) * img_w
      y_min = (cy_norm - h_norm / 2.0) * img_h

      predictions.append({
          "image_id": img_id,
          "category_id": cat_id,
          "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
          "score": round(conf, 4),
      })

  print(f"Parsed {len(predictions)} predictions from label files.")
  return predictions


def main():
  """Main entry point for command line execution."""
  args = parse_args()
  coco_val = Path(args.coco_val)
  output = Path(args.output)

  if not coco_val.exists():
    raise FileNotFoundError(f"Ground-truth COCO file not found: {coco_val}")

  # Formulate predictions
  if args.labels_dir:
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
      raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    predictions = parse_yolo_txt_files(labels_dir, coco_val)
  else:
    raise ValueError(
        "Please provide --labels_dir to parse saved label text files."
    )

  # Ensure parent directories for output exist
  output.parent.mkdir(parents=True, exist_ok=True)

  # Save predictions to JSON
  with open(output, "w") as f:
    json.dump(predictions, f, indent=2)

  print(f"Successfully wrote COCO predictions to {output.resolve()}")


if __name__ == "__main__":
  main()

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

r"""Converts YOLO annotations to COCO JSON format.

Example usage:
    .venv/bin/python3 common/yolo_to_coco.py \
        --images data/Coffee_flattened/images \
        --labels data/Coffee_flattened/labels \
        --class_map common/class_map.json \
        --dataset coffee \
        --output data/coco/ \
        --split_ratio 0.8
"""

import argparse
import json
import pathlib
import random
from typing import Any, Optional
from PIL import Image


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Convert YOLO annotations to COCO JSON format."
  )
  parser.add_argument(
      "--images", type=str, required=True, help="Path to image directory"
  )
  parser.add_argument(
      "--labels", type=str, required=True, help="Path to label directory"
  )
  parser.add_argument(
      "--class_map", type=str, required=True, help="Path to class map JSON"
  )
  parser.add_argument(
      "--dataset",
      type=str,
      required=True,
      choices=["cashew", "coffee"],
      help="Dataset name (cashew or coffee)",
  )
  parser.add_argument(
      "--output", type=str, required=True, help="Output directory for COCO JSON"
  )
  parser.add_argument(
      "--split_ratio",
      type=float,
      default=0.8,
      help="Train/val split ratio (default: 0.8)",
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed for reproducibility"
  )
  return parser.parse_args()


def load_class_map(class_map_path, dataset_name):
  """Loads class mapping from JSON file for a specific dataset."""
  with open(class_map_path, "r") as f:
    class_map = json.load(f)
  return class_map[dataset_name]


def yolo_to_coco_bbox(
    cx, cy, w, h, img_w, img_h
):
  """Converts normalized YOLO bbox (cx, cy, w, h) to absolute COCO bbox (x_min, y_min, w, h)."""
  x_min = (cx - w / 2) * img_w
  y_min = (cy - h / 2) * img_h
  width = w * img_w
  height = h * img_h
  return [x_min, y_min, width, height]


def get_image_paths(image_dir):
  """Finds all images in the directory with supported extensions."""
  image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
  image_paths = []
  for ext in image_extensions:
    image_paths.extend(image_dir.glob(ext))
  return image_paths


def create_coco_image(
    img_id, img_path, image_dir
):
  """Creates a COCO image entry."""
  try:
    img = Image.open(img_path)
    width, height = img.size
    return {
        "id": img_id,
        "file_name": str(img_path.relative_to(image_dir)),
        "width": width,
        "height": height,
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error reading image {img_path}: {e}")
    return None


def create_coco_annotations(
    img_id,
    label_path,
    width,
    height,
    tree_classes,
    start_ann_id,
):
  """Creates COCO annotation entries from a YOLO label file."""
  annotations = []
  ann_id = start_ann_id

  if not label_path.exists():
    return annotations, ann_id

  with open(label_path, "r") as f:
    lines = f.readlines()

  for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
      continue

    cls_id = parts[0]
    cx, cy, w, h = map(float, parts[1:])

    bbox = yolo_to_coco_bbox(cx, cy, w, h, width, height)
    area = bbox[2] * bbox[3]

    annotation = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": int(cls_id),
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }

    if cls_id in tree_classes:
      annotation["is_tree_class"] = True

    annotations.append(annotation)
    ann_id += 1

  return annotations, ann_id


def process_split(
    paths,
    split_name,
    image_dir,
    label_dir,
    output_dir,
    class_map,
    dataset,
    tree_classes,
):
  """Processes a list of image paths, converts labels to COCO format, and saves to JSON."""
  coco_data = {
      "images": [],
      "annotations": [],
      "categories": [{"id": int(k), "name": v} for k, v in class_map.items()],
  }

  img_id = 1
  ann_id = 1

  for img_path in paths:
    if dataset == "coffee" and "Copy of DJI_" in img_path.name:
      continue

    coco_img = create_coco_image(img_id, img_path, image_dir)
    if not coco_img:
      continue

    coco_data["images"].append(coco_img)

    label_path = label_dir / (img_path.stem + ".txt")
    anns, next_ann_id = create_coco_annotations(
        img_id,
        label_path,
        coco_img["width"],
        coco_img["height"],
        tree_classes,
        ann_id,
    )

    coco_data["annotations"].extend(anns)
    ann_id = next_ann_id
    img_id += 1

  output_file = output_dir / f"{dataset}_{split_name}.json"
  with open(output_file, "w") as f:
    json.dump(coco_data, f, indent=2)
  print(f"Saved {split_name} annotations to {output_file}")


def main():
  """Main execution function."""
  args = parse_args()

  random.seed(args.seed)

  class_map = load_class_map(args.class_map, args.dataset)

  tree_classes = []
  if args.dataset == "cashew":
    tree_classes = ["0"]
  elif args.dataset == "coffee":
    tree_classes = ["4"]

  image_dir = pathlib.Path(args.images)
  label_dir = pathlib.Path(args.labels)
  output_dir = pathlib.Path(args.output)
  output_dir.mkdir(parents=True, exist_ok=True)

  image_paths = get_image_paths(image_dir)
  print(f"Found {len(image_paths)} images in {image_dir}")

  random.shuffle(image_paths)
  split_idx = int(len(image_paths) * args.split_ratio)
  train_paths = image_paths[:split_idx]
  val_paths = image_paths[split_idx:]

  print(f"Split: {len(train_paths)} train, {len(val_paths)} val")

  process_split(
      train_paths,
      "train",
      image_dir,
      label_dir,
      output_dir,
      class_map,
      args.dataset,
      tree_classes,
  )
  process_split(
      val_paths,
      "val",
      image_dir,
      label_dir,
      output_dir,
      class_map,
      args.dataset,
      tree_classes,
  )


if __name__ == "__main__":
  main()

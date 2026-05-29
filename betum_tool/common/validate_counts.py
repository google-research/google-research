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

r"""Validates COCO annotation counts against original YOLO files.

Example usage:
    .venv/bin/python3 shared/validate_counts.py \
        --json data/coco/coffee_train.json \
        --images data/Coffee_flattened/images \
        --dataset coffee
"""

import argparse
import pathlib
from typing import Any
from pycocotools import coco as coco_lib

COCO = coco_lib.COCO
Path = pathlib.Path


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Validate COCO annotation counts against YOLO files."
  )
  parser.add_argument(
      "--json", type=str, required=True, help="Path to COCO JSON file"
  )
  parser.add_argument(
      "--images",
      type=str,
      required=True,
      help="Path to image root directory (for searching)",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      required=True,
      choices=["cashew", "coffee"],
      help="Dataset name",
  )
  return parser.parse_args()


def get_image_path(images_root, file_name):
  """Gets the absolute path to an image file if it exists."""
  img_path = Path(images_root) / file_name
  if not img_path.exists():
    print(f"Warning: Image {file_name} not found on disk.")
    return None
  return img_path


def get_label_path(img_path):
  """Infers the corresponding YOLO label file path by replacing 'images' directory with 'labels'."""
  for label_dirname in ["labels", "Labels"]:
    parts = []
    for part in img_path.parts:
      if part.lower() == "images":
        parts.append(label_dirname)
      else:
        parts.append(part)

    try_path = Path(*parts).with_suffix(".txt")
    if try_path.exists():
      return try_path
  return None


def count_yolo_annotations(label_path):
  """Counts valid annotations in a YOLO text file."""
  if not label_path or not label_path.exists():
    return 0
  count = 0
  with open(label_path, "r") as f:
    for line in f:
      if line.strip().split():
        count += 1
  return count


def print_results(
    total_images,
    total_coco,
    total_yolo,
    missing_labels,
    skipped_images,
    mismatches,
):
  """Prints validation results summary."""
  if skipped_images > 0:
    pct = skipped_images / total_images * 100
    print(
        f"\n⚠️  WARNING: {skipped_images}/{total_images} images ({pct:.0f}%)"
        " not found on disk!"
    )
    if pct > 50:
      print(
          "   → This likely means --images is pointing at the wrong directory."
      )

  print("\nValidation Results:")
  print(f"Total Images in COCO: {total_images}")
  print(f"Total Annotations in COCO: {total_coco}")
  print(f"Total Annotations in YOLO: {total_yolo}")
  print(f"Images with missing labels: {missing_labels}")

  if total_coco == total_yolo:
    print("✅ Success: Total annotation counts match!")
  else:
    print("❌ Error: Total annotation counts DO NOT match!")

  if mismatches:
    print(f"\nFound {len(mismatches)} images with count mismatches:")
    for m in mismatches[:10]:  # Show first 10
      print(
          f"  {m['file_name']}: COCO={m['coco']}, YOLO={m['yolo']} (Label:"
          f" {m['label_path']})"
      )
    if len(mismatches) > 10:
      print(f"  ... and {len(mismatches) - 10} more.")
  else:
    print("✅ All individual image counts match!")


def main():
  """Main execution function."""
  args = parse_args()

  coco = COCO(args.json)
  img_ids = coco.getImgIds()

  total_coco_anns = len(coco.getAnnIds())
  total_yolo_anns = 0

  mismatches = []
  missing_labels = 0
  skipped_images = 0

  print(f"Validating {args.json} against YOLO files...")

  for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]

    img_path = get_image_path(args.images, file_name)
    if not img_path:
      skipped_images += 1
      continue

    label_path = get_label_path(img_path)

    coco_ann_ids = coco.getAnnIds(imgIds=img_id)
    coco_count = len(coco_ann_ids)

    yolo_count = count_yolo_annotations(label_path)
    if not label_path:
      missing_labels += 1

    total_yolo_anns += yolo_count

    if coco_count != yolo_count:
      mismatches.append({
          "file_name": file_name,
          "coco": coco_count,
          "yolo": yolo_count,
          "label_path": str(label_path) if label_path else "None",
      })

  print_results(
      len(img_ids),
      total_coco_anns,
      total_yolo_anns,
      missing_labels,
      skipped_images,
      mismatches,
  )


if __name__ == "__main__":
  main()

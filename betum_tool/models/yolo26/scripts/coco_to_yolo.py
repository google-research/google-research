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

r"""Converts COCO JSON split annotations into the standard YOLO directory structure.

This ensures that YOLO training uses the exact same train/val splits as other models.

Example usage:
    python3 coco_to_yolo.py \
        --coco_train data/coco/cashew_train.json \
        --coco_val data/coco/cashew_val.json \
        --image_dir data/Cashew/Cashew-Uganda/images \
        --output_dir data/yolo/cashew
"""

import argparse
import json
import pathlib
import shutil

Path = pathlib.Path


def parse_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Convert COCO JSON annotations to YOLO directory structure."
  )
  parser.add_argument(
      "--coco_train",
      type=str,
      required=True,
      help="Path to train COCO JSON file",
  )
  parser.add_argument(
      "--coco_val", type=str, required=True, help="Path to val COCO JSON file"
  )
  parser.add_argument(
      "--image_dir",
      type=str,
      required=True,
      help="Path to directory containing raw images",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Path to output directory for YOLO dataset structure",
  )
  return parser.parse_args()


def convert_bbox_coco_to_yolo(
    bbox, img_w, img_h
):
  """Converts COCO bbox [x_min, y_min, w, h] to normalized YOLO bbox [cx, cy, w, h].

  Clamps values to [0.0, 1.0] to prevent boundary errors during YOLO validation.

  Args:
    bbox: A list of four floats representing the COCO bounding box [x_min,
      y_min, w, h].
    img_w: The width of the image.
    img_h: The height of the image.

  Returns:
    A list of four floats representing the normalized YOLO bounding box
    [center_x, center_y, width, height].
  """
  x_min, y_min, w, h = bbox

  # Calculate center coordinates
  cx = x_min + w / 2.0
  cy = y_min + h / 2.0

  # Normalize by image dimensions
  cx_norm = cx / img_w
  cy_norm = cy / img_h
  w_norm = w / img_w
  h_norm = h / img_h

  # Clamp to safe ranges [0.0, 1.0]
  cx_norm = max(0.0, min(1.0, cx_norm))
  cy_norm = max(0.0, min(1.0, cy_norm))
  w_norm = max(0.0, min(1.0, w_norm))
  h_norm = max(0.0, min(1.0, h_norm))

  return [cx_norm, cy_norm, w_norm, h_norm]


def process_coco_split(
    coco_path, image_dir, output_dir, split
):
  """Processes a single COCO split, copying images and creating label text files.

  Args:
    coco_path: Path to the COCO JSON annotation file.
    image_dir: Path to the directory containing the images.
    output_dir: Path to the root output directory for the YOLO dataset.
    split: The name of the split (e.g., "train", "val").

  Returns:
    A dictionary mapping category_id (int) to category name (str).
  """
  print(f"Loading COCO annotations from {coco_path}...")
  with open(coco_path, "r") as f:
    coco_data = json.load(f)

  # Extract categories mapping
  categories = {}
  for cat in coco_data.get("categories", []):
    categories[int(cat["id"])] = cat["name"]

  # Map images by ID
  images = {}
  for img in coco_data.get("images", []):
    images[img["id"]] = img

  # Group annotations by image ID
  annotations_by_image = {}
  for ann in coco_data.get("annotations", []):
    img_id = ann["image_id"]
    if img_id not in annotations_by_image:
      annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

  # Ensure target directories exist
  images_split_dir = output_dir / "images" / split
  labels_split_dir = output_dir / "labels" / split
  images_split_dir.mkdir(parents=True, exist_ok=True)
  labels_split_dir.mkdir(parents=True, exist_ok=True)

  print(f"Processing {len(images)} images for split '{split}'...")
  copied_count = 0
  missing_count = 0

  for img_id, img_info in images.items():
    file_name = img_info["file_name"]
    # Get base filename to handle flat copying cleanly
    base_name = Path(file_name).name
    src_img_path = image_dir / file_name

    # Fallback: try stripping whitespace or finding in the directory
    if not src_img_path.exists():
      src_img_path = image_dir / file_name.strip()

    if not src_img_path.exists():
      # Print sample of missing images occasionally
      if missing_count < 5:
        print(f"Warning: Image not found: {src_img_path}")
      missing_count += 1
      continue

    # Destination paths
    dst_img_path = images_split_dir / base_name
    # YOLO labels have .txt extension matching image name
    label_file_name = Path(base_name).stem + ".txt"
    dst_label_path = labels_split_dir / label_file_name

    # Copy image file
    shutil.copy2(src_img_path, dst_img_path)
    copied_count += 1

    # Process annotations for this image
    img_anns = annotations_by_image.get(img_id, [])
    yolo_lines = []

    img_w = img_info["width"]
    img_h = img_info["height"]

    for ann in img_anns:
      cat_id = int(ann["category_id"])
      bbox = ann["bbox"]

      # Convert COCO to YOLO format
      cx, cy, w, h = convert_bbox_coco_to_yolo(bbox, img_w, img_h)
      yolo_lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Write out label file (even empty if there are no annotations)
    with open(dst_label_path, "w") as lf:
      lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

  print(
      f"Split '{split}' finished: Copied {copied_count} images. Missing:"
      f" {missing_count}."
  )
  return categories


def generate_yaml_config(output_dir, categories):
  """Generates the data.yaml configuration file for YOLO training."""
  yaml_path = output_dir / "data.yaml"

  # Prepare names dictionary sorted by category ID
  names_dict_str = ""
  for cat_id in sorted(categories.keys()):
    name = categories[cat_id]
    names_dict_str += f"  {cat_id}: {name}\n"

  # Write the YAML content (standard formatted string)
  yaml_content = (
      f"path: {output_dir.resolve()}\n"
      "train: images/train\n"
      "val: images/val\n"
      "\n"
      "names:\n"
      f"{names_dict_str}"
  )

  with open(yaml_path, "w") as f:
    f.write(yaml_content)

  print(f"Generated dataset configuration at {yaml_path.resolve()}")


def main():
  """Main entry point."""
  args = parse_args()

  coco_train = Path(args.coco_train)
  coco_val = Path(args.coco_val)
  image_dir = Path(args.image_dir)
  output_dir = Path(args.output_dir)

  # Validate inputs
  if not coco_train.exists():
    raise FileNotFoundError(f"Train COCO file not found: {coco_train}")
  if not coco_val.exists():
    raise FileNotFoundError(f"Val COCO file not found: {coco_val}")
  if not image_dir.exists():
    raise FileNotFoundError(f"Image directory not found: {image_dir}")

  # Create output directories
  output_dir.mkdir(parents=True, exist_ok=True)

  # Process both train and validation splits
  train_categories = process_coco_split(
      coco_train, image_dir, output_dir, "train"
  )
  val_categories = process_coco_split(coco_val, image_dir, output_dir, "val")

  # Merge categories (should be identical)
  categories = {**train_categories, **val_categories}

  # Generate the dataset config YAML file
  generate_yaml_config(output_dir, categories)
  print("Dataset successfully converted to YOLO format!")


if __name__ == "__main__":
  main()

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

r"""Visualizes COCO annotations by drawing bounding boxes on images.

Example usage:
    .venv/bin/python3 shared/visualize_coco.py \
        --json data/coco/coffee_train.json \
        --images data/Coffee_flattened/images \
        --num_images 5 \
        --output_dir data/visualization/coffee
"""

import argparse
import pathlib
import random
from typing import Any, Optional

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from pycocotools import coco as coco_lib


Path = pathlib.Path
COCO = coco_lib.COCO


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(description="Visualize COCO annotations.")
  parser.add_argument(
      "--json", type=str, required=True, help="Path to COCO JSON file"
  )
  parser.add_argument(
      "--images", type=str, required=True, help="Path to image directory"
  )
  parser.add_argument(
      "--num_images", type=int, default=5, help="Number of images to visualize"
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="data/visualization",
      help="Directory to save output images",
  )
  return parser.parse_args()


def get_image_path(image_dir, file_name):
  """Gets the path to an image file, searching recursively if not found directly."""
  img_path = Path(image_dir) / file_name
  if not img_path.exists():
    # Try recursive search if not found directly
    found = list(Path(image_dir).rglob(file_name))
    if found:
      return found[0]
    else:
      print(f"Image not found: {file_name}")
      return None
  return img_path


# Distinct colors per category so class boundaries are easy to see.
_CATEGORY_COLORS = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
)


def draw_annotations(
    img, anns, coco
):
  """Draws bounding boxes and category names on the image, colored by category."""
  draw = ImageDraw.Draw(img)
  for ann in anns:
    color = _CATEGORY_COLORS[ann["category_id"] % len(_CATEGORY_COLORS)]
    x_min, y_min, w, h = ann["bbox"]
    draw.rectangle([x_min, y_min, x_min + w, y_min + h], outline=color, width=2)

    cat = coco.loadCats(ann["category_id"])[0]
    draw.text((x_min, max(y_min - 10, 0)), cat["name"], fill=color)
  return img


def visualize(
    coco_json,
    image_dir,
    num_images = 5,
    output_dir = None,
):
  """Loads COCO JSON, samples random images, and visualizes them."""
  coco = COCO(coco_json)
  img_ids = coco.getImgIds()

  if len(img_ids) == 0:
    print("No images found in COCO JSON.")
    return

  sampled_ids = random.sample(img_ids, min(num_images, len(img_ids)))

  if output_dir:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

  for img_id in sampled_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    img_path = get_image_path(image_dir, img_info["file_name"])
    if not img_path:
      continue

    with Image.open(img_path) as img:
      draw_annotations(img, anns, coco)

      if output_dir:
        out_path = Path(output_dir) / f"vis_{img_info['file_name']}"
        img.save(out_path)
        print(f"Saved visualization to {out_path}")
      else:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def main():
  """Main execution function."""
  args = parse_args()
  visualize(args.json, args.images, args.num_images, args.output_dir)


if __name__ == "__main__":
  main()

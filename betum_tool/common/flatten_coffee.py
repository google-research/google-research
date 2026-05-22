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

"""Flattens Coffee dataset batches into a single directory structure.

This toolkit prefixes filenames to avoid collisions.

Example usage:
    python common/flatten_coffee.py
"""

import pathlib
import shutil


def process_batch(
    batch, dst_images, dst_labels
):
  """Processes a single batch directory, copying images and labels with batch prefix."""
  batch_name = batch.name
  print(f"Processing {batch_name}...")

  img_dir = batch / "images"
  label_dir = batch / "labels"

  if not img_dir.exists():
    print(f"Warning: {img_dir} does not exist. Skipping.")
    return 0, 0

  image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
  image_paths = []
  for ext in image_extensions:
    image_paths.extend(img_dir.glob(ext))

  total_images = 0
  skipped_duplicates = 0

  for img_path in image_paths:
    if "Copy of DJI_" in img_path.name:
      skipped_duplicates += 1
      continue

    new_name = f"{batch_name}_{img_path.name}"
    shutil.copy(img_path, dst_images / new_name)

    label_path = label_dir / (img_path.stem + ".txt")
    if label_path.exists():
      new_label_name = f"{batch_name}_{label_path.name}"
      shutil.copy(label_path, dst_labels / new_label_name)

    total_images += 1

  return total_images, skipped_duplicates


def main():
  """Main execution function."""
  src_root = pathlib.Path("data/Coffee")
  dst_root = pathlib.Path("data/Coffee_flattened")

  dst_images = dst_root / "images"
  dst_labels = dst_root / "labels"

  dst_images.mkdir(parents=True, exist_ok=True)
  dst_labels.mkdir(parents=True, exist_ok=True)

  batches = list(src_root.glob("Batch*"))
  print(f"Found {len(batches)} batches.")

  total_images = 0
  skipped_duplicates = 0

  for batch in batches:
    imgs, skipped = process_batch(batch, dst_images, dst_labels)
    total_images += imgs
    skipped_duplicates += skipped

  print("\nFlattening completed:")
  print(f"Total images processed: {total_images}")
  print(f"Skipped duplicates: {skipped_duplicates}")
  print(f"Flattened data saved to {dst_root}")


if __name__ == "__main__":
  main()

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

"""Data loading and processing utilities."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
from typing import Any

from absl import logging
import datasets
from etils import epath
import ml_collections
from PIL import Image


def _load_coco_json(json_path):
  """Loads a COCO-style JSON annotation file using epath."""
  with json_path.open("r") as f:
    return json.load(f)


def _coco_to_hf_dict(
    image_dir, annotation_file
):
  """Converts COCO annotations to a dictionary for Hugging Face Dataset.

  Args:
    image_dir: EPath to the directory containing images.
    annotation_file: EPath to the COCO JSON annotation file.

  Returns:
    A tuple containing:
      - hf_data: A dictionary structured for `datasets.Dataset.from_dict`.
      - features: A `datasets.Features` object defining the dataset schema.
  """
  coco_data = _load_coco_json(annotation_file)

  empty_hf_data = {"image_id": [], "image": [], "objects": []}

  if "categories" not in coco_data:
    logging.warning("No categories found in annotation file.")
    return empty_hf_data, datasets.Features()

  if "images" not in coco_data:
    logging.warning("No images found in annotation file.")
    return empty_hf_data, datasets.Features()

  if "annotations" not in coco_data:
    logging.warning("No annotations found in annotation file.")
    return empty_hf_data, datasets.Features()

  id2label = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
  # Ensure background is included if not present, as it's often used in models.
  if 0 not in id2label:
    id2label[0] = "background"

  sorted_ids = sorted(id2label.keys())
  if any(sorted_ids[i] != i for i in range(len(sorted_ids))):
    raise ValueError(
        "COCO category IDs are not contiguous starting from 0. "
        "Found IDs: %s" % sorted_ids
    )
  sorted_labels = [id2label[i] for i in sorted_ids]

  # Group annotations by image_id
  img_id_to_anns = {}
  for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in img_id_to_anns:
      img_id_to_anns[img_id] = []
    img_id_to_anns[img_id].append(ann)

  hf_data = empty_hf_data

  for image_info in coco_data["images"]:
    image_id = image_info["id"]
    file_name = image_info["file_name"]
    image_path = image_dir / file_name

    if not image_path.exists():
      logging.warning("Warning: Image not found: %s", image_path)
      continue

    try:
      with image_path.open("rb") as f:
        image = Image.open(f).convert("RGB")
    except (IOError, OSError, Image.UnidentifiedImageError) as e:
      logging.warning("Warning: Could not open image %s: %s", image_path, e)
      continue

    annotations = img_id_to_anns.get(image_id, [])
    categories = []
    bboxes = []
    for ann in annotations:
      # COCO bbox format: [xmin, ymin, width, height]
      bbox = ann["bbox"]
      if isinstance(bbox, dict):
        bbox = [
            float(bbox["x"]),
            float(bbox["y"]),
            float(bbox["width"]),
            float(bbox["height"]),
        ]
      category_id = ann["category_id"]
      categories.append(category_id)
      bboxes.append(bbox)

    hf_data["image_id"].append(image_id)
    hf_data["image"].append(image)
    hf_data["objects"].append({"category": categories, "bbox": bboxes})

  features = datasets.Features(
      image_id=datasets.Value(dtype="int64"),
      image=datasets.Image(),
      objects=datasets.Sequence(
          feature={
              "category": datasets.ClassLabel(names=sorted_labels),
              "bbox": datasets.Sequence(
                  feature=datasets.Value(dtype="float32")
              ),
          }
      ),
  )

  return hf_data, features


def convert_coco_folder_to_hf(
    coco_root_dir,
    output_hf_path,
):
  """Converts a COCO-formatted folder to a Hugging Face DatasetDict.

  This function is adapted to the user's specific COCO directory structure:
  - Annotation files: train.json, valid.json, test.json are at the root.
  - All images are in a single 'images/' folder at the root.

  Args:
    coco_root_dir: Path to the root directory containing COCO files.
    output_hf_path: Path where the Hugging Face DatasetDict will be saved.
  """
  coco_root_dir = epath.Path(coco_root_dir)
  output_hf_path = epath.Path(output_hf_path)
  dataset_dict = {}
  image_dir = coco_root_dir / "images"

  if not image_dir.exists():
    logging.error("Error: Images directory not found at %s", image_dir)
    return

  split_configs = {
      "train": coco_root_dir / "train.json",
      "valid": coco_root_dir / "valid.json",
      "test": coco_root_dir / "test.json",
  }

  for split, annotation_file in split_configs.items():
    if not annotation_file.exists():
      logging.warning(
          "Skipping split '%s': Annotation file not found at %s",
          split,
          annotation_file,
      )
      continue

    logging.info("Processing split: %s...", split)
    logging.info("  - Annotations: %s", annotation_file)
    logging.info("  - Images: %s", image_dir)

    hf_data, features = _coco_to_hf_dict(image_dir, annotation_file)
    if not hf_data["image_id"]:
      logging.warning("Skipping split '%s': No valid images found.", split)
      continue

    dataset = datasets.Dataset.from_dict(hf_data, features=features)
    dataset_dict[split] = dataset
    logging.info("  - Loaded %d samples for '%s'.", len(dataset), split)

  if not dataset_dict:
    logging.warning("No valid COCO splits found to convert.")
    return

  hf_dataset_dict = datasets.DatasetDict(dataset_dict)
  hf_dataset_dict.save_to_disk(output_hf_path)
  logging.info("Dataset saved to %s", output_hf_path)






def get_dataset(cfg):
  """Loads and prepares the dataset."""
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  # Ensure HuggingFace datasets cache is writable inside Borg containers.
  # (Default ~/.cache is not writable.)
  hf_cache_dir = os.path.join(
      os.environ.get("TMPDIR", "/tmp"), "hf_datasets_cache"
  )
  os.makedirs(hf_cache_dir, exist_ok=True)
  os.environ["HF_DATASETS_CACHE"] = hf_cache_dir

  if world_size > 1 and str(dataset_path).startswith("/cns"):
    return _get_dataset_distributed(cfg)

  # Try loading as a pre-saved HuggingFace Dataset first
  try:
    return datasets.load_from_disk(str(dataset_path))
  except FileNotFoundError:
    # Fall back to loading raw parquet files from directory
    logging.info(
        "Directory %s is not a saved Dataset, trying to load as parquet files.",
        dataset_path,
    )
    dataset = datasets.load_dataset(
        "parquet",
        data_dir=str(dataset_path),
        cache_dir=hf_cache_dir,
        streaming=True,
    )

    return dataset

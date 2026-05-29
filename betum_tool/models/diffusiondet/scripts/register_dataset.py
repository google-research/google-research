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

"""Register Coffee and Cashew datasets with Detectron2."""

import os
import pathlib

from detectron2 import data
from detectron2.data import datasets


MetadataCatalog = data.MetadataCatalog
Path = pathlib.Path
register_coco_instances = datasets.register_coco_instances


def register_plant_datasets(
    data_root = "data", coco_root = "data/coco"
):
  """Registers cashew and coffee datasets for both train and val splits.

  Args:
      data_root: Root directory containing the raw image datasets.
      coco_root: Directory containing the COCO JSON annotations.
  """
  data_root_path = Path(data_root)
  coco_root_path = Path(coco_root)

  splits = ["train", "val"]
  dataset_names = ["cashew", "coffee"]

  for split in splits:
    for dataset in dataset_names:
      name = f"{dataset}_{split}"
      json_path = os.path.join(coco_root_path, f"{dataset}_{split}.json")

      # Resolve exact image directories
      if dataset == "cashew":
        img_dir = os.path.join(
            data_root_path, "Cashew", "Cashew-Uganda", "images"
        )
      else:
        img_dir = os.path.join(data_root_path, "Coffee_flattened", "images")

      # Register with Detectron2
      if name not in MetadataCatalog.list():
        print(f"Registering dataset: {name}")
        print(f"  - Annotations: {json_path}")
        print(f"  - Images: {img_dir}")
        register_coco_instances(name, {}, json_path, img_dir)
      else:
        print(f"Dataset {name} already registered.")


if __name__ == "__main__":
  # Simple validation or registration check if run standalone

  register_plant_datasets()
  print("All datasets registered successfully.")
  print(f"Registered datasets list: {MetadataCatalog.list()}")

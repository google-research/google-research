# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# pylint: disable=missing-module-docstring
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=consider-using-from-import
# pylint: disable=missing-function-docstring
# pylint: disable=g-importing-member
# pylint: disable=broad-exception-caught

import os
from pathlib import Path

from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)
from omegaconf import OmegaConf


DATA_URL = {
    "train": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "train2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}


def download_datasets(root, url):
  download_and_extract_archive(
      url=url, download_root=root, extract_root=storage_dir
  )


if __name__ == "__main__":

  config_path = get_abs_path("configs/datasets/vg/defaults_caption.yaml")

  storage_dir = OmegaConf.load(
      config_path
  ).datasets.vg_caption.build_info.images.storage

  download_dir = Path(get_cache_path(storage_dir)).parent / "download"
  storage_dir = Path(get_cache_path(storage_dir))

  if storage_dir.exists():
    print(f"Dataset already exists at {storage_dir}. Aborting.")
    exit(0)

  try:
    for k, v in DATA_URL.items():
      print("Downloading {} to {}".format(v, k))
      download_datasets(download_dir, v)
  except Exception as e:
    # remove download dir if failed
    cleanup_dir(download_dir)
    print("Failed to download or extracting datasets. Aborting.")

  cleanup_dir(download_dir)

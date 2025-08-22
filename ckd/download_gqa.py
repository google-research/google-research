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

from pathlib import Path  # pylint: disable=g-importing-member

from lavis.common.utils import cleanup_dir
from lavis.common.utils import download_and_extract_archive
from lavis.common.utils import get_abs_path
from lavis.common.utils import get_cache_path
from omegaconf import OmegaConf  # pylint: disable=g-importing-member


DATA_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"


def download_datasets(root, url):
  download_and_extract_archive(
      url=url, download_root=root, extract_root=storage_dir.parent
  )


if __name__ == "__main__":

  config_path = get_abs_path("configs/datasets/gqa/defaults.yaml")

  storage_dir = OmegaConf.load(
      config_path
  ).datasets.gqa.build_info.images.storage

  download_dir = Path(get_cache_path(storage_dir)).parent / "download"
  storage_dir = Path(get_cache_path(storage_dir))

  if storage_dir.exists():
    print(f"Dataset already exists at {storage_dir}. Aborting.")
    exit(0)

  try:
    print("Downloading {}".format(DATA_URL))
    download_datasets(download_dir, DATA_URL)
  except Exception:  # pylint: disable=broad-exception-caught
    # remove download dir if failed
    cleanup_dir(download_dir)
    print("Failed to download or extracting datasets. Aborting.")

  cleanup_dir(download_dir)

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

r"""Converts COCO JSON annotations to Hugging Face DatasetDict.

Usage:

python convert_coco_json_to_hf_dataset.py -- \
    --coco_root_dir=/path/to/coco/root \
    --output_hf_path=/path/to/output/hf
"""

import pathlib
import sys

# Add repo root to sys.path to allow imports from common
repo_root = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(repo_root))

# pylint: disable=g-import-not-at-top
from absl import app
from absl import flags
from absl import logging

from common import data


_COCO_ROOT_DIR = flags.DEFINE_string(
    "coco_root_dir",
    None,
    "Path to the root directory containing COCO annotations and images.",
    required=True,
)
_OUTPUT_HF_PATH = flags.DEFINE_string(
    "output_hf_path",
    None,
    "Path where the Hugging Face DatasetDict will be saved.",
    required=True,
)


def main(_):
  logging.info("Starting COCO to Hugging Face conversion...")
  logging.info("Input COCO root: %s", _COCO_ROOT_DIR.value)
  logging.info("Output HF path: %s", _OUTPUT_HF_PATH.value)

  data.convert_coco_folder_to_hf(
      _COCO_ROOT_DIR.value,
      _OUTPUT_HF_PATH.value,
  )

  logging.info("Conversion completed.")


if __name__ == "__main__":
  app.run(main)

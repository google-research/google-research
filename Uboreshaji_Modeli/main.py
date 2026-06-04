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

"""Main entry point for fine-tuning experiments."""

import json
import os
import sys
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
import ml_collections
import torch  # pylint: disable=g-import-not-at-top,unused-import
from torch.utils import tensorboard  # pylint: disable=unused-import
import transformers  # pylint: disable=unused-import

from Uboreshaji_Modeli import main_lib
from Uboreshaji_Modeli.common import config_utils



_CONFIG = flags.DEFINE_string("config", None, "Path to Python config file.")
_CONFIG_JSON = flags.DEFINE_string(
    "config_json", None, "JSON string of the experiment configuration."
)
_MODEL_ID = flags.DEFINE_string(
    "model_id", None, "Path or ID of the pretrained model (overrides config)."
)
_DATASET_PATH = flags.DEFINE_string(
    "dataset_path", None, "Path to the dataset (overrides config)."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Directory to save outputs (overrides config)."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", None, "Name of the experiment (overrides config)."
)

flags.mark_flags_as_mutual_exclusive(["config", "config_json"], required=True)


def main(argv):
  """Runs fine-tuning experiments.

  Dispatches to the appropriate training pipeline based on the model_flavor
  configuration field.

  Args:
    argv: Command-line arguments.

  Raises:
    app.UsageError: If the config is not provided or if there are too many
      command-line arguments.
    ValueError: If the dataset type is unsupported.
    RuntimeError: If the Python version is lower than 3.11.
  """
  if len(argv) > 1:
    raise app.UsageError(f"Too many command-line arguments: {argv!r}")

  if sys.version_info < (3, 11):
    raise RuntimeError("This script requires Python 3.11 or higher.")

  # Early environmental variables configuration.
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

  if _CONFIG.value:
    cfg = config_utils.load_config(_CONFIG.value)
  else:
    cfg = ml_collections.ConfigDict(json.loads(_CONFIG_JSON.value))

  if _MODEL_ID.value:
    cfg.model_id = _MODEL_ID.value
  if _DATASET_PATH.value:
    cfg.dataset.dataset_path = _DATASET_PATH.value
  if _OUTPUT_DIR.value:
    cfg.output_dir = _OUTPUT_DIR.value
  if _EXPERIMENT_NAME.value:
    cfg.experiment_name = _EXPERIMENT_NAME.value

  output_path = epath.Path(cfg.output_dir)
  logging.info("Root output directory parsed: %s", output_path)

  if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)

  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  if local_rank == 0:
    config_save_path = output_path / "config.json"
    config_save_path.write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str)
    )

  # Delegate training execution directly to main_lib orchestrator.

if __name__ == "__main__":
  app.run(main)

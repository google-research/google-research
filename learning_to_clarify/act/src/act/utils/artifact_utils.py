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

from dataclasses import asdict
import json
import logging
from typing import Optional, Union

from act.config.base_config import BaseConfig, BaseInitializationConfig
from act.utils.storage_utils import save_model, write_json
from transformers import Trainer


logger = logging.getLogger(__name__)


def write_all_artifacts(
    config,
    trainer,
    metrics,
    eval_metrics,
):
  """Write artifacts."""
  logger.info("*** Save model ***")
  save_model(config.training_config.output_dir, trainer)
  logger.info(f"Model saved to {config.training_config.output_dir}")

  write_json(
      config.training_config.output_dir + "training_metrics.json", metrics
  )
  if eval_metrics:
    write_json(
        config.training_config.output_dir + "eval_metrics.json", eval_metrics
    )

  write_json(config.training_config.output_dir + "config.json", asdict(config))

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

"""Configuration for Gemma 4 fine-tuning on phenotyping datasets."""

import ml_collections
from Uboreshaji_Modeli.common import config


def get_config():
  """Returns the experiment configuration for Gemma 4 detection."""
  cfg = config.get_detection_base_config()

  cfg.experiment_name = "gemma4_detection_finetune"
  cfg.model_flavor = config.ModelFlavor.GEMMA_4
  cfg.task_modality = config.TaskModality.VISION

  cfg.dataset.dataset_base = ""
  cfg.dataset.dataset_uri = "mega_dataset_big"
  cfg.dataset.dataset_path = ""
  cfg.dataset.dataset_version = "1.1.1"
  cfg.dataset.train_split = "train"
  cfg.task_type = config.TaskType.VISION_DETECTION


  cfg.model_id = "path/to/gemma4-model"

  cfg.output_dir = "/path/to/gemma4_detection_output"

  cfg.max_seq_length = 8192
  cfg.prompt = ""
  cfg.freeze_vision_tower = True
  cfg.detection_format = "json"
  cfg.do_pan_and_scan = False

  cfg.use_lora = True
  cfg.use_torchao = False
  cfg.lora = ml_collections.ConfigDict()
  cfg.lora.r = 8
  cfg.lora.alpha = 16
  cfg.lora.dropout = 0.05
  cfg.lora.target_modules = [
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",
  ]

  # Let the Gemma 4 processor handle native image resolution.
  cfg.dataset.image_size = 0

  cfg.platform.hardware = "h100=8"
  cfg.platform.tmp_ram_fs_gb = 200
  cfg.platform.env_vars = {
      "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
  cfg.training.batch_size = 1
  cfg.training.num_train_epochs = 1
  cfg.training.gradient_accumulation_steps = 16
  cfg.training.learning_rate = 2e-5
  cfg.training.precision = config.Precision.BF16
  cfg.training.gradient_checkpointing = True
  cfg.training.save_steps = 100
  cfg.training.logging_steps = 10
  cfg.training.save_total_limit = 2

  return cfg

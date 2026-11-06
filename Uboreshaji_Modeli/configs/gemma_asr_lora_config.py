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

"""Gemma 3n E2B LoRA ASR fine-tuning on one WAXAL language.

Reads the WAXAL Parquet files from the google/WaxalNLP Hugging Face dataset.
Global batch size is batch_size * gradient_accumulation_steps * num_gpus = 16.
Select the language directory with
`--dataset_path=/path/to/WaxalNLP/data/ASR/<lang>` (default `aka`).
"""

import ml_collections
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils



def get_config():
  """Returns the experiment configuration."""
  cfg = config.get_base_config()

  cfg.experiment_name = "gemma_3n_asr_lora"
  cfg.model_flavor = config.ModelFlavor.GEMMA_3N_ASR
  cfg.task_modality = config.TaskModality.AUDIO
  cfg.task_type = config.TaskType.ASR
  cfg.model_id = "google/gemma-3n-E2B-it"
  cfg.output_dir = "/path/to/output"

  # WAXAL Parquet files, laid out as in google/WaxalNLP:
  # <dataset_base>/<lang>/<lang>-<split>-NNNNN.parquet.
  cfg.dataset.format = "parquet"
  cfg.dataset.dataset_base = "/path/to/WaxalNLP/data/ASR"
  cfg.dataset.dataset_uri = "aka"
  cfg.dataset.dataset_version = "bagz"
  cfg.dataset.train_split = "train"
  cfg.dataset.eval_split = "validation"
  cfg.dataset.eval_num_records = 256
  cfg.dataset.load_from_hub = False


  cfg.training.batch_size = 4
  cfg.training.gradient_accumulation_steps = 4
  cfg.training.max_steps = 5000
  cfg.training.learning_rate = 1e-4
  cfg.training.logging_steps = 10
  cfg.training.save_strategy = "steps"
  cfg.training.save_steps = 100
  cfg.training.save_total_limit = 2
  cfg.training.resume_from_checkpoint = "auto"
  cfg.training.use_lora = True
  cfg.training.lora = ml_collections.ConfigDict({
      "r": 1024,
      "alpha": 2048,
      "dropout": 0.0,
      "target_modules": [
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
          "multi_modal_projector",
      ],
  })

  cfg.platform.hardware = "h100=1"

  config_utils.derive_paths(cfg)
  return cfg

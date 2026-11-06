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

"""Reference template configuration for Gemma 3n full-model ASR fine-tuning.

Illustrates how to wire the `GEMMA_3N_ASR` engine (`TaskType.ASR`). For the
WAXAL Gemma 3n 2B LoRA ASR setup, use `configs/gemma_asr_lora_config.py`; when
reproducing other benchmarks (such as Section 4.1 of the WAXAL paper), pass
`--dataset_path`, `--model_id`, and adapt hyperparameters accordingly.
"""

import ml_collections
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils


def get_config():
  """Returns the experiment configuration."""
  cfg = config.get_base_config()

  cfg.experiment_name = "gemma_3n_asr_finetune"
  cfg.model_flavor = config.ModelFlavor.GEMMA_3N_ASR
  cfg.task_modality = config.TaskModality.AUDIO
  cfg.task_type = config.TaskType.ASR

  cfg.model_id = "google/gemma-3n-asr"

  # Data overrides
  cfg.dataset.dataset_uri = "waxal"
  cfg.dataset.load_from_hub = True

  cfg.output_dir = "/path/to/output"


  cfg.platform.hardware = "a100=1"
  cfg.training.batch_size = 1
  cfg.training.num_train_epochs = 5
  cfg.training.max_steps = 5000
  cfg.training.learning_rate = 1e-5
  cfg.training.audio_sample_rate = 16000

  cfg.prompt = "Please transcribe this audio."

  config_utils.derive_paths(cfg)
  return cfg

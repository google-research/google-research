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

"""Reference template configuration for Whisper ASR fine-tuning.

Illustrates how to wire the `WHISPER` engine (`TaskType.ASR`). Pass
`--dataset_path` and adapt hyperparameters for your target dataset.
"""

import ml_collections
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils


def get_config():
  """Returns the experiment configuration."""
  cfg = config.get_base_config()

  cfg.experiment_name = "whisper_asr_finetune"
  cfg.model_flavor = config.ModelFlavor.WHISPER
  cfg.task_modality = config.TaskModality.AUDIO
  cfg.task_type = config.TaskType.ASR

  cfg.model_id = "openai/whisper-large-v3"

  # Data overrides
  cfg.dataset.dataset_uri = "waxal"
  cfg.dataset.load_from_hub = True

  cfg.output_dir = "/path/to/output"


  cfg.platform.hardware = "a100=1"
  cfg.training.batch_size = 2
  cfg.training.num_train_epochs = 3
  cfg.training.max_steps = 5000
  cfg.training.learning_rate = 1e-5
  cfg.training.audio_sample_rate = 16000

  config_utils.derive_paths(cfg)
  return cfg

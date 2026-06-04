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

"""Configuration for overriding the base config for fine-tuning on a detection dataset."""

import ml_collections
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils


def get_config():
  """Returns the experiment configuration."""
  cfg = config.get_detection_base_config()

  cfg.experiment_name = "maize_owlv2_finetune"

  cfg.model_id = "/path/to/models/owlv2"

  # Data overrides
  cfg.dataset.dataset_uri = "maize_1"
  cfg.dataset.image_size = 960

  cfg.output_dir = "/path/to/maize_owlv2_finetune_output"
  cfg.dataset.dataset_path = "/path/to/huggingface_dataset"

  cfg.platform.hardware = "a100=1"
  cfg.training.batch_size = 4
  cfg.training.num_train_epochs = 5
  cfg.training.gradient_accumulation_steps = 4
  cfg.training.seed = 42
  cfg.training.data_seed = 42
  cfg.training.gradient_checkpointing = True
  cfg.training.learning_rate = 1e-5
  cfg.training.use_amp = False
  cfg.training.precision = config.Precision.BF16
  cfg.training.eval_every = 10
  cfg.training.save_every = 50

  return cfg

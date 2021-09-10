# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Config for fine-tuning on the GLUE and SuperGLUE benchmarks."""

from f_net.configs import base as base_config
from f_net.configs.base import ModelArchitecture
from f_net.configs.base import TrainingMode


def get_config():
  """Config for fine-tuning (classification)."""
  config = base_config.get_config()

  # Determines which model to use.
  config.model_arch: ModelArchitecture = ModelArchitecture.F_NET

  config.mode: TrainingMode = TrainingMode.CLASSIFICATION

  # This is either "glue/DS_g", where DS_g is one of the following:
  # [cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli].
  config.dataset_name: str = "glue/rte"

  # How often to save the model checkpoint.
  config.save_checkpoints_steps: int = 200
  # Training metrics will be computed (1 / eval_proportion) times during
  # training at regularly spaced intervals, regardless of dataset size.
  config.eval_proportion: float = 0.05

  # Total batch size for training.
  config.train_batch_size: int = 64
  # Total batch size for eval (and predictions).
  config.eval_batch_size: int = 32

  # The base learning rate for Adam.
  config.learning_rate: float = 1e-5

  # Total number of training epochs to perform.
  config.num_train_epochs: float = 3
  # Proportion of training to perform linear learning rate warmup for.
  # E.g., 0.1 = 10% of training steps.
  config.warmup_proportion: float = 0.1

  # Maximum number of eval steps on validation split. Actual number of step may
  # be less for small eval datasets.
  config.max_num_eval_steps: int = int(1e5)

  # Initial checkpoint directory or filepath (usually from a pre-trained model).
  config.init_checkpoint_dir: str = ""

  # Dummy attribute for repeated runs.
  config.trial: int = 0

  return config



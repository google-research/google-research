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

"""Config for pre-training on the C4 dataset."""

from typing import Optional

from f_net.configs import base as base_config
from f_net.configs.base import ModelArchitecture
from f_net.configs.base import TrainingMode


def get_config():
  """Config for pre-training."""
  config = base_config.get_config()

  # Determines which model to use.
  config.model_arch: ModelArchitecture = ModelArchitecture.F_NET

  config.mode: TrainingMode = TrainingMode.PRETRAINING

  # Total batch size for training.
  config.train_batch_size: int = 64
  # Total batch size for eval.
  config.eval_batch_size: int = 64

  # The base learning rate for Adam.
  config.learning_rate: float = 1e-4
  # If set, determines how much to clip the gradient during training.
  config.clipped_grad_norm: Optional[float] = None

  # Number of training steps.
  config.num_train_steps: int = int(1e6)
  # Number of warm-up steps. We generally find that that larger models need more
  # warm-up steps.
  config.num_warmup_steps: int = int(1e4)

  # How often to save the model checkpoint.
  config.save_checkpoints_steps: int = 2000
  # Frequency fo eval during training, e.g. every 2000 steps.
  config.eval_frequency: int = 2000

  # Maximum number of eval steps.
  config.max_num_eval_steps: int = 100

  # Do not start from a pre-trained checkpoint.
  config.init_checkpoint_dir: str = ''

  # Maximum number of masked LM predictions per sequence.
  config.max_predictions_per_seq: int = 80
  # Proportion of tokens for masked LM predictions. Total number of selected
  # tokens will be at most config.max_predictions_per_seq.
  config.masking_rate: float = 0.15
  # Proportion of masked tokens to replace with ['MASK'].
  config.mask_token_proportion: float = 0.8
  # Proportion of masked tokens to replace with a random token.
  config.random_token_proportion: float = 0.1
  # Remaining 1 - config.mask_token_proportion - config.random_token_proportion
  # fraction of selected tokens are left as is.

  # Dummy attribute for repeated runs.
  config.trial: int = 0

  return config



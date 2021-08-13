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

"""Base template config for pre-training and fine-tuning."""

import enum

import ml_collections


class ModelArchitecture(enum.Enum):
  """Determines model architecture - in particular, the mixing layer."""
  BERT = 'bert'
  F_NET = 'f_net'
  FF_ONLY = 'ff_only'  # Feed forward sublayers only; no token mixing
  LINEAR = 'linear'  # Matrix multiplications with learnable weights
  RANDOM = 'random'  # Constant, random matrix multiplications


class TrainingMode(enum.Enum):
  """Determines type of training."""
  PRETRAINING = 'pretraining'
  CLASSIFICATION = 'classification'


def get_config():
  """Base config for training models."""
  config = ml_collections.ConfigDict()

  # How often to save the model checkpoint.
  config.save_checkpoints_steps: int = 1000
  # Frequency fo eval during training, e.g. every 1000 steps.
  config.eval_frequency: int = 1000

  # Total batch size for training.
  config.train_batch_size: int = 32
  # Total batch size for eval.
  config.eval_batch_size: int = 8

  # The base learning rate for Adam.
  config.learning_rate: float = 1e-4

  # Initial checkpoint directory (usually from a pre-trained model).
  config.init_checkpoint_dir: str = ''

  # Whether to lower case the input text. Should be True for uncased models and
  # False for cased models.
  config.do_lower_case: bool = True

  # Model parameters.

  # For pre-training, we only need 2 segment types (for NSP), but we allow up to
  # 4 for GLUE/SuperGLUE fine-tuning.
  config.type_vocab_size: int = 4
  # Embedding dimension for each token.
  config.d_emb: int = 768
  # Hidden dimension of model.
  config.d_model: int = 768
  # Hidden dimension for feed-forward layer.
  config.d_ff: int = 3072
  # The maximum total input sequence length after tokenization. Sequences longer
  # than this will be truncated, and sequences shorter than this will be padded.
  config.max_seq_length: int = 512
  # Number of self-attention heads. Only used for BERT models.
  config.num_heads: int = 12
  # Number of model blocks / layers.
  config.num_layers: int = 12
  # Regular dropout rate, applied throughout model.
  config.dropout_rate: float = 0.1
  # Dropout rate used in mixing module, e.g. self-attention sublayer.
  config.mixing_dropout_rate: float = 0.1

  # Determines how discrete Fourier Transforms are computed. Only used for FNet
  # models. Set to true if running on TPU hardware, in which case matrix
  # multiplications will be favored for relatively shorter input sequences. Set
  # to false for GPU/CPU hardware, in which case FFTs are used for all input
  # sequence lengths.
  config.use_tpu_fourier_optimizations: bool = False

  # Dummy parameter for repeated runs.
  config.trial: int = 0

  return config



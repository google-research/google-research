# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
  F_NET = 'f_net'  # Fourier Transform mixing
  FF_ONLY = 'ff_only'  # Feed forward sublayers only; no token mixing
  LINEAR = 'linear'  # Matrix multiplications with learnable weights
  RANDOM = 'random'  # Constant, random matrix multiplications


class TrainingMode(str, enum.Enum):
  """Determines type of training."""
  PRETRAINING = 'pretraining'
  CLASSIFICATION = 'classification'


class HybridAttentionLayout(str, enum.Enum):
  """Where, in hybrid models, attention sublayers replace mixing sublayers."""
  BOTTOM = 'bottom'  # First mixing sublayers.
  MIDDLE = 'middle'  # Middle mixing sublayers.
  MIXED = 'mixed'  # Interspersed throughout model.
  TOP = 'top'  # Final mixing sublayers.


def get_config():
  """Base config for training models."""
  config = ml_collections.ConfigDict()

  # Determines which model to use.
  # Specific mixing sublayers may be replaced with attention using
  # config.attention_layout and config.num_attention_layers.
  config.model_arch: ModelArchitecture = ModelArchitecture.F_NET

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

  # Initial checkpoint directory or filepath (usually from a pre-trained model).
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

  # Determines whether or not the FFT is used in lieu of matrix multiplications.
  # Only relevant for FNet: If true, favor FFT over matrix multiplications to
  # compute the DFT.
  config.use_fft: bool = True

  # For hybrid models, attention layers replace a subset of the mixing
  # sublayers.
  config.attention_layout: HybridAttentionLayout = HybridAttentionLayout.TOP
  config.num_attention_layers: int = 0

  # Random number generator seed.
  config.seed: int = 0

  # Dummy parameter for repeated runs.
  config.trial: int = 0

  return config



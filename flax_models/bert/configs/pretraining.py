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

"""A config for pre-training BERT on the C4 dataset."""
from typing import Optional
import ml_collections


def get_config():
  """Config for pre-training BERT."""
  config = ml_collections.ConfigDict()

  config.model_name = "bert"
  # mode is either "pretraining" or "classification".
  # TODO(marcvanzee): Turn this into an enum.
  config.mode = "pretraining"
  # Whether to run training.
  config.do_train: bool = True
  # Whether to run eval on the dev set.
  config.do_eval: bool = True
  # Total batch size for training.
  config.train_batch_size: int = 256
  # Total batch size for eval.
  config.eval_batch_size: int = 32
  # The base learning rate for Adam.
  config.learning_rate: float = 1e-4
  # Number of training steps.
  config.num_train_steps: int = 1000000
  # Number of warmup steps.
  config.num_warmup_steps: int = 100000
  # The maximum total input sequence length after WordPiece tokenization.
  # Sequences longer than this will be truncated, and sequences shorter
  # than this will be padded. Must match data generation.
  config.max_seq_length: int = 512
  # Maximum number of masked LM predictions per sequence.
  config.max_predictions_per_seq: int = 80
  # How often to save the model checkpoint.
  config.save_checkpoints_steps: int = 5000
  # Maximum number of eval steps.
  config.max_eval_steps: int = 100
  # Do not start from a pre-trained checkpoint.
  config.init_checkpoint: str = ""
  # Pre-trained vocab file.

  # Model parameters.
  # TODO(marcvanzee): Move this out in a separate config (currently duplicate
  # with configs/pretraining.py).
  config.vocab_size: int = 30000
  config.type_vocab_size: int = 2
  config.d_emb: int = 128
  config.d_model: int = 768
  config.d_ff: int = 3072
  config.max_len: int = 512
  config.num_heads: int = 12
  config.num_layers: int = 12
  config.dropout_rate: float = 0.0
  config.attention_dropout_rate: float = 0.0
  # pylint: disable=g-bare-generic
  # TODO(marcvanzee): Enable these two. I cannot import flax.nn here, and
  # adhoc importing them gives a jaxlib import error.
  # config.hidden_activation: Callable = nn.gelu
  # config.kernel_initializer: Callable = nn.initializers.xavier_uniform()
  config.num_parallel_heads: Optional[int] = None

  config.trial = 0  # Dummy for repeated runs.
  return config


def get_hyper(h):
  return h.product([
      h.sweep("trial", range(1)),
  ], name="config")

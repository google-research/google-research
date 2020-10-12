# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A config for training BERT on SST2."""
from typing import Optional
import ml_collections


def get_config():
  """Config for fine-tuning BERT (classification)."""
  config = ml_collections.ConfigDict()

  config.model_name: str = "bert"
  # mode is either "pretraining" or "classification".
  # TODO(marcvanzee): Turn this into an enum.
  config.mode: str = "classification"
  # This is "glue/DS", where DS is one of the following:
  # [sst2, mrpc, qqp, stsb, mnli, qnli, rte].
  # TODO(marcvanzee): Add the SQuAD dataset.
  config.dataset_name: str = "glue/sst2"
  # Whether to run training.
  config.do_train: bool = True
  # Whether to run eval on the dev set.
  config.do_eval: bool = True
  # Whether to run the model in inference mode on the test set.
  config.do_predict: bool = True
  # Total batch size for training.
  config.train_batch_size: int = 16
  # Total batch size for eval.
  config.eval_batch_size: int = 8
  # Total batch size for predict.
  config.predict_batch_size: int = 8
  # The base learning rate for Adam.
  config.learning_rate: float = 1e-5
  # Total number of training epochs to perform.
  config.num_train_epochs: float = 3.0
  # Proportion of training to perform linear learning rate warmup for.
  # E.g., 0.1 = 10% of training.
  config.warmup_proportion: float = 0.1
  # The maximum total input sequence length after WordPiece tokenization.
  # Sequences longer than this will be truncated, and sequences shorter
  # than this will be padded.
  config.max_seq_length: int = 128
  # Whether to lower case the input text. Should be True for uncased models and
  # False for cased models.
  config.do_lower_case: bool = True

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
  # TODO(marcvanzee): Enable these two, by figuring out how to import flax.nn
  # from this config.
  # config.hidden_activation: Callable = nn.gelu
  # config.kernel_initializer: Callable = nn.initializers.xavier_uniform()
  config.num_parallel_heads: Optional[int] = None

  config.trial = 0  # Dummy for repeated runs.
  return config


def get_hyper(h):
  return h.product([
      h.sweep("trial", range(1)),
  ], name="config")

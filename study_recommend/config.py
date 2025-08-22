# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""A ConfigDict with default parameters for an experiment."""

from typing import Optional
import ml_collections


def get_config(
    vocab,
    per_device_batch_size,
    seq_len,
    working_dir,
    vocab_path = None,
    tensorboard_dir=None,
):
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.model_class = 'individual'
  # Path to load or store vocab file.
  config.vocab_path = vocab_path

  # Token Indices start at 1.
  config.vocab_size = len(vocab) + 1

  config.separator_token = vocab['<SEP>']
  config.oov_token = vocab.oov_value

  # Per device batch size for training.
  config.per_device_batch_size = per_device_batch_size

  # Per device batch size for training.
  config.eval_per_device_batch_size = per_device_batch_size

  config.num_train_steps = 500_000

  # Base learning rate.
  config.learning_rate = 0.0016 * (config.per_device_batch_size // 64)

  # Linear learning rate warmup.
  config.warmup_steps = 1000

  # Decay factor for AdamW style weight decay.
  config.weight_decay = 0.1

  # Maximum length cutoff for training examples.
  config.max_target_length = seq_len

  config.chunk_length = 10
  # Maximum length cutoff for eval examples.
  config.max_eval_target_length = 512
  # Maximum length cutoff for predicted tokens.
  config.max_predict_length = 50

  # Final logit transform uses embedding matrix transpose.
  config.logits_via_embedding = False

  # Number of transformer layers.
  config.num_layers = 6

  # Size of query/key/value for attention.
  config.qkv_dim = 256
  # Size of embeddings.
  config.emb_dim = 256
  # Size of the MLP.
  config.mlp_dim = 1024

  # Number of attention heads.
  config.num_heads = 8

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True

  # Save a checkpoint every these number of steps.
  config.checkpoint_every_steps = 1_000
  # Frequency of eval during training, e.g. every 1_000 steps.
  config.eval_every_steps = 500
  config.log_every_steps = 100

  # Use bfloat16 mixed precision training instead of float32.
  config.use_bfloat16 = True

  # Integer for PRNG random seed.
  config.seed = 0

  # Size of reference batches for tensorboard logging
  config.reference_train_batch_size = 4096 * 16
  config.reference_valid_batch_size = 4096 * 16

  config.working_dir = working_dir
  config.tensorboard_dir = tensorboard_dir
  config.write_summary = True
  config.num_grade_levels = None

  return config.lock()

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

"""Hyperparameter configuration for the language model baseline.

This is the standard ARM version of the model, the way that language models
are typically trained.
"""

import ml_collections


def get_config_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Path to load or store sentencepiece vocab file.
  config.vocab_path = None

  # Choose a model type
  config.model = 'standard_arm'  # standard_arm, permute_arm, ao_arm, bit_ao
  config.elbo_mode = 'uniform'
  config.ce_term = 0.00

  # Default text8 path
  config.text8_path = '/path/to/data'
  config.seq_length = 250
  config.context_length = 0
  config.batch_size = 64
  config.test_batch_size = 200

  # Name of TFDS translation dataset to use.
  config.dataset_name = 'enwik8'   # text8 or enwik8.

  # Sampling temperature for language model inference.
  config.sampling_temperature = 0.6

  # Top k cutoff for logit sampling. If 0 then no top-k cutoff is used.
  config.sampling_top_k = 20

  config.num_train_steps = 1_000_000

  # Number of steps to take during evaluation. Large enough to evaluate all.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  config.num_eval_steps = 2_000
  # Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  config.num_predict_steps = -1

  # Base learning rate.
  config.optimizer = 'adam'
  config.learning_rate = 0.0002
  config.lr_factors = 'constant * linear_warmup'

  # Linear learning rate warmup.
  config.warmup_steps = 5_000

  # Decay factor for AdamW style weight decay.
  config.weight_decay = 0.001

  config.kernel_init = 'xavier'

  # Maximum length cutoff for training examples.
  # config.max_target_length = 128
  # Maximum length cutoff for eval examples.
  # config.max_eval_target_length = 512
  # Maximum length cutoff for predicted tokens.
  # config.max_predict_length = 50

  # Number of transformer layers.
  config.num_layers = 12

  # Size of query/key/value for attention.
  config.qkv_dim = 768
  # Size of embeddings.
  config.emb_dim = 768
  # Size of the MLP.
  config.mlp_dim = 3072

  # Number of attention heads.
  config.num_heads = 12

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True

  config.clip_grad = 0.25

  # Save a checkpoint every these number of steps.
  config.checkpoint_every_steps = 10_000
  # Frequency of eval during training, e.g. every 1_000 steps.
  config.eval_every_steps = 1_000
  config.detailed_eval_every_steps = 50_000

  # Use bfloat16 mixed precision training instead of float32.
  config.use_bfloat16 = False

  # Integer for PRNG random seed.
  config.seed = 0

  return config

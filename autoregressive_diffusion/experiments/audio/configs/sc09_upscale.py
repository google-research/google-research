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

"""Bit upscaling config for the SC09 dataset."""

import ml_collections


D = lambda **kwargs: ml_collections.ConfigDict(kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Dataset configuration.
  config.dataset = D(
      name='speech_commands09',
      train_split='train',
      eval_split='validation',
      test_split='test',
      max_length=16000  # Some audio files are shorter than 16000.
      )
  config.batch_size = 256
  config.eval_batch_size = 512
  config.mask_shape = (16000, 1)

  # Training.
  config.num_train_steps = 1_000_000
  config.beta2 = 0.999
  config.clip_grad = 1000.
  config.weight_decay = 0.
  config.ema_momentum = 0.995
  config.learning_rate = D(
      base_learning_rate=1e-4,
      factors='constant',
      warmup_steps=15000,
      )

  config.label_smoothing = 0.0
  config.ce_term = 0.0001

  config.restore_checkpoints = True
  config.checkpoint_every_steps = 5_000
  config.eval_every_steps = 2_500
  config.log_every_steps = 1_000
  config.sample_every_steps = None
  config.seed = 42

  # Evaluation.
  config.sample_batch_size = 32
  config.num_eval_passes = 32

  # Model.
  config.model = 'bit_ao'
  config.upscale_branch_factor = 2
  config.upscale_mode = 'zero_least_significant'
  config.upscale_direct_parametrization = True
  config.elbo_mode = 'uniform'
  config.output_distribution = 'categorical'
  config.num_mixtures = 30

  # Architecture.
  config.arch = D(
      name='diff_wave',
      config=D(
          num_blocks=36,
          features=256,
          dilation_cycle=12,
          )
      )

  return config

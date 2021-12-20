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

"""Configuration file for autoregressive diffusion on images."""

# pylint: disable=invalid-name

import ml_collections


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.save_every = 2
  config.model = 'bit_ao'  # ao_arm, bit_ao
  # upscale_mode choices: zero_least_significant, augment_least_significant
  config.upscale_mode = 'zero_least_significant'
  config.upscale_direct_parametrization = False
  config.upscale_branch_factor = 4
  config.elbo_mode = 'uniform'  # choices: uniform, antithetic
  config.ce_term = 0.001
  config.learning_rate = 0.0001
  config.beta2 = 0.999  # Turns out to be more stable.
  config.momentum = 0.9
  config.clip_grad = 100.
  config.batch_size = 128
  config.test_batch_size = 200  # Divisible by 8 (TPU pods), divides test 10000
  config.num_epochs = 6000
  config.dataset = 'cifar10'
  config.data_augmentation = False
  config.detailed_eval_every = 50
  config.num_eval_passes = 4
  config.seed = 0
  config.num_samples = 16

  # Only for ao_arm and upscale ardm, not possible for more fancy
  # destruction processes:
  config.output_distribution = 'softmax'
  config.num_mixtures = 30

  config.architecture = D(
      n_channels=256,
      num_res_blocks=32,
      num_heads=1,
      ch_mult=[1],
      attn_resolutions=[32, 16, 14, 8, 7, 4],
      dropout=0.
    )
  return config


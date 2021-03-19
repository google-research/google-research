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

"""Configuration and hyperparameter sweeps."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # General
  config.debug = False
  config.experiment_name = ''  # Set in train.sh
  config.on_gcp = True
  config.local_output_dir = '/tmp'
  config.use_gpu = True
  config.random_seed = 42

  # Model params
  config.model_key = 'resnet18'
  config.cascaded = True
  config.tdl_mode = 'OSD'  # OSD, EWS, noise
  config.tdl_alpha = 0.0
  config.noise_var = 0.0
  config.bn_time_affine = False
  config.bn_time_stats = True

  # Loss params
  config.lambda_val = 0.0
  config.normalize_loss = False

  # Dataset params
  config.dataset_name = 'CIFAR10'  # CIFAR10, CIFAR100, TinyImageNet
  config.val_split = 0.1
  config.split_idxs_root = None
  config.augmentation_noise_type = 'occlusion'
  config.num_workers = 16
  config.drop_last = False

  # Train params
  config.batch_size = 128
  config.epochs = 100
  config.eval_freq = 10
  config.upload_freq = 5

  # Optimizer params
  config.learning_rate = 0.1
  config.momentum = 0.9
  config.weight_decay = 0.0005
  config.nesterov = True

  config.lr_milestones = [30, 60, 120, 150]
  config.lr_schedule_gamma = 0.2

  return config

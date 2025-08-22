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

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.dataset_name = 'cifar10'

  config.num_steps = 1000
  config.batch_size = 128
  config.traj_length = 100
  config.traj_mul_factor = 1.0  # Do not increase traj length.
  config.steps_before_expansion = config.num_steps  # No expansion.
  config.num_optimize_batches = 1
  config.num_decide_batches = 1
  config.seed = 42
  config.skip_decide = True  # Do not run decide phase.

  config.optimize_ucb_alpha = 0  # No CI
  config.decide_ucb_alpha = 0  # No CI

  config.tx_list = [{
      'name': 'sgd',
      'kwargs': {'momentum': 0.9},
  }]
  config.lr_list = [{
      'learning_rate': 0.1,
      'cosine_steps': config.num_steps * config.traj_length,
  }]
  config.loss_temp_list = [1]

  config.nn_model = 'ResNetCF20'
  config.nn_expand_fn = None

  return config


def metrics():
  return [
      'train_acc',
      'eval_loss',
      'eval_acc',
      'train_lcb',
      'train_ucb',
  ]

# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
  config.dataset_name = 'cifar100'

  config.num_steps = -1  # Infinite.
  config.batch_size = 128
  config.traj_length = 10000
  config.traj_mul_factor = 1.0  # Do not increase traj length.
  config.steps_before_expansion = 10
  config.num_optimize_batches = 1  # No optimize
  config.num_decide_batches = 1
  config.seed = 42
  config.skip_decide = False
  config.skip_decide_with_expansion = True

  config.num_passes = 1  # Single-pass
  config.init_model_every_pass = False

  config.optimize_ucb_alpha = 0  # No CI.
  config.decide_ucb_alpha = 0  # No CI.

  config.tx_list = [{
      'name': 'adamw',
      'kwargs': {},
  }]
  config.lr_list = [{
      'learning_rate': 0.001,
  }]
  config.loss_temp_list = [1]

  config.nn_model = 'ResNetCIFAR'
  config.nn_model_num_filters = 16
  config.nn_model_stage_size = 1  # Starting from the tiny model.
  config.nn_expand_fn = 'deepen_resnet_all_blocks'
  config.nn_expand_kwargs = {'morphism': True}

  return config


def sweep(add):
  for steps_before_expansion in [11]:
    add(steps_before_expansion=steps_before_expansion)


def metrics():
  return [
      'train_acc',
      'eval_loss',
      'eval_acc',
      'train_lcb',
      'train_ucb',
      'resnet_num_filters',
      'resnet_num_layers',
      'resnet_num_params',
      'pass_idx',
      'train_selected_loss_idx',
  ]

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
  config.solver = 'linear_regression'
  config.dataset = 'cifar100_i21k'

  # Integer for PRNG random seed.
  config.seed = 42
  config.tuning_mode = False
  config.is_private = True

  config.epsilon = 0.01
  config.alpha = 2.0
  config.reg = 3e-4
  return config


def get_hyper(hyper):
  return hyper.product([
      hyper.sweep(
          'config.seed', [42, 42+2, 42+3, 42+5, 42+7]),
  ])

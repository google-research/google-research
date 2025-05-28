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

"""Config for running GBRT-RealismLoss."""

import configs.config as base_config


def get_config():
  """Returns config object for GBRT-RealismLoss."""
  config = base_config.get_config()
  config.unlock()

  exp = config.experiment_kwargs.config

  exp.num_input_tokens = 6
  exp.num_output_tokens = 4
  exp.num_eval_output_tokens = 15

  # The tokens which appear before the adversarial input.
  exp.prefix = '0 Therefore'
  exp.input_for_classify = None

  # exp.exclude_tokens = []
  exp.exclude_tokens = ['(((', ')))']

  exp.exclude_no_space = False

  training_steps = 1001
  exp.training.steps = training_steps
  config.training_steps = training_steps

  # Learning rate
  # init_value is the value the parameter will have at the start of the exp.
  exp.training.learning_rate.init_value = 0.003
  # end_value is the value the parameter will have at the end of the exp.
  # The value will decay logarithmically.
  exp.training.learning_rate.end_value = 1.5

  # Input gumbel softmax.
  # A lower temp approximates sampling more closely.
  exp.training.input_gs.temp.init_value = 47.0
  exp.training.input_gs.temp.end_value = 0.001
  # The fraction of time the Gumbel Softmax will be soft.
  exp.training.input_gs.soft_train_fract = 0.87

  # Decode gumbel softmax.
  exp.training.decode_gs.temp.init_value = 100.0
  exp.training.decode_gs.temp.end_value = 0.01
  exp.training.decode_gs.soft_train_fract = 5.5

  # Weight to probability mass difference loss term (logit loss has weight 1)
  exp.training.difference_loss_weight.init_value = 0.0
  exp.training.difference_loss_weight.end_value = 1.9

  exp.training.batch_size = 4

  config.lock()

  return config

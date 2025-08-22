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

"""Base config.

The config specifies the hyperparameter values used in the experiment.
This config defines general hyperparameters.
The parameters here are meant to be modified by other configs.
"""

import random

import jax.numpy as jnp
from jaxline import base_config
from ml_collections import config_dict


def get_config():
  """Returns config object for training on borg."""
  config = base_config.get_base_config()

  # Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict()
  exp = config.experiment_kwargs.config = config_dict.ConfigDict()

  config.random_seed = random.getrandbits(32)  # Get a different seed each time.

  exp.checkpoint_dir = None

  # This file contains a mask
  # which is used to filter additional tokens from the vocab.
  exp.vocab_filter_file = None

  # exp.dtype = jnp.bfloat16
  exp.dtype = jnp.float32

  exp.training = config_dict.ConfigDict()

  exp.training.difference_loss_weight = config_dict.ConfigDict()
  exp.training.learning_rate = config_dict.ConfigDict()

  # Input gumbel softmax.
  exp.training.input_gs = config_dict.ConfigDict()
  exp.training.input_gs.temp = config_dict.ConfigDict()

  # Decode gumbel softmax.
  exp.training.decode_gs = config_dict.ConfigDict()
  exp.training.decode_gs.temp = config_dict.ConfigDict()

  exp.evaluation = config_dict.ConfigDict()

  config.checkpoint_dir = '/tmp/gbrt_run'
  config.interval_type = 'steps'
  log_interval = 10
  config.log_train_data_interval = log_interval  # Logging to tensorboard
  config.log_tensors_interval = log_interval  # Logging to console
  config.save_checkpoint_interval = 10
  config.eval_specific_checkpoint_dir = ''
  config.best_model_eval_metric = 'unsafe_prob'
  config.best_model_eval_metric_higher_is_better = True

  config.lock()

  return config

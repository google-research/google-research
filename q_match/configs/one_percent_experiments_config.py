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

"""Default config for 1% experiments."""
import ml_collections


def get_config():
  """Returns the default configuration as a ConfigDict."""

  config = ml_collections.ConfigDict()

  # ****************************************************************************
  # Experiment params
  # ****************************************************************************

  # Directory for the experiment logs.
  config.logdir = '/tmp/tabular_ssl_logs/'

  config.dataset_path = './'

  config.random_seed = 0  # Base seed.
  config.num_trials = 5  # Number of random trials.

  config.model = 'iMixClassifier'

  # Number of parallel calls to use when reading data.
  config.num_parallel_calls = 60
  config.batch_size = 512
  config.supervised_epochs = 500  # Max because early stopping is used.
  config.pretext_epochs = 200
  config.patience = 64
  config.pretext_weight_decay = 0.0
  config.learning_rate = 1e-3
  config.pretext_learning_rate = 1e-5
  config.supervised_weight_decay = 1e-1

  # Specific to algos
  config.support_set_size = 1024  # For Q-Match
  config.corruption_p = 0.4  # The probability of corrupting an input.
  config.query_corruption_p = 0.0
  config.student_temperature = 0.15
  config.temperature = 0.1
  config.label_smoothing = 0.0
  config.use_momentum_encoder = True
  config.tau = 0.

  config.use_gpu = True

  config.datasets = [
      'higgs100k1p',
      'covtype_new_1p',
      'mnist_1p',
      'adult_1p',
  ]

  config.algos = ['supervised_training',
                  'vime_pretext+supervised_training',
                  'tabnet_pretext+supervised_training',
                  'npair_imix_pretext+supervised_training',
                  'q_match_pretext+supervised_training',
                  'dino_pretext+supervised_training',
                  'simsiam_pretext+supervised_training',
                  'vicreg_pretext+supervised_training',
                  'simclr_pretext+supervised_training',
                 ]

  config.strictly_supervised_algos = ['supervised_training']

  return config


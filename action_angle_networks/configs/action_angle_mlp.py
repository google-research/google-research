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

"""Action-angle neural networks with MLP-based encoder and decoders."""

from typing import Any
import ml_collections


def get_hyper(hyper):
  """Defines the hyperparameter sweeps."""
  return hyper.product([
      hyper.sweep('config.regularizations.actions',
                  [1e-1]),
      hyper.sweep('config.regularizations.encoded_decoded_differences',
                  [0., 1e-3, 1e-2, 1e-1, 1e0, 1e1]),
  ])


def get_config():
  """Returns a training configuration."""
  config = ml_collections.ConfigDict()
  config.rng_seed = 0
  config.num_trajectories = 1
  config.single_step_predictions = True
  config.num_samples = 1000
  config.split_on = 'times'
  config.train_split_proportion = 80 / 1000
  config.time_delta = 1.
  config.train_time_jump_range = (1, 10)
  config.test_time_jumps = (1, 2, 5, 10, 20, 50)
  config.num_train_steps = 5000
  config.latent_size = 100
  config.activation = 'relu'
  config.model = 'action-angle-network'
  config.encoder_decoder_type = 'mlp'
  config.polar_action_angles = True
  config.scaler = 'identity'
  config.learning_rate = 1e-3
  config.batch_size = 100
  config.eval_cadence = 50
  config.simulation = 'shm'
  config.regularizations = ml_collections.FrozenConfigDict({
      'actions': 1.,
      'angular_velocities': 0.,
      'encoded_decoded_differences': 0.,
  })
  config.simulation_parameter_ranges = ml_collections.FrozenConfigDict({
      'phi': (0, 0),
      'A': (1, 10),
      'm': (1, 5),
      'w': (0.05, 0.1),
  })
  return config

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

"""DP-MLP hyperparameter configuration."""

from typing import Any
import ml_collections


def get_hyper(hyper):
  """Defines the hyperparameter sweeps."""
  return hyper.product([
      hyper.sweep('config.num_layers', [1, 2, 3, 4]),
      hyper.sweep('config.learning_rate', [1e-3, 2e-3, 3e-3, 5e-3])
  ])


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.dataset = 'ogbn-arxiv'
  config.dataset_path = 'datasets/'
  config.pad_subgraphs_to = 1
  config.multilabel = False
  config.adjacency_normalization = 'inverse-degree'
  config.model = 'mlp'
  config.latent_size = 256
  config.num_layers = 1
  config.activation_fn = 'tanh'
  config.num_classes = 40
  config.max_degree = 1
  config.differentially_private_training = True
  config.num_estimation_samples = 10
  config.l2_norm_clip_percentile = 75
  config.training_noise_multiplier = 3.
  config.num_training_steps = 500
  config.max_training_epsilon = 10
  config.evaluate_every_steps = 50
  config.checkpoint_every_steps = 50
  config.rng_seed = 0
  config.optimizer = 'adam'
  config.learning_rate = 0.003
  config.batch_size = 10000
  return config

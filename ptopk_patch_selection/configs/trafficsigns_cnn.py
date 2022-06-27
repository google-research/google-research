# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
# pylint: disable=line-too-long

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 42
  config.trial = 0  # Dummy for repeated runs.

  config.dataset = "trafficsigns"
  config.train_preprocess_str = "to_float_0_1|pad(ensure_small=(1160, 1480))|random_crop(resolution=(960, 1280))|random_linear_transform((.8,1.2),(-.1,.1),.8)"
  config.eval_preprocess_str = "to_float_0_1"

  # Top-k extraction.
  config.model = "ats-traffic"

  # Same set up as usual.
  config.optimizer = "adam"
  config.learning_rate = 1e-4
  config.gradient_value_clip = 1.
  config.momentum = .9

  config.weight_decay = 1e-4
  config.cosine_decay = True
  config.warmup_ratio = 0.
  config.batch_size = 32
  config.num_train_steps = 70_000

  config.log_loss_every_steps = 50
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 5000

  config.trial = 0  # Dummy for repeated runs.

  return config


def get_sweep(h):
  """Get the hyperparamater sweep."""
  sweeps = []

  sweeps.append(h.sweep("config.seed", range(5)))
  sweeps.append(h.sweep("config.learning_rate", [1e-3, 5e-4, 1e-4, 5e-5]))

  return h.product(sweeps)


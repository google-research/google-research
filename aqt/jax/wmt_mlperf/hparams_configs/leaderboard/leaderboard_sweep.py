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

"""Creates a sweep over all the leaderboard models.

Runs each model with three random seeds.
"""

import copy

import ml_collections

from aqt.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_4bit_weights_and_auto_acts
from aqt.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_4bit_weights_only
from aqt.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_8bit_weights_and_auto_acts
from aqt.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_8bit_weights_only
from aqt.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_bfloat16


def get_config():
  """Returns sweep configuration (see module docstring)."""
  sweep_config = ml_collections.ConfigDict()
  base_configs = [
      full_model_bfloat16.get_config(),
      full_model_8bit_weights_only.get_config(),
      full_model_8bit_weights_and_auto_acts.get_config(),
      full_model_4bit_weights_only.get_config(),
      full_model_4bit_weights_and_auto_acts.get_config()
  ]
  configs = []
  for base_config in base_configs:
    for seed in range(3):
      config = copy.deepcopy(base_config)
      config.random_seed = seed
      config.metadata.hyper_str = f"{config.metadata.hyper_str}_seed={seed}"
      configs.append(config)
  sweep_config.configs = configs
  return sweep_config

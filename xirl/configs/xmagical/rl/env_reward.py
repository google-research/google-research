# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Env reward config."""

from base_configs.rl import get_config as _get_config
from configs.constants import XMAGICALTrainingIterations
from ml_collections import ConfigDict
from utils import copy_config_and_replace


def get_config(embodiment):
  """Parameterize base RL config based on provided embodiment.

  This simply modifies the number of training steps based on presets defined
  in `constants.py`.

  Args:
    embodiment (str): String denoting embodiment name.

  Returns:
    ConfigDict corresponding to given embodiment string.
  """
  config = _get_config()

  possible_configs = dict()
  for emb, iters in XMAGICALTrainingIterations.iteritems():
    possible_configs[emb] = copy_config_and_replace(
        config,
        {"num_train_steps": iters},
    )

  return possible_configs[embodiment]

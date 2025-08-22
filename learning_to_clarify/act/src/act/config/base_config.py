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

import dataclasses

from act.config.data import data_config
from act.config.model import hf_model_config
from act.config.model import model_config
from act.config.training import training_config


@dataclasses.dataclass
class BaseConfig:
  """Base configuration for the ACT Algorithm."""
  policy_model_config: hf_model_config.HFModelConfig
  action_model_config: model_config.ModelConfig
  user_simulator_config: model_config.ModelConfig
  intent_model_config: model_config.ModelConfig
  training_config: training_config.ACTConfig
  preference_model_config: model_config.ModelConfig
  data_config: data_config.DataConfig

@dataclasses.dataclass
class BaseInitializationConfig:
  """Base configuration for the SFT Initialization of the ACT Algorithm."""
  policy_model_config: hf_model_config.HFModelConfig
  training_config: training_config.ACTInitializationConfig
  preference_model_config: model_config.ModelConfig
  data_config: data_config.DataConfig

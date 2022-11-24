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

"""Configs for substrates."""

from typing import AbstractSet, Any, Dict, Tuple
from ml_collections import config_dict

from concept_marl.experiments.meltingpot.substrates import cooking_basics as base_config


def get_config(
    substrate_name):
  if substrate_name not in AVAILABLE_SUBSTRATES:
    raise ValueError(f'Unknown substrate {substrate_name!r}.')

  config, concept_spec = base_config.get_config(substrate_name)
  return config.lock(), concept_spec


AVAILABLE_SUBSTRATES: AbstractSet[str] = frozenset({
    # keep-sorted start
    'cooking_asym',
    'cooking_asym_mini',
    'cooking_basic',
    'cooking_basic_mini',
    'cooking_impassable',
    'cooking_impassable_mini',
    'cooking_nav',
    'cooking_passable',
    'cooking_passable_mini',
    # keep-sorted end
})

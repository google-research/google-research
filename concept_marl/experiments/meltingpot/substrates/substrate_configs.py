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

"""Configs for substrates."""

from typing import AbstractSet, Any, Dict, Tuple
from ml_collections import config_dict

from concept_marl.experiments.meltingpot.substrates import capture_the_flag_mod as capture_config
from concept_marl.experiments.meltingpot.substrates import clean_up_mod as cleaning_config
from concept_marl.experiments.meltingpot.substrates import cooking_basics as cooking_config


def get_config(
    substrate_name):
  """Load config for substrate.

  Args:
    substrate_name: Name of substrate for which to load config.

  Returns:
    Substrate config and concept spec.

  Raises:
    ValueError: Incorrect substrate name.
  """
  if substrate_name in AVAILABLE_COOKING_SUBSTRATES:
    config, concept_spec = cooking_config.get_config(substrate_name)
  elif substrate_name in AVAILABLE_CLEANING_SUBSTRATES:
    config, concept_spec = cleaning_config.get_config(substrate_name)
  elif substrate_name in AVAILABLE_CTF_SUBSTRATES:
    config, concept_spec = capture_config.get_config(substrate_name)
  else:
    raise ValueError(f'Unknown substrate {substrate_name!r}.')

  return config.lock(), concept_spec


AVAILABLE_COOKING_SUBSTRATES: AbstractSet[str] = frozenset({
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

AVAILABLE_CLEANING_SUBSTRATES: AbstractSet[str] = frozenset({
    # keep-sorted start
    'clean_up_mod',
    'clean_up_mod_mini',
    # keep-sorted end
})

AVAILABLE_CTF_SUBSTRATES: AbstractSet[str] = frozenset({
    # keep-sorted start
    'capture_the_flag_mod',
    'capture_the_flag_mod_mini',
    # keep-sorted end
})

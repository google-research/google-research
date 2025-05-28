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

"""Configuration for Cooking Impassable Mini.

This is an extension of the cooking environments from:
meltingpot/python/configs/substrates/collaborative_cooking.py

The structure of this file (and how configs are handled for these environments)
follow the design patterns introduced there.

Same as Cooking Impassable but a miniature version See Cooking Impassable for
full details.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.
"""

from concept_marl.experiments.meltingpot.substrates import cooking_basics as base_config


def get_config():
  """Default config for training on collaborative cooking."""
  config = base_config.get_config("impassable_mini")
  return config

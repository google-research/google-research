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

"""Utility functions for Melting Pot experiments."""

from typing import List

from acme import wrappers as acme_wrappers

from concept_marl.experiments.meltingpot.substrates import substrate_configs
from concept_marl.experiments.meltingpot.wrappers import ma_concept_extraction_wrapper
from concept_marl.experiments.meltingpot.wrappers import meltingpot_cooking_dense_rewards_wrapper as dense_rewards_wrapper
from concept_marl.experiments.meltingpot.wrappers import meltingpot_wrapper
from meltingpot.python.utils.substrates import builder as meltingpot_builder


def make_and_wrap_cooking_environment(
    env_name,
    observation_types,
    action_type = "nested",
    dense_rewards = False,
    episode_length = 1000,
    seed = 612):
  """Returns wrapped meltingpot cooking environment."""

  # init custom MeltingPot environment substrate
  substrate_config, concept_spec = substrate_configs.get_config(env_name)
  substrate_config.lab2d_settings.maxEpisodeLengthFrames = episode_length

  # build meltingpot environment
  environment = meltingpot_builder.builder(**substrate_config, seed=seed)

  # wrappers
  local_obs_types = [ot for ot in observation_types if "WORLD" not in ot]
  global_obs_types = [ot for ot in observation_types if "WORLD" in ot]
  environment = meltingpot_wrapper.MeltingPotWrapper(
      environment=environment,
      local_observation_types=local_obs_types,
      global_observation_types=global_obs_types,
      action_type=action_type)

  # if necessary, add cooking domain-specific pseudorewards
  if dense_rewards:
    environment = dense_rewards_wrapper.MPCookingDenseRewardsWrapper(
        environment, environment.num_agents)

  # concept extraction wrapper (interventions turned off for training)
  environment = ma_concept_extraction_wrapper.MAConceptExtractionWrapper(
      environment,
      environment.num_agents,
      concept_spec=concept_spec,
      intervene=False,
      concept_noise=None,
      concepts_to_intervene=None)

  environment = acme_wrappers.SinglePrecisionWrapper(environment)
  return environment

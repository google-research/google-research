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

"""Utility functions for Melting Pot experiments."""

from typing import Dict, List, Optional, Tuple

from acme import wrappers as acme_wrappers
from meltingpot.utils.substrates import builder as meltingpot_builder

from concept_marl.experiments.meltingpot.substrates import substrate_configs
from concept_marl.experiments.meltingpot.wrappers import ma_concept_extraction_wrapper
from concept_marl.experiments.meltingpot.wrappers import meltingpot_cooking_dense_rewards_wrapper as dense_rewards_wrapper
from concept_marl.experiments.meltingpot.wrappers import meltingpot_pixels_wrapper
from concept_marl.experiments.meltingpot.wrappers import meltingpot_wrapper


ENV_OBS_TYPES = {
    "cooking": [
        "RGB",
        "POSITION",
        "ORIENTATION",
        "WORLD.CONCEPT_AGENT_POSITIONS",
        "WORLD.CONCEPT_AGENT_ORIENTATIONS",
        "WORLD.CONCEPT_TOMATO_POSITIONS",
        "WORLD.CONCEPT_DISH_POSITIONS",
        "WORLD.CONCEPT_COOKING_POT_POSITIONS",
        "WORLD.CONCEPT_DELIVERY_POSITIONS",
        "WORLD.CONCEPT_AGENT_HAS_TOMATO",
        "WORLD.CONCEPT_AGENT_HAS_DISH",
        "WORLD.CONCEPT_AGENT_HAS_SOUP",
        "WORLD.CONCEPT_COOKING_POT_TOMATO_COUNTS",
        "WORLD.CONCEPT_COOKING_POT_PROGRESS",
        "WORLD.CONCEPT_LOOSE_TOMATO_POSITIONS",
        "WORLD.CONCEPT_LOOSE_DISH_POSITIONS",
        "WORLD.CONCEPT_LOOSE_SOUP_POSITIONS"],
    "clean_up": [
        "RGB",
        "POSITION",
        "ORIENTATION",
        "WORLD.CONCEPT_AGENT_POSITIONS",
        "WORLD.CONCEPT_AGENT_ORIENTATIONS",
        "WORLD.CONCEPT_CLOSEST_APPLE_POSITIONS",
        "WORLD.CONCEPT_CLOSEST_POLLUTION_POSITIONS"],
    "capture": [
        "RGB",
        "POSITION",
        "ORIENTATION",
        "WORLD.CONCEPT_AGENT_POSITIONS",
        "WORLD.CONCEPT_AGENT_ORIENTATIONS",
        "WORLD.CONCEPT_AGENT_HEALTH_STATES",
        "WORLD.CONCEPT_FLAG_POSITIONS",
        "WORLD.CONCEPT_AGENT_HAS_FLAG",
        "WORLD.CONCEPT_FLAG_STATE_INDICATOR"]
}


def make_and_wrap_cooking_environment(
    env_name,
    action_type = "nested",
    dense_rewards = False,
    episode_length = 1000,
    seed = 612,
    grayscale = False,
    scale_dims = None,
    intervene = False,
    concepts_to_intervene = None,
    mask_agent_self = False
):
  """Returns wrapped meltingpot cooking environment."""
  observation_types = ENV_OBS_TYPES["cooking"]

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
      action_set=substrate_config.action_set,
      local_observation_types=local_obs_types,
      global_observation_types=global_obs_types,
      action_type=action_type)

  # if necessary, add cooking domain-specific pseudorewards
  if dense_rewards:
    environment = dense_rewards_wrapper.MPCookingDenseRewardsWrapper(
        environment, environment.num_agents)

  # observation scaling (if necessary)
  if scale_dims or grayscale:
    environment = meltingpot_pixels_wrapper.MeltingPotPixelsWrapper(
        environment, grayscale=grayscale, scale_dims=scale_dims)

  # concept extraction wrapper (interventions turned off for training)
  environment = ma_concept_extraction_wrapper.MAConceptExtractionWrapper(
      environment,
      environment.num_agents,
      concept_spec=concept_spec,
      intervene=intervene,
      concept_noise=None,
      concepts_to_intervene=concepts_to_intervene,
      mask_agent_self=mask_agent_self)

  environment = acme_wrappers.SinglePrecisionWrapper(environment)
  return environment


def make_and_wrap_cleanup_environment(
    env_name,
    action_type = "nested",
    dense_rewards = False,
    episode_length = 1000,
    seed = 612,
    grayscale = False,
    scale_dims = None,
    intervene = False,
    concepts_to_intervene = None,
    mask_agent_self = False,
    eat_reward = 0.1,
    clean_reward = 0.005
):
  """Returns wrapped meltingpot cleanup environment."""
  observation_types = ENV_OBS_TYPES["clean_up"]

  # init MeltingPot environment substrate
  substrate_config, concept_spec = substrate_configs.get_config(env_name)
  substrate_config.lab2d_settings.maxEpisodeLengthFrames = episode_length

  if dense_rewards:
    for game_obj in substrate_config.lab2d_settings.simulation.gameObjects:
      for component in game_obj["components"]:
        if component["component"] == "Taste":
          component["kwargs"]["cleanRewardAmount"] = clean_reward
          component["kwargs"]["eatRewardAmount"] = eat_reward

  # build meltingpot environment
  environment = meltingpot_builder.builder(**substrate_config, seed=seed)

  # wrappers
  local_obs_types = [ot for ot in observation_types if "WORLD" not in ot]
  global_obs_types = [ot for ot in observation_types if "WORLD" in ot]
  environment = meltingpot_wrapper.MeltingPotWrapper(
      environment=environment,
      action_set=substrate_config.action_set,
      local_observation_types=local_obs_types,
      global_observation_types=global_obs_types,
      action_type=action_type)

  # observation stacking (if necessary)
  if scale_dims or grayscale:
    environment = meltingpot_pixels_wrapper.MeltingPotPixelsWrapper(
        environment, grayscale=grayscale, scale_dims=scale_dims)

  # interventions turned off for training
  environment = ma_concept_extraction_wrapper.MAConceptExtractionWrapper(
      environment,
      environment.num_agents,
      concept_spec=concept_spec,
      intervene=intervene,
      concept_noise=None,
      concepts_to_intervene=concepts_to_intervene,
      mask_agent_self=mask_agent_self)

  environment = acme_wrappers.SinglePrecisionWrapper(environment)
  return environment


def make_and_wrap_capture_environment(
    env_name,
    action_type = "nested",
    episode_length = 1000,
    seed = 612,
    grayscale = False,
    scale_dims = None,
    intervene = False,
    concepts_to_intervene = None,
    mask_agent_self = False,
):
  """Returns wrapped meltingpot capture the flag environment."""
  observation_types = ENV_OBS_TYPES["capture"]

  # init MeltingPot environment substrate
  substrate_config, concept_spec = substrate_configs.get_config(env_name)
  substrate_config.lab2d_settings.maxEpisodeLengthFrames = episode_length

  # build meltingpot environment
  environment = meltingpot_builder.builder(**substrate_config, seed=seed)

  # wrappers
  local_obs_types = [ot for ot in observation_types if "WORLD" not in ot]
  global_obs_types = [ot for ot in observation_types if "WORLD" in ot]
  environment = meltingpot_wrapper.MeltingPotWrapper(
      environment=environment,
      action_set=substrate_config.action_set,
      local_observation_types=local_obs_types,
      global_observation_types=global_obs_types,
      action_type=action_type)

  # observation stacking (if necessary)
  if scale_dims or grayscale:
    environment = meltingpot_pixels_wrapper.MeltingPotPixelsWrapper(
        environment, grayscale=grayscale, scale_dims=scale_dims)

  # interventions turned off for training
  environment = ma_concept_extraction_wrapper.MAConceptExtractionWrapper(
      environment,
      environment.num_agents,
      concept_spec=concept_spec,
      intervene=intervene,
      concept_noise=None,
      concepts_to_intervene=concepts_to_intervene,
      mask_agent_self=mask_agent_self)

  environment = acme_wrappers.SinglePrecisionWrapper(environment)
  return environment

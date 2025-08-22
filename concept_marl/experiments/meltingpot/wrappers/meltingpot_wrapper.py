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

"""Meltingpot environment wrapped for Acme compatibility."""
from typing import Any, Dict, Iterator, List, Mapping, Tuple, TypeVar

from acme import specs
from acme import types
from acme import wrappers as acme_wrappers
from acme.wrappers import base
import dm_env
from meltingpot import substrate as meltingpot_substrate
from meltingpot.utils.substrates import builder as meltingpot_builder
import numpy as np
from PIL import Image

V = TypeVar("V")


class MeltingPotWrapper(base.EnvironmentWrapper):

  """Wrapper that converts the Meltingpot environment to use dict-indexing.

  Specifically, default meltingpot environment observations are:
    observation = {
      "1.POSITION": Array(shape=(2,), ... ),
      "1.ORIENTATION" : Array(shape=(), ... ),
      "1.RGB" : Array(shape=(40, 40, 3), ... ),
      ...
      "N.POSITION": Array(shape=(2,), ... ),
      "N.ORIENTATION" : Array(shape=(), ... ),
      "N.RGB" : Array(shape=(40, 40, 3), ... )
    }

    They are converted instead to:
      observation = {0: {"POSITION": ..., "ORIENTATION": ..., "RGB": ...}}

    This wrapper is essentially the same as combining:
      1. Meltingpot multiplayer wrapper:
        meltingpot/python/.../wrappers/multiplayer_wrapper.py
      2. Multiagent dict key wrapper:
        multiagent_acme/wrappers/multiagent_dict_key_wrapper.py
    in sequence, but instead converts meltingpot --> dict-index directly
    instead of meltingpot --> list-index --> dict-index.

    Note: Does NOT currently support global observations (e.g. WORLD.RGB)
    as part of observation space.
  """

  def __init__(self, environment,
               action_set,
               action_type,
               local_observation_types,
               global_observation_types):
    """Constructor.

    Args:
      environment: Environment to wrap.
      action_set: Default action set for this meltingpot environment.
      action_type: Type of action to use (nested or flat).
      local_observation_types: Observation types for agent's local observations.
      global_observation_types: Observation types for global observations.
    """
    self._environment = environment
    self._num_players = self._get_num_players()
    self.num_agents = self._num_players
    self.local_observation_types = local_observation_types
    self.global_observation_types = global_observation_types
    self.unflattened_action = {i: z for i, z in enumerate(action_set)}
    self._action_type = action_type

    if self._action_type == "flat":
      self._action_spec = self._convert_actions_to_flat_spec(
          self._environment.action_spec())
    else:
      self._action_spec = self._convert_actions_to_nested_spec(
          self._environment.action_spec())

    self._observation_spec = self._convert_obs_to_spec(
        self._environment.observation_spec())
    self._reward_spec = self._convert_rewards_to_spec(
        self._environment.observation_spec())

  def _get_num_players(self):
    """Returns number of playing agents by extracting maximum agent-index from dmlab2d action spec.

    The agent-indexed MeltingPot action-space for N-agents is:
    action = {
      "1.move": BoundedArray(shape=(), ... ),
      "1.turn" : BoundedArray(shape=(), ... ),
      "1.interaction" : BoundedArray(shape=(), ... ),
      ...
      "N.move": BoundedArray(shape=(), ... ),
      "N.turn" : BoundedArray(shape=(), ... ),
      "N.interaction" : BoundedArray(shape=(), ... ),
    }

    Output: N
    """
    action_spec_keys = self._environment.action_spec().keys()
    lua_player_indices = (int(key.split(".", 1)[0]) for key in action_spec_keys)
    return max(lua_player_indices)

  def _convert_obs_to_spec(self, source):
    """Converts MeltingPot (i.e.dmlab2d) observations to dict-indexed Acme spec."""
    player_observations = {str(i): {} for i in range(self._num_players)}
    for suffix in self.local_observation_types:
      for i, value in self._player_observations(source,
                                                suffix,
                                                self._num_players):
        player_observations[str(i)][suffix] = value
    for name in self.global_observation_types:
      value = source[name]
      for i in range(self._num_players):
        player_observations[str(i)][name] = value

    return player_observations

  def _convert_actions_to_nested_spec(
      self, source):
    """Converts MeltingPot (i.e.dmlab2d) actions to flat Acme spec."""
    action_spec = {str(i): {} for i in range(self._num_players)}
    for key, spec in source.items():
      lua_player_index, suffix = key.split(".", 1)
      player_index = int(lua_player_index) - 1
      action_spec[str(player_index)][suffix] = spec.replace(name=suffix)
    return action_spec

  def _convert_actions_to_flat_spec(
      self, source):
    """Converts MeltingPot (i.e.dmlab2d) actions to nested Acme spec."""

    # unique action values (assuming the same for each agent)
    num_unique_values = np.sum([(spec.maximum - spec.minimum)
                                for key, spec in source.items()
                                if "1." in key]) + 1

    # dtype (assuming the same for each agent and action type)
    dtype = list(source.values())[0].dtype

    action_spec = {
        str(i): specs.DiscreteArray(
            num_values=num_unique_values, dtype=dtype, name="action")
        for i in range(self._num_players)
    }
    return action_spec

  def _convert_rewards_to_spec(
      self, source):
    """Creates Acme rewards from MeltingPot (i.e. dmlab2d) observations."""
    rewards = {str(i): None for i in range(self._num_players)}
    for i, value in self._player_observations(source,
                                              "REWARD",  # pytype: disable=attribute-error  # dynamic-method-lookup
                                              self._num_players):
      rewards[str(i)] = value
    return rewards

  def _process_actions(
      self, source):
    """Converts dict-indexed Acme actions to MeltingPot (i.e.dmlab2d) actions."""
    dmlab2d_actions = {}
    for player_index, action in source.items():
      if self._action_type == "flat":
        # map flat actions to meltingpot Dict
        action = self.unflattened_action[action.item()]

      for key, value in action.items():
        dmlab2d_actions[f"{int(player_index) + 1}.{key}"] = value
    return dmlab2d_actions

  def _player_observations(self,
                           observations,
                           suffix,
                           num_players):
    """Yields observations for each player.

    Args:
      observations: dmlab2d observations source to check.
      suffix: suffix of player key to return.
      num_players: the number of players.

    A KeyError is triggered if either:
      (i) an incorrect number of players is passed in, or
      (ii) an incorrect observation suffix is requested.
      e.g. for the observation:
        observation = {
          "1.POSITION": Array(shape=(2,), ... ),
          "1.RGB" : Array(shape=(40, 40, 3), ... )
        }
      a KeyError exception would be triggered for:
      - num_players = 2, suffix = 'RGB' --> observation['2.RGB']
      - num_players = 1, suffix = 'ORIENTATION' --> observation['1.ORIENTATION']

    This exception is handled silently to ignore additional agents or
    observation types that appear mid-trajectory (rather than terminate
    via KeyError).
    """
    for player_index in range(num_players):
      try:
        value = observations[f"{player_index + 1}.{suffix}"]
      except KeyError:
        pass
      else:
        if isinstance(value, dm_env.specs.Array):
          value = value.replace(name=suffix)
        yield player_index, value    # pytype: disable=bad-return-type  # dynamic-method-lookup

  def _convert_timestep(self, source):
    """Returns multiplayer timestep from dmlab2d observations."""
    return dm_env.TimeStep(
        step_type=source.step_type,
        reward=self._convert_rewards_to_spec(source.observation),
        discount=0. if source.discount is None else source.discount,
        observation=self._convert_obs_to_spec(source.observation))

  @property
  def environment(self):
    """Returns the wrapped environment."""
    return self._environment

  def reset(self):
    timestep = self._environment.reset()
    return self._convert_timestep(timestep)

  def step(self, action):
    action = self._process_actions(action)
    timestep = self._environment.step(action)
    return self._convert_timestep(timestep)

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def reward_spec(self):  # pytype: disable=signature-mismatch
    return self._reward_spec

  def render(self, mode = "rgb_array"):
    img_arr = self._environment.observation()["WORLD.RGB"]
    if mode == "human":
      img = Image.fromarray(img_arr, "RGB")
      img.show()
    return img_arr


def make_and_wrap_meltingpot_environment(
    env_name,
    observation_types,
    action_type = "nested",
    episode_length = 1000,
    seed = 612):
  """Returns wrapped meltingpot environment.

  Args:
    env_name (str): MeltingPot environment to init.
    observation_types (List[str]): Observation types for agents to receive.
    action_type (str): Action type {'nested', 'flat'}
    episode_length (int): Number of time-steps per episode.
    seed (int): Random seed.

  Returns:
    dm_env.Environment: The wrapped MeltingPot environment.
  """

  # init MeltingPot environment substrate
  substrate_config = meltingpot_substrate.get_config(env_name)
  substrate_config.lab2d_settings.maxEpisodeLengthFrames = episode_length

  # config overrides (e.g. dense rewards)
  environment = meltingpot_builder.builder(**substrate_config, seed=seed)

  # wrappers
  local_obs_types = [ot for ot in observation_types if "WORLD" not in ot]
  global_obs_types = [ot for ot in observation_types if "WORLD" in ot]
  environment = MeltingPotWrapper(
      environment=environment,
      action_set=substrate_config.action_set,
      local_observation_types=local_obs_types,
      global_observation_types=global_obs_types,
      action_type=action_type)

  environment = acme_wrappers.SinglePrecisionWrapper(environment)
  return environment

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

"""Dense reward wrapper for Melting Pot cooking environments."""

from typing import Any, Dict

from acme import types
from acme.wrappers import base
import dm_env
import numpy as np


class MPCookingDenseRewardsWrapper(base.EnvironmentWrapper):
  """Dense reward wrapper for Melting Pot cooking environments.

  This wrapper converts sparse rewards from the Meltingpot cooking
  environments to dense rewards based on sub-task progress. Only
  the environment's reward spec is modified.

  Specifically, this wrapper adds the following pseudorewards:
    1) Tomato picking: Small positive reward for putting a tomato into a pot.
    2) Soup cooking: Small positive reward for picking up a bowl while soup is
    cooking.
    3) Soup pick up: Small positive reward for pouring soup into bowl.

  This wrapper assumes that concepts from:
  concept_marl/experiments/meltingpot_utils/cooking_basics.py
  are available.
  """

  def __init__(self,
               environment,
               num_agents,
               agent_idx = 0):
    """Constructor.

    Args:
      environment: Environment to wrap.
      num_agents: Number of agents in the environment.
      agent_idx: agent idx to use for concept extraction (concepts are in same
        order for all agents).
    """
    self._environment = environment
    self.num_agents = num_agents
    self.agent_idx = agent_idx

    self._observation_spec = self._environment.observation_spec()
    self._action_spec = self._environment.action_spec()
    self._reward_spec = self._environment.reward_spec()

    self.previous_observation = None
    self.num_pickups_available = None
    self.num_soups_available = None

  def _densify_rewards(self,
                       source_observation,
                       source_reward,
                       dense_reward=0.1):
    """Creates dense rewards from MeltingPot subtask progress."""

    if self.previous_observation is not None:
      # 1) tomato picking
      # positive reward if tomato added to pot
      # We extract agent/pot positions, has_tomato, and tomato counts from
      # agent_idx's observation (concepts are in same order for all agents).

      # current agent/pot positions
      agent_positions = self._extract_concept_from_observation(
          source_observation, 'WORLD.CONCEPT_AGENT_POSITIONS')
      pot_positions = self._extract_concept_from_observation(
          source_observation, 'WORLD.CONCEPT_COOKING_POT_POSITIONS')

      # has tomatos concepts (previous and current observations)
      prev_has_tomatos = self._extract_concept_from_observation(
          self.previous_observation, 'WORLD.CONCEPT_AGENT_HAS_TOMATO')
      current_has_tomatos = self._extract_concept_from_observation(
          source_observation, 'WORLD.CONCEPT_AGENT_HAS_TOMATO')

      # tomato count concepts (previous and current observations)
      prev_tomato_counts = self._extract_concept_from_observation(
          self.previous_observation, 'WORLD.CONCEPT_COOKING_POT_TOMATO_COUNTS')
      current_tomato_counts = self._extract_concept_from_observation(
          source_observation, 'WORLD.CONCEPT_COOKING_POT_TOMATO_COUNTS')

      # make sure agent is next to the pot that increased in tomato count
      distances = np.sum(
          np.abs(agent_positions - pot_positions[:, np.newaxis]), axis=2)
      close_to_pot = distances <= 1
      tomato_increase = current_tomato_counts == prev_tomato_counts + 1
      tomato_drop = prev_has_tomatos != current_has_tomatos
      tomato_added = tomato_increase & tomato_drop

      # add reward if all tomato picking conditions are met
      tomato_pick_reward = dense_reward * np.asarray(
          [np.any(tomato_added & close) for close in close_to_pot])

      # 2) soup cooking
      # positive reward if dish picked up while soup is cooking
      # we again extract from agent 0's concepts (concepts are in
      # same order for all agents)
      prev_has_dish = self._extract_concept_from_observation(
          self.previous_observation, 'WORLD.CONCEPT_AGENT_HAS_DISH')
      current_has_dish = self._extract_concept_from_observation(
          source_observation, 'WORLD.CONCEPT_AGENT_HAS_DISH')
      prev_cooking_progress = self._extract_concept_from_observation(
          self.previous_observation, 'WORLD.CONCEPT_COOKING_POT_PROGRESS')
      current_cooking_progress = self._extract_concept_from_observation(
          source_observation, 'WORLD.CONCEPT_COOKING_POT_PROGRESS')

      # check if new dish is available (i.e. pot recently started cooking)
      # pot cooks for 20 time-steps, so current_pot_is_cooking will be true
      # for 20 time-steps. We use whether/or not the pot was previously cooking
      # to ensure that this reward is only given out once. If we only used
      # current_pot_is_cooking, agents are incentivized to just pick up and
      # put down bowls and have no reason to get soup out of the bowl.
      prev_pot_pre_cook = prev_cooking_progress == 0.
      current_pot_is_cooking = current_cooking_progress > 0.
      self.num_pickups_available += np.sum(current_pot_is_cooking
                                           & prev_pot_pre_cook)

      # check if dish picked up
      if self.num_pickups_available > 0:
        dish_pickup = [
            bool(current and not prev)
            for current, prev in zip(current_has_dish, prev_has_dish)
        ]

        dish_pickup_reward = dense_reward * np.asarray(
            [np.any(current_pot_is_cooking) & dish for dish in dish_pickup])
      else:
        dish_pickup_reward = np.zeros(self.num_agents)

      # adjust number of dish pickups available if dish picked up
      if np.any(dish_pickup_reward > 0):
        num_dishes_picked_up = np.sum(dish_pickup_reward > 0)
        self.num_pickups_available = max(
            0, self.num_pickups_available - num_dishes_picked_up)

      # 3) soup pick up
      # positive reward if soup picked up

      # check if new soup is available (i.e. pot recently finished cooking)
      prev_pot_not_done_cooking = prev_cooking_progress < 1.
      current_pot_done_cooking = current_cooking_progress == 1.
      self.num_soups_available += np.sum(prev_pot_not_done_cooking
                                         & current_pot_done_cooking)

      # check if soup picked up
      # we again extract from agent 0's concepts (concepts are in
      # same order for all agents)
      if self.num_soups_available > 0:
        prev_has_soup = self._extract_concept_from_observation(
            self.previous_observation, 'WORLD.CONCEPT_AGENT_HAS_SOUP')
        current_has_soup = self._extract_concept_from_observation(
            source_observation, 'WORLD.CONCEPT_AGENT_HAS_SOUP')
        soup_pickup_reward = dense_reward * np.asarray([
            current and not prev
            for current, prev in zip(current_has_soup, prev_has_soup)
        ])
      else:
        soup_pickup_reward = np.zeros(self.num_agents)

      # adjust number of soups available if dish picked up
      if np.any(soup_pickup_reward > 0):
        num_soups_picked_up = np.sum(soup_pickup_reward > 0)
        self.num_soups_available = max(
            0, self.num_soups_available - num_soups_picked_up)

      # override original reward
      for i, agent_id in enumerate(source_reward.keys()):
        source_reward[agent_id] += tomato_pick_reward[i] + dish_pickup_reward[
            i] + soup_pickup_reward[i]

    return source_reward

  def _convert_timestep(self, source):
    """Returns multiplayer timestep from dmlab2d observations."""
    return dm_env.TimeStep(
        step_type=source.step_type,
        reward=self._densify_rewards(source.observation, source.reward),
        discount=source.discount,
        observation=source.observation)

  def _extract_concept_from_observation(self, observation, concept_key):
    return observation[str(self.agent_idx)][concept_key][self.agent_idx]

  @property
  def environment(self):
    """Returns the wrapped environment."""
    return self._environment

  def reset(self):
    timestep = self._convert_timestep(self._environment.reset())
    self.previous_observation = timestep.observation

    # reset cooking reward availability
    self.num_soups_available = 0
    self.num_pickups_available = 0
    return timestep

  def step(self, action):
    timestep = self._convert_timestep(self._environment.step(action))
    self.previous_observation = timestep.observation
    return timestep

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def reward_spec(self):  # pytype: disable=signature-mismatch
    return self._reward_spec

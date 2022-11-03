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

"""Implements the multi-agent stag hunt environments.

One agent must toggle the stag while another agent is adjacent.
"""
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class Stag(minigrid.Box):
  """Stag."""

  def __init__(self, **kwargs):
    super().__init__('green')
    self.toggles = 0

  def can_pickup(self):
    return False

  def can_overlap(self):
    return True


class Plant(minigrid.Ball):
  """Plant."""

  def __init__(self, **kwargs):
    super().__init__('yellow')
    self.toggles = 0

  def can_pickup(self):
    return False

  def can_overlap(self):
    return True


class StagHuntEnv(multigrid.MultiGridEnv):
  """Grid world environment with two competing goals."""

  def __init__(self,
               size=15,
               n_agents=2,
               n_stags=2,
               n_plants=2,
               n_clutter=0,
               penalty=1.0,
               max_steps=250,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      n_agents: The number of agents playing in the world.
      n_stags: The number of stags in the environment.
      n_plants: The number of plants in the environment.
      n_clutter: The number of blocking objects in the environment.
      penalty: Penalty for collecting a stag alone.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      **kwargs: See superclass.
    """
    self.n_clutter = n_clutter
    self.n_stags = n_stags
    self.stags = []
    for _ in range(n_stags):
      self.stags.append(Stag())
    self.plants = []
    for _ in range(n_plants):
      self.plants.append(Plant())
    self.penalty = penalty
    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        fully_observed=True,
        **kwargs)
    self.metrics = {'good_stag': 0, 'bad_stag': 0, 'plant': 0}

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    for stag in self.stags:
      self.place_obj(stag, max_tries=100)
    for plant in self.plants:
      self.place_obj(plant, max_tries=100)
    for _ in range(self.n_clutter):
      self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'Toggle the stag at the same time'

  def move_agent(self, agent_id, new_pos):
    stepped_on = self.grid.get(*new_pos)
    if stepped_on:
      if isinstance(stepped_on, Plant):
        self.metrics['plant'] += 1
        self.rewards[agent_id] += 1
      elif isinstance(stepped_on, Stag):
        good_stag = False
        for i, pos in enumerate(self.agent_pos):
          if i == agent_id:
            continue
          if np.sum(np.abs(pos - new_pos)) == 1:
            good_stag = True
            break
        if good_stag:
          self.metrics['good_stag'] += 1
          self.rewards += 5
        else:
          self.metrics['bad_stag'] += 1
          self.rewards[agent_id] -= self.penalty
      stepped_on.cur_pos = None
    super().move_agent(agent_id, new_pos)

  def step(self, action):
    self.rewards = np.zeros(self.n_agents)
    obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
    for stag in self.stags:
      if stag.cur_pos is None:  # Object has been picked up
        self.place_obj(stag, max_tries=100)
    for plant in self.plants:
      if plant.cur_pos is None:  # Object has been picked up
        self.place_obj(plant, max_tries=100)
    reward = self.rewards.tolist()
    return obs, reward, done, info


class EmptyStagHuntEnv6x6(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_clutter=0, **kwargs)


class EmptyStagHuntEnv7x7(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=7, n_agents=2, n_stags=1, n_plants=2, penalty=0.5, **kwargs)


class EmptyStagHuntEnv8x8(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=2, n_stags=2, n_plants=3, **kwargs)


class RandomStagHuntEnv8x8(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=8, n_agents=2, n_stags=2, n_plants=3, n_clutter=5, **kwargs)


class NoStagHuntEnv8x8(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=2, n_stags=0, n_plants=4, **kwargs)


class AllStagHuntEnv8x8(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=2, n_stags=3, n_plants=0, **kwargs)


class EmptyStagHuntEnv10x10(StagHuntEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=10, n_agents=2, n_stags=2, n_plants=3, n_clutter=0, **kwargs)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-StagHunt-v0', entry_point=module_path + ':StagHuntEnv')

register(
    env_id='MultiGrid-StagHunt-Empty-6x6-v0',
    entry_point=module_path + ':EmptyStagHuntEnv6x6')

register(
    env_id='MultiGrid-StagHunt-Empty-8x8-v0',
    entry_point=module_path + ':EmptyStagHuntEnv8x8')

register(
    env_id='MultiGrid-StagHunt-NoStag-8x8-v0',
    entry_point=module_path + ':NoStagHuntEnv8x8')

register(
    env_id='MultiGrid-StagHunt-AllStag-8x8-v0',
    entry_point=module_path + ':AllStagHuntEnv8x8')

register(
    env_id='MultiGrid-StagHunt-Random-8x8-v0',
    entry_point=module_path + ':RandomStagHuntEnv8x8')

register(
    env_id='MultiGrid-StagHunt-Empty-10x10-v0',
    entry_point=module_path + ':EmptyStagHuntEnv10x10')

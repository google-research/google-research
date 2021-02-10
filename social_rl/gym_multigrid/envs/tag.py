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

# Lint as: python3
"""Implements a minigrid tag environment.

The agents are split into two teams, where one team is rewarded for being
near the other team and the other team has a symmetric penalty.
"""
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class TagEnv(multigrid.MultiGridEnv):
  """Tag grid environment with obstacles, sparse reward."""

  def __init__(self,
               size=15,
               hide_agents=1,
               seek_agents=1,
               n_clutter=25,
               agent_view_size=5,
               max_steps=250,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      hide_agents: The number of agents hiding.
      seek_agents: The number of agents seeking.
      n_clutter: The number of blocking objects in the environment.
      agent_view_size: Unused in this environment.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      **kwargs: See superclass.
    """
    self.n_clutter = n_clutter
    self.hide_agents = hide_agents
    self.seek_agents = seek_agents
    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=hide_agents + seek_agents,
        agent_view_size=size,
        **kwargs)

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)

    for _ in range(self.n_clutter):
      self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'Play tag'

  def step(self, action):
    obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
    reward = [0] * self.n_agents
    for i in range(self.hide_agents):
      for j in range(self.hide_agents, self.hide_agents + self.seek_agents):
        if np.sum(np.abs(self.agent_pos[i] - self.agent_pos[j])) == 1:
          reward[i] -= 10.0
          reward[j] += 10.0
    return obs, reward, done, info


class RandomTagEnv6x6(TagEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=6, hide_agents=1, seek_agents=1, n_clutter=5, **kwargs)


class RandomTagEnv8x8(TagEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=8, hide_agents=2, seek_agents=3, n_clutter=10, **kwargs)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='MultiGrid-Tag-v0', entry_point=module_path + ':TagEnv')

register(
    env_id='MultiGrid-Tag-Random-6x6-v0',
    entry_point=module_path + ':RandomTagEnv6x6')

register(
    env_id='MultiGrid-Tag-Random-8x8-v0',
    entry_point=module_path + ':RandomTagEnv8x8')

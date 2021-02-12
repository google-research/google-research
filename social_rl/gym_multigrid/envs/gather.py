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
"""Implements the multi-agent gather environments.

The agents must pick up (move on top of) items in the environment.
"""
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class GatherEnv(multigrid.MultiGridEnv):
  """Object gathering environment."""

  def __init__(self,
               size=15,
               n_agents=3,
               n_goals=3,
               n_clutter=0,
               n_colors=1,
               random_colors=False,
               max_steps=250,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      n_agents: The number of agents playing in the world.
      n_goals: The number of coins in the environment.
      n_clutter: The number of blocking objects in the environment.
      n_colors: The number of different object colors.
      random_colors: If true, each color has a random number of coins assigned.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      **kwargs: See superclass.
    """
    self.n_clutter = n_clutter
    self.n_goals = n_goals
    self.n_colors = n_colors
    self.random_colors = random_colors
    if n_colors >= len(minigrid.IDX_TO_COLOR):
      raise ValueError('Too many colors requested')

    self.collected_colors = [0] * n_colors
    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        fully_observed=True,
        **kwargs)
    self.metrics = {'max_gathered': 0, 'other_gathered': 0}

  def reset(self):
    self.collected_colors = [0] * self.n_colors
    return super(GatherEnv, self).reset()

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    self.objects = []
    self.colors = (np.random.choice(
        len(minigrid.IDX_TO_COLOR) - 1, size=self.n_colors, replace=False) +
                   1).tolist()
    for i in range(self.n_goals):
      if self.random_colors:
        color = minigrid.IDX_TO_COLOR[np.random.choice(self.colors)]
      else:
        color = minigrid.IDX_TO_COLOR[self.colors[i % self.n_colors]]
      self.objects.append(minigrid.Ball(color=color))
      self.place_obj(self.objects[i], max_tries=100)
    for _ in range(self.n_clutter):
      self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'pick up objects'

  def step(self, action):
    obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
    reward = [0] * self.n_agents
    for i, obj in enumerate(self.carrying):
      if obj:
        color_idx = self.colors.index(minigrid.COLOR_TO_IDX[obj.color])
        self.collected_colors[color_idx] += 1
        if max(self.collected_colors) == self.collected_colors[color_idx]:
          reward[i] += 1
          self.metrics['max_gathered'] += 1
        else:
          self.metrics['other_gathered'] += 1
        self.place_obj(obj, max_tries=100)
        self.carrying[i] = None
    return obs, reward, done, info


class EmptyGatherEnv6x6(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_agents=3, n_goals=3, n_clutter=0, **kwargs)


class RandomGatherEnv8x8(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=3, n_goals=3, n_clutter=5, **kwargs)


class RandomGatherEnv10x10(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(size=10, n_agents=3, n_goals=3, n_clutter=10, **kwargs)


class EmptyColorGatherEnv6x6(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=6, n_agents=2, n_goals=4, n_clutter=0, n_colors=2, **kwargs)


class RandomColorGatherEnv8x8(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=8, n_agents=2, n_goals=4, n_clutter=5, n_colors=2, **kwargs)


class EmptyColorGatherEnv10x10(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=10, n_agents=2, n_goals=6, n_clutter=0, n_colors=3, **kwargs)


class EmptyColorGatherEnv12x12(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=12, n_agents=3, n_goals=9, n_clutter=0, n_colors=3, **kwargs)


class RandomCountsColorGatherEnv12x12(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=12,
        n_agents=3,
        n_goals=9,
        n_clutter=0,
        n_colors=3,
        random_colors=True,
        **kwargs)


class EmptyColorGatherEnv15x15(GatherEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=15, n_agents=3, n_goals=12, n_clutter=0, n_colors=4, **kwargs)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='MultiGrid-Gather-v0', entry_point=module_path + ':GatherEnv')

register(
    env_id='MultiGrid-Gather-Empty-6x6-v0',
    entry_point=module_path + ':EmptyGatherEnv6x6')

register(
    env_id='MultiGrid-Gather-Random-8x8-v0',
    entry_point=module_path + ':RandomGatherEnv8x8')

register(
    env_id='MultiGrid-Gather-Random-10x10-v0',
    entry_point=module_path + ':RandomGatherEnv10x10')

register(
    env_id='MultiGrid-Color-Gather-Empty-6x6-v0',
    entry_point=module_path + ':EmptyColorGatherEnv6x6')

register(
    env_id='MultiGrid-Color-Gather-Random-8x8-v0',
    entry_point=module_path + ':RandomColorGatherEnv8x8')

register(
    env_id='MultiGrid-Color-Gather-Empty-10x10-v0',
    entry_point=module_path + ':EmptyColorGatherEnv10x10')

register(
    env_id='MultiGrid-Color-Gather-Empty-12x12-v0',
    entry_point=module_path + ':EmptyColorGatherEnv12x12')

register(
    env_id='MultiGrid-Color-Gather-RandomCountsColorGatherEnv12x12-12x12-v0',
    entry_point=module_path + ':RerandomColorGatherEnv12x12')

register(
    env_id='MultiGrid-Color-Gather-Empty-15x15-v0',
    entry_point=module_path + ':EmptyColorGatherEnv15x15')

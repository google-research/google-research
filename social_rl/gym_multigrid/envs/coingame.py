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

"""Implements the multi-agent coingame environments.

The agents must pick up (move adjacent to) coins in the environment. In each
round, each agent is assigned a color. The agents are rewarded for picking up
their color or teammates' colors.
"""
import gym
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class Coin(minigrid.Ball):
  """Coin."""

  def __init__(self, color='red', **kwargs):
    super().__init__(color=color)

  def can_pickup(self):
    return False

  def can_overlap(self):
    return True


class CoinGameEnv(multigrid.MultiGridEnv):
  """Coin gathering environment."""

  def __init__(self,
               size=15,
               n_agents=2,
               n_goals=3,
               n_clutter=0,
               n_colors=3,
               max_steps=20,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      n_agents: The number of agents playing in the world.
      n_goals: The number of coins in the environment.
      n_clutter: The number of blocking objects in the environment.
      n_colors: The number of different coin colors.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      **kwargs: See superclass.
    """
    self.n_clutter = n_clutter
    self.n_goals = n_goals
    self.n_colors = n_colors
    self.objects = []
    if n_colors >= len(minigrid.IDX_TO_COLOR):
      raise ValueError('Too many colors requested')

    for i in range(n_goals):
      color = minigrid.IDX_TO_COLOR[i % n_colors]
      self.objects.append(Coin(color=color))
    self.agent_colors = [minigrid.IDX_TO_COLOR[i] for i in range(n_colors)]
    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        fully_observed=True,
        **kwargs)
    if self.minigrid_mode:
      self.position_obs_space = gym.spaces.Box(
          low=0, high=max(size, n_colors), shape=(2 + n_colors,), dtype='uint8')
    else:
      self.position_obs_space = gym.spaces.Box(
          low=0,
          high=max(size, n_colors),
          shape=(self.n_agents, 2 + n_colors),
          dtype='uint8')

    self.observation_space = gym.spaces.Dict({
        'image': self.image_obs_space,
        'direction': self.direction_obs_space,
        'position': self.position_obs_space
    })
    self.metrics = {'self_pickups': 0, 'friend_pickups': 0, 'wrong_pickups': 0}

  def _get_color_obs(self, obs):
    for i in range(self.n_agents):
      color = np.zeros(self.n_colors)
      color[minigrid.COLOR_TO_IDX[self.agent_colors[i]]] = 1
      if self.minigrid_mode:
        obs['position'] = np.concatenate((obs['position'], color))
      else:
        obs['position'][i] = np.concatenate((obs['position'][i], color))
    return obs

  def reset(self):
    np.random.shuffle(self.agent_colors)
    obs = super(CoinGameEnv, self).reset()
    return self._get_color_obs(obs)

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    for i in range(self.n_goals):
      self.place_obj(self.objects[i], max_tries=100)
    for _ in range(self.n_clutter):
      self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'pick up coins corresponding to your color'

  def move_agent(self, agent_id, new_pos):
    stepped_on = self.grid.get(*new_pos)
    if stepped_on:
      stepped_on.cur_pos = None
      for j, c in enumerate(self.agent_colors):
        if stepped_on.color == c:
          if j == agent_id:
            self._reward += 1
            self.metrics['self_pickups'] += 1
          elif j < self.n_agents:
            self._reward += 1
            self.metrics['friend_pickups'] += 1
          else:
            self._reward -= 1
            self.metrics['wrong_pickups'] += 1
          break
    super().move_agent(agent_id, new_pos)

  def step(self, action):
    self._reward = 0
    obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
    obs = self._get_color_obs(obs)
    for obj in self.objects:
      if obj.cur_pos is None:  # Object has been picked up
        self.place_obj(obj, max_tries=100)

    reward = [self._reward] * self.n_agents
    return obs, reward, done, info


class EmptyCoinGameEnv10x10Minigrid(CoinGameEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=10,
        n_agents=1,
        n_goals=2,
        n_colors=2,
        n_clutter=0,
        minigrid_mode=True,
        **kwargs)


class EmptyCoinGameEnv10x10(CoinGameEnv):

  def __init__(self, **kwargs):
    super().__init__(size=10, n_agents=2, n_goals=12, n_clutter=0, **kwargs)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-CoinGame-v0', entry_point=module_path + ':CoinGameEnv')

register(
    env_id='MultiGrid-CoinGame-Empty-6x6-Minigrid-v0',
    entry_point=module_path + ':EmptyCoinGameEnv10x10Minigrid')

register(
    env_id='MultiGrid-CoinGame-Empty-10x10-v0',
    entry_point=module_path + ':EmptyCoinGameEnv10x10')

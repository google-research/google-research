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
"""Implements the multi-agent meetup environments.

The agents must meet at one of several predetermined locations.
"""
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class MeetupEnv(multigrid.MultiGridEnv):
  """Meetup environment."""

  def __init__(self,
               size=15,
               n_agents=3,
               n_goals=3,
               n_clutter=0,
               agent_view_size=5,
               max_steps=250,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      n_agents: The number of agents playing in the world.
      n_goals: The number of goals in the environment.
      n_clutter: The number of blocking objects in the environment.
      agent_view_size: Unused in this environment.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      **kwargs: See superclass.
    """
    self.n_clutter = n_clutter
    self.n_goals = n_goals
    self.goal_pos = [None] * n_goals
    self.past_goal_dist = None
    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        agent_view_size=size,
        **kwargs)
    self.metrics['reached_goal'] = 0

  def reset(self):
    obs = super(MeetupEnv, self).reset()
    self.past_goal_dist = self.get_dist()
    return obs

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    for i in range(self.n_goals):
      pos = self.place_obj(
          multigrid.Door(color='red', is_locked=True), max_tries=100)
      self.goal_pos[i] = pos
    for _ in range(self.n_clutter):
      self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'meet up'

  def get_dist(self):
    dist = np.zeros((self.n_agents, self.n_goals))
    for i, goal in enumerate(self.goal_pos):
      for j, agent in enumerate(self.agent_pos):
        dist[j, i] = np.sum(np.abs(goal - agent))
    goal_dist = np.sum(dist, axis=0)
    return dist[:, np.argmin(goal_dist)]

  def step(self, action):
    obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
    goal_dist = self.get_dist()
    reward = (self.past_goal_dist - goal_dist).tolist()
    if np.sum(goal_dist) == self.n_agents:
      reward = [r + 1 for r in reward]
      self.metrics['reached_goal'] += 1
      done = True
    self.past_goal_dist = goal_dist
    return obs, reward, done, info


class EmptyMeetupEnv6x6(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_agents=3, n_goals=3, n_clutter=0, **kwargs)


class SingleTargetMeetupEnv6x6Minigrid(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=6,
        n_agents=1,
        n_goals=1,
        n_clutter=0,
        minigrid_mode=True,
        **kwargs)


class EmptyMeetupEnv6x6Minigrid(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=6,
        n_agents=1,
        n_goals=3,
        n_clutter=0,
        minigrid_mode=True,
        **kwargs)


class SingleMeetupEnv6x6(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_agents=3, n_goals=1, n_clutter=0, **kwargs)


class RandomMeetupEnv8x8(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=3, n_goals=3, n_clutter=5, **kwargs)


class RandomMeetupEnv8x8Minigrid(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(
        size=8,
        n_agents=1,
        n_goals=3,
        n_clutter=5,
        minigrid_mode=True,
        **kwargs)


class SingleMeetupEnv8x8(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=3, n_goals=1, n_clutter=5, **kwargs)


class RandomMeetupEnv10x10(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=10, n_agents=3, n_goals=3, n_clutter=10, **kwargs)


class EmptyMeetupEnv12x12(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=12, n_agents=3, n_goals=3, n_clutter=0, **kwargs)


class EmptyMeetupEnv15x15(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=15, n_agents=3, n_goals=3, n_clutter=0, **kwargs)


class RandomMeetupEnv12x12(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=12, n_agents=3, n_goals=3, n_clutter=10, **kwargs)


class SingleMeetupEnv12x12(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=12, n_agents=3, n_goals=1, n_clutter=0, **kwargs)


class MultiMeetupEnv12x12(MeetupEnv):

  def __init__(self, **kwargs):
    super().__init__(size=12, n_agents=3, n_goals=5, n_clutter=0, **kwargs)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='MultiGrid-Meetup-v0', entry_point=module_path + ':MeetupEnv')

register(
    env_id='MultiGrid-Meetup-Empty-6x6-v0',
    entry_point=module_path + ':EmptyMeetupEnv6x6')

register(
    env_id='MultiGrid-Meetup-SingleTarget-6x6-Minigrid-v0',
    entry_point=module_path + ':SingleTargetMeetupEnv6x6Minigrid')

register(
    env_id='MultiGrid-Meetup-Empty-6x6-Minigrid-v0',
    entry_point=module_path + ':EmptyMeetupEnv6x6Minigrid')

register(
    env_id='MultiGrid-Meetup-Single-6x6-v0',
    entry_point=module_path + ':SingleMeetupEnv6x6')

register(
    env_id='MultiGrid-Meetup-Random-8x8-v0',
    entry_point=module_path + ':RandomMeetupEnv8x8')

register(
    env_id='MultiGrid-Meetup-Random-8x8-Minigrid-v0',
    entry_point=module_path + ':RandomMeetupEnv8x8Minigrid')

register(
    env_id='MultiGrid-Meetup-Single-8x8-v0',
    entry_point=module_path + ':SingleMeetupEnv8x8')

register(
    env_id='MultiGrid-Meetup-Random-10x10-v0',
    entry_point=module_path + ':RandomMeetupEnv10x10')

register(
    env_id='MultiGrid-Meetup-Empty-12x12-v0',
    entry_point=module_path + ':EmptyMeetupEnv12x12')

register(
    env_id='MultiGrid-Meetup-Empty-15x15-v0',
    entry_point=module_path + ':EmptyMeetupEnv15x15')

register(
    env_id='MultiGrid-Meetup-Random-12x12-v0',
    entry_point=module_path + ':RandomMeetupEnv12x12')

register(
    env_id='MultiGrid-Meetup-Single-12x12-v0',
    entry_point=module_path + ':SingleMeetupEnv12x12')

register(
    env_id='MultiGrid-Meetup-Multi-12x12-v0',
    entry_point=module_path + ':MultiMeetupEnv12x12')

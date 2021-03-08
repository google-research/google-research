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
"""Implements the multi-agent version of minigrid empty environments.

These have a goal which can be fixed or at a random location, and are otherwise
empty.
"""
import math
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class EmptyEnv(multigrid.MultiGridEnv):
  """Empty grid environment, no obstacles, sparse reward."""

  def __init__(self, n_agents=2, size=5, agent_start='fixed', agent_view_size=5,
               randomize_goal=False, minigrid_mode=False, **kwargs):
    self.randomize_goal = randomize_goal
    if agent_start == 'fixed':
      assert n_agents < size - 2, "Can't fit so many agents in fixed position"
      self.agent_start_pos = [np.array([1, i]) for i in range(1, n_agents+1)]
      self.agent_start_dir = [0] * n_agents
      fixed_environment = True
    else:
      self.agent_start_pos = None
      fixed_environment = False

    super().__init__(
        n_agents=n_agents,
        grid_size=size,
        agent_view_size=agent_view_size,
        max_steps=2*size*size,
        see_through_walls=True,  # Set this to True for maximum speed
        fixed_environment=fixed_environment,
        minigrid_mode=minigrid_mode,
        **kwargs
    )

  def _gen_grid(self, width, height):
    # Create an empty grid
    self.grid = multigrid.Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    if self.randomize_goal:
      self.place_obj(minigrid.Goal(), max_tries=100)
    else:
      # Place a goal square in the bottom-right corner
      self.put_obj(minigrid.Goal(), width - 2, height - 2)

    # Place the agents
    self.place_agent()

    self.mission = 'get to the green goal square'

  def place_one_agent(self,
                      agent_id,
                      top=None,
                      size=None,
                      rand_dir=True,
                      max_tries=math.inf,
                      agent_obj=None):
    """Set the agent's starting point at an empty position in the grid."""

    if self.agent_start_pos is not None:
      # Move any other agents in this one's start spot back to their own start
      pos = self.agent_start_pos[agent_id]
      other_agent = self.grid.get(pos[0], pos[1])
      if other_agent:
        self.place_one_agent(other_agent.agent_id, agent_obj=other_agent)

      # Agents always start in the same location
      self.agent_pos[agent_id] = pos
      self.agent_dir[agent_id] = self.agent_start_dir[agent_id]
    else:
      # Randomly place agent
      self.agent_pos[agent_id] = None
      pos = self.place_obj(None, top, size, max_tries=max_tries)
      self.agent_pos[agent_id] = pos

      if rand_dir:
        self.agent_dir[agent_id] = self._rand_int(0, 4)

    # Place the agent object into the grid
    if not agent_obj:
      agent_obj = multigrid.Agent(agent_id, self.agent_dir[agent_id])
      agent_obj.init_pos = pos
    else:
      agent_obj.dir = self.agent_dir[agent_id]
    agent_obj.cur_pos = pos
    self.grid.set(pos[0], pos[1], agent_obj)

    return pos


class EmptyRandomEnv5x5(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(agent_start='random', **kwargs)


class EmptyEnv8x8(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=3, size=8, **kwargs)


class EmptyRandomEnv8x8(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=3, size=8, agent_start='random', **kwargs)


class EmptyEnv16x16(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=5, size=16, **kwargs)


class EmptyRandomEnv6x6(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=5, size=16, agent_start='random', **kwargs)


class EmptyEnv5x5Single(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, **kwargs)


class EmptyRandomEnv6x6Minigrid(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, size=6, agent_view_size=5,
                     agent_start='random', randomize_goal=True,
                     minigrid_mode=True, **kwargs)


class EmptyRandomEnv15x15Minigrid(EmptyEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, size=15, agent_view_size=5,
                     agent_start='random', randomize_goal=True,
                     minigrid_mode=True)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-Empty-5x5-v0',
    entry_point=module_path + ':EmptyEnv'
)

register(
    env_id='MultiGrid-Empty-Random-5x5-v0',
    entry_point=module_path + ':EmptyRandomEnv5x5'
)

register(
    env_id='MultiGrid-Empty-8x8-v0',
    entry_point=module_path + ':EmptyEnv8x8'
)

register(
    env_id='MultiGrid-Empty-Random-8x8-v0',
    entry_point=module_path + ':EmptyRandomEnv8x8'
)

register(
    env_id='MultiGrid-Empty-16x16-v0',
    entry_point=module_path + ':EmptyEnv16x16'
)

register(
    env_id='MultiGrid-Empty-Random-16x16-v0',
    entry_point=module_path + ':EmptyRandomEnv16x16'
)

register(
    env_id='MultiGrid-Empty-5x5-Single-v0',
    entry_point=module_path + ':EmptyEnv5x5Single'
)

register(
    env_id='MultiGrid-Empty-Random-6x6-Minigrid-v0',
    entry_point=module_path + ':EmptyRandomEnv6x6Minigrid'
)

register(
    env_id='MultiGrid-Empty-Random-15x15-Minigrid-v0',
    entry_point=module_path + ':EmptyRandomEnv15x15Minigrid'
)

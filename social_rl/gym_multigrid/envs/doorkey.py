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

"""Implements the multi-agent version of minigrid doorkey environments.

These have a goal on the other side of a door that must be opened with a key.
"""
import math

import gym_minigrid.minigrid as minigrid
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class DoorKeyEnv(multigrid.MultiGridEnv):
  """Environment with a door and key, sparse reward."""

  def __init__(self, size=8, n_agents=3, **kwargs):
    super().__init__(
        grid_size=size, max_steps=10 * size * size, n_agents=n_agents, **kwargs)

  def _gen_grid(self, width, height):
    self.height = height

    # Create an empty grid
    self.grid = multigrid.Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    # Place a goal in the bottom-right corner
    self.put_obj(minigrid.Goal(), width - 2, height - 2)

    # Create a vertical splitting wall
    if width <= 5:
      start_idx = 2
    else:
      start_idx = 3
    self.split_idx = self._rand_int(start_idx, width - 2)
    self.grid.vert_wall(self.split_idx, 0)

    # Place the agent at a random position and orientation
    # on the left side of the splitting wall
    self.place_agent(size=(self.split_idx, height))

    # Place a door in the wall
    door_idx = self._rand_int(1, width - 2)
    self.put_obj(multigrid.Door('yellow', is_locked=True),
                 self.split_idx, door_idx)

    # Place a yellow key on the left side
    self.place_obj(
        obj=minigrid.Key('yellow'), top=(0, 0), size=(self.split_idx, height))

    self.mission = 'Use the key to open the door and then get to the goal'

  def place_one_agent(self,
                      agent_id,
                      top=None,
                      size=None,
                      rand_dir=True,
                      max_tries=math.inf,
                      agent_obj=None):
    """Override so that agents are always placed on other side of the door."""
    if size is None:
      size = (self.split_idx, self.height)

    self.agent_pos[agent_id] = None
    pos = self.place_obj(None, top, size, max_tries=max_tries)

    self.place_agent_at_pos(agent_id, pos, agent_obj=agent_obj,
                            rand_dir=rand_dir)

    return pos


class DoorKeyEnv6x6(DoorKeyEnv):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_agents=2, **kwargs)


class DoorKeyEnv16x16(DoorKeyEnv):

  def __init__(self, **kwargs):
    super().__init__(size=16, n_agents=5, **kwargs)


class DoorKeyEnv5x5Single(DoorKeyEnv):

  def __init__(self, **kwargs):
    super().__init__(size=5, n_agents=1, **kwargs)


class DoorKeyEnv6x6Single(DoorKeyEnv):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_agents=1, **kwargs)


class DoorKeyEnv8x8Single(DoorKeyEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=1, **kwargs)


class DoorKeyEnv16x16Single(DoorKeyEnv):

  def __init__(self, **kwargs):
    super().__init__(size=16, n_agents=1, **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-DoorKey-6x6-v0',
    entry_point=module_path + ':DoorKeyEnv6x6'
)

register(
    env_id='MultiGrid-DoorKey-8x8-v0',
    entry_point=module_path + ':DoorKeyEnv'
)

register(
    env_id='MultiGrid-DoorKey-16x16-v0',
    entry_point=module_path + ':DoorKeyEnv16x16'
)

register(
    env_id='MultiGrid-DoorKey-5x5-Single-v0',
    entry_point=module_path + ':DoorKeyEnv5x5Single'
)

register(
    env_id='MultiGrid-DoorKey-6x6-Single-v0',
    entry_point=module_path + ':DoorKeyEnv6x6Single'
)

register(
    env_id='MultiGrid-DoorKey-8x8-Single-v0',
    entry_point=module_path + ':DoorKeyEnv8x8Single'
)

register(
    env_id='MultiGrid-DoorKey-16x16-Single-v0',
    entry_point=module_path + ':DoorKeyEnv16x16Single'
)

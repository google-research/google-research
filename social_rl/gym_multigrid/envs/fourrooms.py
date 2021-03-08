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
"""Implements the multi-agent version of minigrid four rooms environments.

This environment is a classic exploration problem where the goal must be located
in one of four rooms.
"""

import gym_minigrid.minigrid as minigrid
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class FourRoomsEnv(multigrid.MultiGridEnv):
  """Classic 4 rooms gridworld environment.

  Can specify agent and goal position, if not it is set at random.
  """

  def __init__(self, agent_pos=None, goal_pos=None, n_agents=5, grid_size=19,
               agent_view_size=7, two_rooms=False, minigrid_mode=False,
               **kwargs):
    """Constructor.

    Args:
        agent_pos: An array of agent positions. Length should match n_agents.
        goal_pos: An (x,y) position.
        n_agents: The number of agents in the environment.
        grid_size: The height and width of the grid.
        agent_view_size: The width of the agent's field of view in grid squares.
        two_rooms: If True, will only build the vertical wall.
        minigrid_mode: If True, observations come back without the multi-agent
          dimension.
        **kwargs: See superclass.
    """
    self._agent_default_pos = agent_pos
    self._goal_default_pos = goal_pos
    self.two_rooms = two_rooms
    super().__init__(grid_size=grid_size, max_steps=100, n_agents=n_agents,
                     agent_view_size=agent_view_size,
                     minigrid_mode=minigrid_mode, **kwargs)

  def _gen_grid(self, width, height):
    # Create the grid
    self.grid = multigrid.Grid(width, height)

    # Generate the surrounding walls
    self.grid.horz_wall(0, 0)
    self.grid.horz_wall(0, height - 1)
    self.grid.vert_wall(0, 0)
    self.grid.vert_wall(width - 1, 0)

    room_w = width // 2
    room_h = height // 2

    # For each row of rooms
    for j in range(0, 2):
      # For each column
      for i in range(0, 2):
        x_left = i * room_w
        y_top = j * room_h
        x_right = x_left + room_w
        y_bottom = y_top + room_h

        # Vertical wall and door
        if i + 1 < 2:
          self.grid.vert_wall(x_right, y_top, room_h)
          if not (j == 1 and self.two_rooms and height < 7):
            pos = (x_right, self._rand_int(y_top + 1, y_bottom))
            if not (pos[0] <= 1 or pos[0] >= width -1 or
                    pos[1] <= 0 or pos[1] >= height -1):
              self.grid.set(*pos, None)

        # Horizontal wall and door
        if not self.two_rooms:
          if j + 1 < 2:
            self.grid.horz_wall(x_left, y_bottom, room_w)
            pos = (self._rand_int(x_left + 1, x_right), y_bottom)
            if not (pos[0] <= 1 or pos[0] >= width -1 or
                    pos[1] <= 0 or pos[1] >= height -1):
              self.grid.set(*pos, None)

    # Randomize the player start position and orientation
    if self._agent_default_pos is not None:
      self.agent_pos = self._agent_default_pos
      self.grid.set(*self._agent_default_pos, None)
      self.agent_dir = self._rand_int(0, 4)  # random start direction
    else:
      self.place_agent()

    if self._goal_default_pos is not None:
      goal = minigrid.Goal()
      self.put_obj(goal, *self._goal_default_pos)
      goal.init_pos, goal.cur_pos = self._goal_default_pos
    else:
      self.place_obj(minigrid.Goal())

    self.mission = 'Reach the goal'

  def step(self, action):
    obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
    return obs, reward, done, info


class FourRoomsEnv15x15(FourRoomsEnv):

  def __init__(self, **kwargs):
    super().__init__(grid_size=15, agent_view_size=5, n_agents=3, **kwargs)


class FourRoomsEnvSingle(FourRoomsEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, **kwargs)


class TwoRoomsEnvMinigrid(FourRoomsEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, minigrid_mode=True, agent_view_size=5,
                     grid_size=15, two_rooms=True, **kwargs)


class FourRoomsEnvMinigrid(FourRoomsEnv):

  def __init__(self, **kwargs):
    super().__init__(grid_size=15, agent_view_size=5, minigrid_mode=True,
                     n_agents=1, **kwargs)


class MiniTwoRoomsEnvMinigrid(FourRoomsEnv):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, minigrid_mode=True, agent_view_size=5,
                     grid_size=6, two_rooms=True, **kwargs)


class MiniFourRoomsEnvMinigrid(FourRoomsEnv):

  def __init__(self, **kwargs):
    super().__init__(grid_size=6, agent_view_size=5, minigrid_mode=True,
                     n_agents=1, **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-FourRooms-v0',
    entry_point=module_path + ':FourRoomsEnv'
)

register(
    env_id='MultiGrid-FourRooms-15x15-v0',
    entry_point=module_path + ':FourRoomsEnv15x15'
)

register(
    env_id='MultiGrid-FourRooms-Single-v0',
    entry_point=module_path + ':FourRoomsEnvSingle'
)

register(
    env_id='MultiGrid-TwoRooms-Minigrid-v0',
    entry_point=module_path + ':TwoRoomsEnvMinigrid'
)

register(
    env_id='MultiGrid-FourRooms-Minigrid-v0',
    entry_point=module_path + ':FourRoomsEnvMinigrid'
)

register(
    env_id='MultiGrid-MiniTwoRooms-Minigrid-v0',
    entry_point=module_path + ':MiniTwoRoomsEnvMinigrid'
)

register(
    env_id='MultiGrid-MiniFourRooms-Minigrid-v0',
    entry_point=module_path + ':MiniFourRoomsEnvMinigrid'
)

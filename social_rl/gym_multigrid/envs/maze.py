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

# Lint as: python3
"""Implements single-agent manually generated Maze environments.

Humans provide a bit map to describe the position of walls, the starting
location of the agent, and the goal location.
"""
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class MazeEnv(multigrid.MultiGridEnv):
  """Single-agent maze environment specified via a bit map."""

  def __init__(self, agent_view_size=5, minigrid_mode=True, max_steps=None,
               bit_map=None, start_pos=None, goal_pos=None, size=15, **kwargs):
    default_agent_start_x = 7
    default_agent_start_y = 1
    default_goal_start_x = 7
    default_goal_start_y = 13
    self.start_pos = np.array(
        [default_agent_start_x,
         default_agent_start_y]) if start_pos is None else start_pos
    self.goal_pos = (
        default_goal_start_x,
        default_goal_start_y) if goal_pos is None else goal_pos

    if max_steps is None:
      max_steps = 2*size*size

    if bit_map is not None:
      bit_map = np.array(bit_map)
      if bit_map.shape != (size-2, size-2):
        print('Error! Bit map shape does not match size. Using default maze.')
        bit_map = None

    if bit_map is None:
      self.bit_map = np.array([
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
          [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
          [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
          [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
          [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
      ])
    else:
      self.bit_map = bit_map

    super().__init__(
        n_agents=1,
        grid_size=size,
        agent_view_size=agent_view_size,
        max_steps=max_steps,
        see_through_walls=True,  # Set this to True for maximum speed
        minigrid_mode=minigrid_mode,
        **kwargs
    )

  def _gen_grid(self, width, height):
    # Create an empty grid
    self.grid = multigrid.Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    # Goal
    self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])

    # Agent
    self.place_agent_at_pos(0, self.start_pos)

    # Walls
    for x in range(self.bit_map.shape[0]):
      for y in range(self.bit_map.shape[1]):
        if self.bit_map[y, x]:
          # Add an offset of 1 for the outer walls
          self.put_obj(minigrid.Wall(), x+1, y+1)


class HorizontalMazeEnv(MazeEnv):
  """A short but non-optimal path is 80 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([1, 7])
    goal_pos = np.array([13, 5])
    bit_map = np.array([
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class Maze3Env(MazeEnv):
  """A short but non-optimal path is 80 moves."""

  def __init__(self, **kwargs):
    # positions go col, row and indexing starts at 1
    start_pos = np.array([4, 1])
    goal_pos = np.array([13, 7])
    bit_map = np.array([
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class LabyrinthEnv(MazeEnv):
  """A short but non-optimal path is 118 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([1, 13])
    goal_pos = np.array([7, 7])
    bit_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class Labyrinth2Env(MazeEnv):
  """A short but non-optimal path is 118 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([1, 1])
    goal_pos = np.array([7, 7])
    bit_map = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class NineRoomsEnv(MazeEnv):
  """Can be completed in 27 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([2, 2])
    goal_pos = np.array([12, 12])
    bit_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class NineRoomsFewerDoorsEnv(MazeEnv):
  """Can be completed in 27 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([2, 2])
    goal_pos = np.array([12, 12])
    bit_map = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class SixteenRoomsEnv(MazeEnv):
  """Can be completed in 16 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([2, 2])
    goal_pos = np.array([12, 12])
    bit_map = np.array([
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class SixteenRoomsFewerDoorsEnv(MazeEnv):
  """Can be completed in 16 moves."""

  def __init__(self, **kwargs):
    # positions go col, row
    start_pos = np.array([2, 2])
    goal_pos = np.array([12, 12])
    bit_map = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    ])
    super().__init__(size=15, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class MiniMazeEnv(MazeEnv):
  """A smaller maze for debugging."""

  def __init__(self, **kwargs):
    start_pos = np.array([1, 1])
    goal_pos = np.array([1, 3])
    bit_map = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    super().__init__(size=6, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)


class MediumMazeEnv(MazeEnv):
  """A 10x10 Maze environment."""

  def __init__(self, **kwargs):
    start_pos = np.array([5, 1])
    goal_pos = np.array([3, 8])
    bit_map = np.array([
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ])
    super().__init__(size=10, bit_map=bit_map, start_pos=start_pos,
                     goal_pos=goal_pos, **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-Maze-v0',
    entry_point=module_path + ':MazeEnv'
)

register(
    env_id='MultiGrid-MiniMaze-v0',
    entry_point=module_path + ':MiniMazeEnv'
)

register(
    env_id='MultiGrid-MediumMaze-v0',
    entry_point=module_path + ':MediumMazeEnv'
)

register(
    env_id='MultiGrid-Maze2-v0',
    entry_point=module_path + ':HorizontalMazeEnv'
)

register(
    env_id='MultiGrid-Maze3-v0',
    entry_point=module_path + ':Maze3Env'
)

register(
    env_id='MultiGrid-Labyrinth-v0',
    entry_point=module_path + ':LabyrinthEnv'
)

register(
    env_id='MultiGrid-Labyrinth2-v0',
    entry_point=module_path + ':Labyrinth2Env'
)

register(
    env_id='MultiGrid-SixteenRooms-v0',
    entry_point=module_path + ':SixteenRoomsEnv'
)

register(
    env_id='MultiGrid-SixteenRoomsFewerDoors-v0',
    entry_point=module_path + ':SixteenRoomsFewerDoorsEnv'
)

register(
    env_id='MultiGrid-NineRooms-v0',
    entry_point=module_path + ':NineRoomsEnv'
)

register(
    env_id='MultiGrid-NineRoomsFewerDoors-v0',
    entry_point=module_path + ':NineRoomsFewerDoorsEnv'
)

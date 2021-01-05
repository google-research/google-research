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
"""Multi-agent goal-seeking task with many static obstacles.
"""

import gym_minigrid.minigrid as minigrid
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class ClutteredMultiGrid(multigrid.MultiGridEnv):
  """Goal seeking environment with obstacles."""

  def __init__(self, size=15, n_agents=3, n_clutter=25, randomize_goal=True,
               agent_view_size=5, max_steps=250, walls_are_lava=False,
               **kwargs):
    self.n_clutter = n_clutter
    self.randomize_goal = randomize_goal
    self.walls_are_lava = walls_are_lava
    super().__init__(grid_size=size, max_steps=max_steps, n_agents=n_agents,
                     agent_view_size=agent_view_size, **kwargs)

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    if self.randomize_goal:
      self.place_obj(minigrid.Goal(), max_tries=100)
    else:
      self.put_obj(minigrid.Goal(), width - 2, height - 2)
    for _ in range(self.n_clutter):
      if self.walls_are_lava:
        self.place_obj(minigrid.Lava(), max_tries=100)
      else:
        self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'get to the green square'

  def step(self, action):
    obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
    return obs, reward, done, info


class ClutteredMultiGridSingle6x6(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, size=6, n_clutter=5, randomize_goal=True,
                     agent_view_size=5, max_steps=50)


class ClutteredMultiGridSingle(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, size=15, n_clutter=25, randomize_goal=True,
                     agent_view_size=5, max_steps=250)


class Cluttered40Minigrid(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=40, minigrid_mode=True)


class Cluttered10Minigrid(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=10, minigrid_mode=True)


class Cluttered50Minigrid(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=50, minigrid_mode=True)


class Cluttered5Minigrid(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=5, minigrid_mode=True)


class Cluttered1MinigridMini(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=1, minigrid_mode=True, size=6)


class Cluttered6MinigridMini(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=6, minigrid_mode=True, size=6)


class Cluttered7MinigridMini(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=7, minigrid_mode=True, size=6)


class ClutteredMinigridLava(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, walls_are_lava=True, minigrid_mode=True)


class ClutteredMinigridLavaMini(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=4, walls_are_lava=True, size=6,
                     minigrid_mode=True)


class ClutteredMinigridLavaMedium(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=15, walls_are_lava=True, size=10,
                     minigrid_mode=True)


class Cluttered15MinigridMedium(ClutteredMultiGrid):

  def __init__(self):
    super().__init__(n_agents=1, n_clutter=15, minigrid_mode=True, size=10)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-Cluttered-v0',
    entry_point=module_path + ':ClutteredMultiGrid'
)

register(
    env_id='MultiGrid-Cluttered-Single-v0',
    entry_point=module_path + ':ClutteredMultiGridSingle'
)

register(
    env_id='MultiGrid-Cluttered-Single-6x6-v0',
    entry_point=module_path + ':ClutteredMultiGridSingle6x6'
)

register(
    env_id='MultiGrid-Cluttered40-Minigrid-v0',
    entry_point=module_path + ':Cluttered40Minigrid'
)

register(
    env_id='MultiGrid-Cluttered10-Minigrid-v0',
    entry_point=module_path + ':Cluttered10Minigrid'
)

register(
    env_id='MultiGrid-Cluttered50-Minigrid-v0',
    entry_point=module_path + ':Cluttered50Minigrid'
)

register(
    env_id='MultiGrid-Cluttered5-Minigrid-v0',
    entry_point=module_path + ':Cluttered5Minigrid'
)

register(
    env_id='MultiGrid-MiniCluttered1-Minigrid-v0',
    entry_point=module_path + ':Cluttered1MinigridMini'
)

register(
    env_id='MultiGrid-MiniCluttered6-Minigrid-v0',
    entry_point=module_path + ':Cluttered6MinigridMini'
)

register(
    env_id='MultiGrid-MiniCluttered7-Minigrid-v0',
    entry_point=module_path + ':Cluttered7MinigridMini'
)

register(
    env_id='MultiGrid-Cluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLava'
)

register(
    env_id='MultiGrid-MiniCluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLavaMini'
)

register(
    env_id='MultiGrid-MediumCluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLavaMedium'
)

register(
    env_id='MultiGrid-MediumCluttered15-Minigrid-v0',
    entry_point=module_path + ':Cluttered15MinigridMedium'
)


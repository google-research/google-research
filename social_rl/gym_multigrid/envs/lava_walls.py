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

"""The walls are secretly lava, but they look to the agent like walls.
"""

import gym_minigrid.minigrid as minigrid
import gym_minigrid.rendering as rendering
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class LavaWall(minigrid.WorldObj):
  """Object which looks like a Wall to the agent but is deadly Lava."""

  def __init__(self):
    super().__init__('lava', 'grey')

  def can_overlap(self):
    return True

  def encode(self):
    """Even though it's lava, it's encoded to look like a wall."""
    return (minigrid.OBJECT_TO_IDX['wall'], minigrid.COLOR_TO_IDX['grey'], 0)

  def render(self, img):
    orange = (255, 128, 0)

    # Background color
    rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), orange)

    # Little waves
    for i in range(3):
      ylo = 0.3 + 0.2 * i
      yhi = 0.4 + 0.2 * i
      rendering.fill_coords(
          img, rendering.point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
      rendering.fill_coords(
          img, rendering.point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
      rendering.fill_coords(
          img, rendering.point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
      rendering.fill_coords(
          img, rendering.point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class WallsAreLavaMultiGrid(multigrid.MultiGridEnv):
  """Goal seeking environment with obstacles."""

  def __init__(self, size=15, n_agents=1, n_clutter=25, randomize_goal=True,
               agent_view_size=5, max_steps=250, walls_are_lava=False,
               minigrid_mode=True, competitive=True, **kwargs):
    self.n_clutter = n_clutter
    self.randomize_goal = randomize_goal
    super().__init__(n_agents=n_agents, minigrid_mode=minigrid_mode,
                     grid_size=size, max_steps=max_steps,
                     competitive=competitive,
                     agent_view_size=agent_view_size, **kwargs)

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)

    if self.randomize_goal:
      self.place_obj(minigrid.Goal(), max_tries=100)
    else:
      self.put_obj(minigrid.Goal(), width - 2, height - 2)
    for _ in range(self.n_clutter):
      self.place_obj(LavaWall(), max_tries=100)

    self.place_agent()

    self.mission = 'get to the green square'

  def step(self, action):
    obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
    return obs, reward, done, info


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-WallsAreLava-v0',
    entry_point=module_path + ':WallsAreLavaMultiGrid'
)

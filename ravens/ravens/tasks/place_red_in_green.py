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

"""Sorting Task."""

import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks.task import Task


class PlaceRedInGreen(Task):
  """Sorting Task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 10
    self.pos_eps = 0.05

  def reset(self, env):
    super().reset(env)
    n_bowls = np.random.randint(1, 4)
    n_blocks = np.random.randint(1, n_bowls + 1)

    # Add bowls.
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'assets/bowl/bowl.urdf'
    bowl_poses = []
    for _ in range(n_bowls):
      bowl_pose = self.get_random_pose(env, bowl_size)
      env.add_object(bowl_urdf, bowl_pose, 'fixed')
      bowl_poses.append(bowl_pose)

    # Add blocks.
    blocks = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'assets/stacking/block.urdf'
    for _ in range(n_blocks):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      blocks.append((block_id, (0, None)))

    # Goal: each red block is in a different green bowl.
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))

    # Colors of distractor objects.
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

    # Add distractors.
    n_distractors = 0
    while n_distractors < 10:
      is_block = np.random.rand() > 0.5
      urdf = block_urdf if is_block else bowl_urdf
      size = block_size if is_block else bowl_size
      colors = block_colors if is_block else bowl_colors
      pose = self.get_random_pose(env, size)
      if not pose[0] or not pose[1]:
        continue
      obj_id = env.add_object(urdf, pose)
      color = colors[n_distractors % len(colors)]
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
      n_distractors += 1

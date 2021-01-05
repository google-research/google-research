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

"""Stacking task."""

import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks.task import Task


class StackBlockPyramid(Task):
  """Stacking task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 12

  def reset(self, env):
    super().reset(env)

    # Add base.
    base_size = (0.05, 0.15, 0.005)
    base_urdf = 'assets/stacking/stand.urdf'
    base_pose = self.get_random_pose(env, base_size)
    env.add_object(base_urdf, base_pose, 'fixed')

    # Block colors.
    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['red']
    ]

    # Add blocks.
    objs = []
    # sym = np.pi / 2
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'assets/stacking/block.urdf'
    for i in range(6):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
      objs.append((block_id, (np.pi / 2, None)))

    # Associate placement locations for goals.
    place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                 (0, 0.05, 0.03), (0, -0.025, 0.08),
                 (0, 0.025, 0.08), (0, 0, 0.13)]
    targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

    # Goal: blocks are stacked in a pyramid (bottom row: green, blue, purple).
    self.goals.append((objs[:3], np.ones((3, 3)), targs[:3],
                       False, True, 'pose', None, 1 / 2))

    # Goal: blocks are stacked in a pyramid (middle row: yellow, orange).
    self.goals.append((objs[3:5], np.ones((2, 2)), targs[3:5],
                       False, True, 'pose', None, 1 / 3))

    # Goal: blocks are stacked in a pyramid (top row: red).
    self.goals.append((objs[5:], np.ones((1, 1)), targs[5:],
                       False, True, 'pose', None, 1 / 6))




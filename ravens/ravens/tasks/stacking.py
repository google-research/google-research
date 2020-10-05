# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
from ravens.tasks import Task


class Stacking(Task):
  """Stacking task."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.max_steps = 12
    self.metric = 'pose'
    self.primitive = 'pick_place'
    self.rotation_eps = np.deg2rad(45)

  def reset(self, env):

    # Add base.
    base_size = (0.05, 0.15, 0.005)
    base_urdf = 'assets/stacking/stand.urdf'
    base_pose = self.random_pose(env, base_size)
    env.add_object(base_urdf, base_pose, fixed=True)

    # Block colors.
    colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['red']
    ]

    # Add blocks.
    block_ids = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'assets/stacking/block.urdf'
    for i in range(6):
      block_pose = self.random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
      block_ids.append(block_id)

    # Associate placement locations.
    self.num_steps = 6
    self.goal = {'places': {}, 'steps': []}
    self.goal['places'] = {
        0: (utils.apply(base_pose, (0, -0.05, 0.03)), base_pose[1]),
        1: (utils.apply(base_pose, (0, 0, 0.03)), base_pose[1]),
        2: (utils.apply(base_pose, (0, 0.05, 0.03)), base_pose[1]),
        3: (utils.apply(base_pose, (0, -0.025, 0.08)), base_pose[1]),
        4: (utils.apply(base_pose, (0, 0.025, 0.08)), base_pose[1]),
        5: (utils.apply(base_pose, (0, 0, 0.13)), base_pose[1])
    }
    block_symmetry = np.pi / 2
    self.goal['steps'] = [{
        block_ids[0]: (block_symmetry, [0, 1, 2]),
        block_ids[1]: (block_symmetry, [0, 1, 2]),
        block_ids[2]: (block_symmetry, [0, 1, 2])
    }, {
        block_ids[3]: (block_symmetry, [3, 4]),
        block_ids[4]: (block_symmetry, [3, 4])
    }, {
        block_ids[5]: (block_symmetry, [5])
    }]

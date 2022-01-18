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

"""Towers of Hanoi task."""

import numpy as np

from ravens import utils
from ravens.tasks.task import Task


class TowersOfHanoi(Task):
  """Towers of Hanoi task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 14

  def reset(self, env):
    super().reset(env)

    # Add stand.
    base_size = (0.12, 0.36, 0.01)
    base_urdf = 'assets/hanoi/stand.urdf'
    base_pose = self.get_random_pose(env, base_size)
    env.add_object(base_urdf, base_pose, 'fixed')

    # Rod positions in base coordinates.
    rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))

    # Add disks.
    disks = []
    n_disks = 3
    for i in range(n_disks):
      disk_urdf = 'assets/hanoi/disk%d.urdf' % i
      pos = utils.apply(base_pose, rod_pos[0])
      z = 0.015 * (n_disks - i - 2)
      pos = (pos[0], pos[1], pos[2] + z)
      disks.append(env.add_object(disk_urdf, (pos, base_pose[1])))

    # Solve Hanoi sequence with dynamic programming.
    hanoi_steps = []  # [[object index, from rod, to rod], ...]

    def solve_hanoi(n, t0, t1, t2):
      if n == 0:
        hanoi_steps.append([n, t0, t1])
        return
      solve_hanoi(n - 1, t0, t2, t1)
      hanoi_steps.append([n, t0, t1])
      solve_hanoi(n - 1, t2, t1, t0)
    solve_hanoi(n_disks - 1, 0, 2, 1)

    # Goal: pick and place disks using Hanoi sequence.
    for step in hanoi_steps:
      disk_id = disks[step[0]]
      targ_pos = rod_pos[step[2]]
      targ_pos = utils.apply(base_pose, targ_pos)
      targ_pose = (targ_pos, (0, 0, 0, 1))
      self.goals.append(([(disk_id, (0, None))], np.int32([[1]]), [targ_pose],
                         False, True, 'pose', None, 1 / len(hanoi_steps)))

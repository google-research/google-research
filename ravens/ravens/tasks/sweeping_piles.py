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

"""Sweeping task."""

import numpy as np
from ravens import primitives
from ravens import utils
from ravens.grippers import Spatula
from ravens.tasks.task import Task


class SweepingPiles(Task):
  """Sweeping task."""

  def __init__(self):
    super().__init__()
    self.ee = Spatula
    self.max_steps = 20
    self.primitive = primitives.push

  def reset(self, env):
    super().reset(env)

    # Add goal zone.
    zone_size = (0.12, 0.12, 0)
    zone_pose = self.get_random_pose(env, zone_size)
    env.add_object('assets/zone/zone.urdf', zone_pose, 'fixed')

    # Add pile of small blocks.
    obj_pts = {}
    obj_ids = []
    for _ in range(50):
      rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
      ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
      xyz = (rx, ry, 0.01)
      theta = np.random.rand() * 2 * np.pi
      xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
      obj_id = env.add_object('assets/block/small.urdf', (xyz, xyzw))
      obj_pts[obj_id] = self.get_object_points(obj_id)
      obj_ids.append((obj_id, (0, None)))

    # Goal: all small blocks must be in zone.
    # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
    # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
    # self.goals.append((goal, metric))
    self.goals.append((obj_ids, np.ones((50, 1)), [zone_pose], True, False,
                       'zone', (obj_pts, [(zone_pose, zone_size)]), 1))

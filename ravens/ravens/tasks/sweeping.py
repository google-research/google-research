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

from ravens import utils
from ravens.tasks import Task


class Sweeping(Task):
  """Sweeping task."""

  def __init__(self):
    super().__init__()
    self.ee = 'spatula'
    self.max_steps = 20
    self.metric = 'zone'
    self.primitive = 'sweep'

  def reset(self, env):
    self.total_rewards = 0

    # Add zone.
    zone_urdf = 'assets/zone/zone.urdf'
    self.zone_size = (0.12, 0.12, 0)
    self.zone_pose = self.random_pose(env, self.zone_size)
    env.add_object(zone_urdf, self.zone_pose, fixed=True)

    # Add morsels.
    self.object_points = {}
    morsel_urdf = 'assets/morsel/morsel.urdf'
    for _ in range(50):
      rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
      ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
      position = (rx, ry, 0.01)
      theta = np.random.rand() * 2 * np.pi
      rotation = utils.get_pybullet_quaternion_from_rot((0, 0, theta))
      pose = (position, rotation)
      object_id = env.add_object(morsel_urdf, pose)
      self.object_points[object_id] = self.get_object_points(object_id)

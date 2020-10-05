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

"""Cable task."""

import os
import time
import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks import Task


class Cable(Task):
  """Cable task."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.max_steps = 20
    self.metric = 'zone'
    self.primitive = 'pick_place'

  def reset(self, env):
    self.total_rewards = 0
    self.goal = {'places': {}, 'steps': [{}]}

    num_parts = 20
    radius = 0.005
    length = 2 * radius * num_parts * np.sqrt(2)

    square_size = (length, length, 0)
    square_pose = self.random_pose(env, square_size)
    square_template = 'assets/square/square-template.urdf'
    replace = {'DIM': (length,), 'HALF': (length / 2 - 0.005,)}
    urdf = self.fill_template(square_template, replace)
    env.add_object(urdf, square_pose, fixed=True)
    os.remove(urdf)

    # Add goal line.
    # line_template = 'assets/line/line-template.urdf'
    self.zone_size = (length, 0.03, 0.2)
    zone_range = (self.zone_size[0], self.zone_size[1], 0.001)
    zone_position = (0, length / 2, 0.001)
    zone_position = utils.apply(square_pose, zone_position)
    self.zone_pose = (zone_position, square_pose[1])
    # urdf = self.fill_template(line_template, {'DIM': (length,)})
    # env.add_object(urdf, self.zone_pose, fixed=True)
    # os.remove(urdf)

    # Add beaded cable.
    distance = length / num_parts
    position, _ = self.random_pose(env, zone_range)
    position = np.float32(position)
    part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
    part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
    # part_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius] * 3)
    self.object_points = {}
    for i in range(num_parts):
      position[2] += distance
      part_id = p.createMultiBody(
          0.1, part_shape, part_visual, basePosition=position)
      if env.objects:
        constraint_id = p.createConstraint(
            parentBodyUniqueId=env.objects[-1],
            parentLinkIndex=-1,
            childBodyUniqueId=part_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, distance),
            childFramePosition=(0, 0, 0))
        p.changeConstraint(constraint_id, maxForce=100)
      if (i > 0) and (i < num_parts - 1):
        color = utils.COLORS['red'] + [1]
        p.changeVisualShape(part_id, -1, rgbaColor=color)
      env.objects.append(part_id)
      self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

      true_position = (radius + distance * i - length / 2, 0, 0)
      true_position = utils.apply(self.zone_pose, true_position)
      self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
      symmetry = 0  # zone-evaluation: symmetry does not matter
      self.goal['steps'][0][part_id] = (symmetry, [part_id])

    # Wait for beaded cable to settle.
    env.start()
    time.sleep(1)
    env.pause()

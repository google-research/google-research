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

"""Palletizing Task."""

import os
import numpy as np

import pybullet as p

from ravens import utils
from ravens.tasks import Task


class Palletizing(Task):
  """Palletizing Task."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.max_steps = 30
    self.metric = 'zone'
    self.primitive = 'pick_place'

  def reset(self, env):

    # Add pallet.
    self.zone_size = (0.3, 0.25, 0.25)
    zone_urdf = 'assets/pallet/pallet.urdf'
    rotation = utils.get_pybullet_quaternion_from_rot((0, 0, 0))
    self.zone_pose = ((0.5, 0.25, 0.02), rotation)
    env.add_object(zone_urdf, self.zone_pose, fixed=True)

    # Add stack of boxes on pallet.
    margin = 0.01
    self.object_points = {}
    stack_size = (0.19, 0.19, 0.19)
    box_template = 'assets/box/box-template.urdf'
    stack_dim = np.random.randint(low=2, high=4, size=3)  # (3, 3, 3)
    box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim
    for z in range(stack_dim[2]):

      # Transpose every layer.
      stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
      box_size[0], box_size[1] = box_size[1], box_size[0]

      for y in range(stack_dim[1]):
        for x in range(stack_dim[0]):
          position = (x + 0.5, y + 0.5, z + 0.5) * box_size
          position[0] += x * margin - stack_size[0] / 2
          position[1] += y * margin - stack_size[1] / 2
          position[2] += z * margin + 0.03
          pose = (position, (0, 0, 0, 1))
          pose = utils.multiply(self.zone_pose, pose)
          urdf = self.fill_template(box_template, {'DIM': box_size})
          box_id = env.add_object(urdf, pose)
          os.remove(urdf)
          self.color_random_brown(box_id)
          self.object_points[box_id] = self.get_object_points(box_id)

    # Randomly select top box on pallet and save ground truth pose.
    self.goal = {'places': {}, 'steps': []}
    boxes = env.objects.copy()
    while boxes:
      _, height, object_mask = self.get_object_masks(env)
      top = np.argwhere(height > (np.max(height) - 0.03))
      rpixel = top[int(np.floor(np.random.random() * len(top)))]  # y, x
      box_id = int(object_mask[rpixel[0], rpixel[1]])
      if box_id in boxes:
        position, rotation = p.getBasePositionAndOrientation(box_id)
        rposition = np.float32(position) + np.float32([0, -10, 0])
        p.resetBasePositionAndOrientation(box_id, rposition, rotation)
        self.goal['places'][box_id] = (position, rotation)
        symmetry = 0  # zone-evaluation: symmetry does not matter
        self.goal['steps'].append({box_id: (symmetry, [box_id])})
        boxes.remove(box_id)
    self.goal['steps'].reverse()  # time-reverse depalletizing
    self.total_rewards = 0
    self.max_steps = len(self.goal['steps']) * 2

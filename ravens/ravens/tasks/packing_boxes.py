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

"""Packing task."""

import os
import numpy as np

import pybullet as p

from ravens import utils
from ravens.tasks.task import Task


class PackingBoxes(Task):
  """Packing task."""

  def __init__(self):
    super().__init__()
    self.max_steps = 20

  def reset(self, env):
    super().reset(env)

    # Add container box.
    zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
    zone_pose = self.get_random_pose(env, zone_size)
    container_template = 'assets/container/container-template.urdf'
    half = np.float32(zone_size) / 2
    replace = {'DIM': zone_size, 'HALF': half}
    container_urdf = self.fill_template(container_template, replace)
    env.add_object(container_urdf, zone_pose, 'fixed')
    os.remove(container_urdf)

    margin = 0.01
    min_object_dim = 0.05
    bboxes = []

    class TreeNode:

      def __init__(self, parent, children, bbox):
        self.parent = parent
        self.children = children
        self.bbox = bbox  # min x, min y, min z, max x, max y, max z

    def KDTree(node):
      size = node.bbox[3:] - node.bbox[:3]

      # Choose which axis to split.
      split = size > 2 * min_object_dim
      if np.sum(split) == 0:
        bboxes.append(node.bbox)
        return
      split = np.float32(split) / np.sum(split)
      split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

      # Split along chosen axis and create 2 children
      cut_ind = np.random.rand() * \
          (size[split_axis] - 2 * min_object_dim) + \
          node.bbox[split_axis] + min_object_dim
      child1_bbox = node.bbox.copy()
      child1_bbox[3 + split_axis] = cut_ind - margin / 2.
      child2_bbox = node.bbox.copy()
      child2_bbox[split_axis] = cut_ind + margin / 2.
      node.children = [
          TreeNode(node, [], bbox=child1_bbox),
          TreeNode(node, [], bbox=child2_bbox)
      ]
      KDTree(node.children[0])
      KDTree(node.children[1])

    # Split container space with KD trees.
    stack_size = np.array(zone_size)
    stack_size[0] -= 0.01
    stack_size[1] -= 0.01
    root_size = (0.01, 0.01, 0) + tuple(stack_size)
    root = TreeNode(None, [], bbox=np.array(root_size))
    KDTree(root)

    colors = [utils.COLORS[c] for c in utils.COLORS if c != 'brown']

    # Add objects in container.
    object_points = {}
    object_ids = []
    bboxes = np.array(bboxes)
    object_template = 'assets/box/box-template.urdf'
    for bbox in bboxes:
      size = bbox[3:] - bbox[:3]
      position = size / 2. + bbox[:3]
      position[0] += -zone_size[0] / 2
      position[1] += -zone_size[1] / 2
      pose = (position, (0, 0, 0, 1))
      pose = utils.multiply(zone_pose, pose)
      urdf = self.fill_template(object_template, {'DIM': size})
      box_id = env.add_object(urdf, pose)
      os.remove(urdf)
      object_ids.append((box_id, (0, None)))
      icolor = np.random.choice(range(len(colors)), 1).squeeze()
      p.changeVisualShape(box_id, -1, rgbaColor=colors[icolor] + [1])
      object_points[box_id] = self.get_object_points(box_id)

    # Randomly select object in box and save ground truth pose.
    object_volumes = []
    true_poses = []
    # self.goal = {'places': {}, 'steps': []}
    for object_id, _ in object_ids:
      true_pose = p.getBasePositionAndOrientation(object_id)
      object_size = p.getVisualShapeData(object_id)[0][3]
      object_volumes.append(np.prod(np.array(object_size) * 100))
      pose = self.get_random_pose(env, object_size)
      p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
      true_poses.append(true_pose)
      # self.goal['places'][object_id] = true_pose
      # symmetry = 0  # zone-evaluation: symmetry does not matter
      # self.goal['steps'].append({object_id: (symmetry, [object_id])})
    # self.total_rewards = 0
    # self.max_steps = len(self.goal['steps']) * 2

    # Sort oracle picking order by object size.
    # self.goal['steps'] = [
    #     self.goal['steps'][i] for i in
    #.    np.argsort(-1 * np.array(object_volumes))
    # ]

    self.goals.append((
        object_ids, np.eye(len(object_ids)), true_poses, False, True, 'zone',
        (object_points, [(zone_pose, zone_size)]), 1))

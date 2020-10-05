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

"""Insertion Tasks."""

import cv2

import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks import Task


class Insertion(Task):
  """Insertion Task base class."""

  def __init__(self):
    super().__init__()
    self.ee = 'suction'
    self.max_steps = 3
    self.metric = 'pose'
    self.primitive = 'pick_place'

  def reset(self, env):
    self.num_steps = 1
    self.goal = {'places': {}, 'steps': []}

    # Add L-shaped block.
    block_size = (0.1, 0.1, 0.04)
    block_urdf = 'assets/insertion/ell.urdf'
    block_pose = self.random_pose(env, block_size)
    # block_pose = ((0.40, -0.15, 0.02),
    #               p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    block_id = env.add_object(block_urdf, block_pose)
    self.goal['steps'].append({block_id: (2 * np.pi, [0])})

    # Add L-shaped hole.
    hole_urdf = 'assets/insertion/hole.urdf'
    hole_pose = self.random_pose(env, block_size)
    # hole_pose = ((0.65, 0.10, 0.02),
    #              p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    env.add_object(hole_urdf, hole_pose, fixed=True)
    self.goal['places'][0] = hole_pose


class InsertionTranslation(Insertion):
  """Insertion Task - Translation Variant."""

  def random_pose(self, env, object_size):
    position, rotation = super(InsertionTranslation,
                               self).random_pose(env, object_size)
    rotation = utils.get_pybullet_quaternion_from_rot((0, 0, 0))
    return position, rotation


class InsertionEasy(Insertion):
  """Insertion Task - Easy Variant."""

  def reset(self, env):
    self.num_steps = 1
    self.goal = {'places': {}, 'steps': []}

    # Add L-shaped block.
    block_size = (0.1, 0.1, 0.04)
    block_urdf = 'assets/insertion/ell.urdf'
    # block_pose = self.random_pose(env, block_size)
    block_pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    block_id = env.add_object(block_urdf, block_pose)
    self.goal['steps'].append({block_id: (2 * np.pi, [0])})

    # Add L-shaped hole.
    hole_urdf = 'assets/insertion/hole.urdf'
    hole_pose = self.random_pose(env, block_size)
    # hole_pose = ((0.65, 0.10, 0.02),
    #              p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    # hole_pose[1] = utils.get_pybullet_quaternion_from_rot((0, 0, 0))
    env.add_object(hole_urdf, hole_pose, fixed=True)
    self.goal['places'][0] = hole_pose

  def random_pose(self, env, object_size):
    position, rotation = super(InsertionEasy,
                               self).random_pose(env, object_size)
    rotation = utils.get_pybullet_quaternion_from_rot((0, 0, np.pi / 2))
    return position, rotation


class InsertionSixDof(Insertion):
  """Insertion Task - 6 DOF Variant."""

  def __init__(self):
    super().__init__()

    # the block can wiggle a little in the slot
    self.position_eps = 0.02

    self.ee = 'suction'
    self.max_steps = 3
    self.metric = 'pose'
    self.primitive = 'pick_place_6dof'

  def reset(self, env):
    self.num_steps = 1
    self.goal = {'places': {}, 'steps': []}

    # Add L-shaped block.
    block_size = (0.1, 0.1, 0.04)
    block_urdf = 'assets/insertion/ell.urdf'
    block_pose = self.random_pose_3dof(env, block_size)
    block_id = env.add_object(block_urdf, block_pose)
    self.goal['steps'].append({block_id: (2 * np.pi, [0])})

    # Add L-shaped hole with bottom.
    hole_urdf = 'assets/insertion/hole_with_bottom.urdf'
    hole_pose = self.random_pose_6dof(env, block_size)
    env.add_object(hole_urdf, hole_pose, fixed=True)
    self.goal['places'][0] = hole_pose

  def random_pose_3dof(self, env, object_size):
    position, rotation = super(InsertionSixDof,
                               self).random_pose(env, object_size)
    return position, rotation

  def random_pose_6dof(self, env, object_size):
    """Get random collision-free pose in workspace bounds for object."""
    plane_id = 1
    max_size = np.linalg.norm(object_size[0:2])
    erode_size = int(np.round(max_size / self.pixel_size))
    _, heightmap, object_mask = self.get_object_masks(env)

    # Sample freespace regions in workspace.
    mask = np.uint8(object_mask == plane_id)
    mask[0, :], mask[:, 0], mask[-1, :], mask[:, -1] = 0, 0, 0, 0
    mask = cv2.erode(mask, np.ones((erode_size, erode_size), np.uint8))
    if np.sum(mask) == 0:
      return
    pixel = utils.sample_distribution(np.float32(mask))
    position = utils.pixel_to_position(pixel, heightmap, self.bounds,
                                       self.pixel_size)

    z_above_table = (np.random.rand(1)[0] / 10) + 0.03

    position = (position[0], position[1], object_size[2] / 2 + z_above_table)

    roll = (np.random.rand() - 0.5) * 0.5 * np.pi
    pitch = (np.random.rand() - 0.5) * 0.5 * np.pi
    yaw = np.random.rand() * 2 * np.pi
    rotation = utils.get_pybullet_quaternion_from_rot((roll, pitch, yaw))

    print(position, rotation)

    return position, rotation


class InsertionGoal(Insertion):
  """Insertion without a receptable, thus need goal images."""

  def reset(self, env, last_info=None):
    self.num_steps = 1
    self.goal = {'places': {}, 'steps': []}

    # Add L-shaped block.
    block_size = (0.1, 0.1, 0.04)
    block_urdf = 'assets/insertion/ell.urdf'
    block_pose = self.random_pose(env, block_size)
    block_id = env.add_object(block_urdf, block_pose)
    self.goal['steps'].append({block_id: (2 * np.pi, [0])})

    # Add L-shaped target pose, but without actually adding it.
    if self.goal_cond_testing:
      assert last_info is not None
      self.goal['places'][0] = self._get_goal_info(last_info)
      # print('\nin insertion reset, goal: {}'.format(self.goal['places'][0]))
    else:
      hole_pose = self.random_pose(env, block_size)
      self.goal['places'][0] = hole_pose
      # print('\nin insertion reset, goal: {}'.format(hole_pose))

  def _get_goal_info(self, last_info):
    """Used to determine the goal given the last `info` dict."""
    position, rotation, _ = last_info[4]  # block ID=4
    return (position, rotation)

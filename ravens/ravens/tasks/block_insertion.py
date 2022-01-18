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

#!/usr/bin/env python
"""Insertion Tasks."""

import numpy as np
import pybullet as p
from ravens import utils
from ravens.tasks.task import Task


class BlockInsertion(Task):
  """Insertion Task - Base Variant."""

  def __init__(self):
    super().__init__()
    self.max_steps = 3

  def reset(self, env):
    super().reset(env)
    block_id = self.add_block(env)
    targ_pose = self.add_fixture(env)
    # self.goals.append(
    #     ([block_id], [2 * np.pi], [[0]], [targ_pose], 'pose', None, 1.))
    self.goals.append(([(block_id, (2 * np.pi, None))], np.int32([[1]]),
                       [targ_pose], False, True, 'pose', None, 1))

  def add_block(self, env):
    """Add L-shaped block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'assets/insertion/ell.urdf'
    pose = self.get_random_pose(env, size)
    return env.add_object(urdf, pose)

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'assets/insertion/fixture.urdf'
    pose = self.get_random_pose(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose


class BlockInsertionTranslation(BlockInsertion):
  """Insertion Task - Translation Variant."""

  def get_random_pose(self, env, obj_size):
    pose = super(BlockInsertionTranslation, self).get_random_pose(env, obj_size)
    pos, rot = pose
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
    return pos, rot

  # Visualization positions.
  # block_pos = (0.40, -0.15, 0.02)
  # fixture_pos = (0.65, 0.10, 0.02)


class BlockInsertionEasy(BlockInsertionTranslation):
  """Insertion Task - Easy Variant."""

  def add_block(self, env):
    """Add L-shaped block in fixed position."""
    # size = (0.1, 0.1, 0.04)
    urdf = 'assets/insertion/ell.urdf'
    pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
    return env.add_object(urdf, pose)


class BlockInsertionSixDof(BlockInsertion):
  """Insertion Task - 6DOF Variant."""

  def __init__(self):
    super().__init__()
    self.sixdof = True
    self.pos_eps = 0.02

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'assets/insertion/fixture.urdf'
    pose = self.get_random_pose_6dof(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose

  def get_random_pose_6dof(self, env, obj_size):
    pos, rot = super(BlockInsertionSixDof, self).get_random_pose(env, obj_size)
    z = (np.random.rand() / 10) + 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z)
    roll = (np.random.rand() - 0.5) * np.pi / 2
    pitch = (np.random.rand() - 0.5) * np.pi / 2
    yaw = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
    return pos, rot


class BlockInsertionNoFixture(BlockInsertion):
  """Insertion Task - No Fixture Variant."""

  def add_fixture(self, env):
    """Add target pose to place block."""
    size = (0.1, 0.1, 0.04)
    # urdf = 'assets/insertion/fixture.urdf'
    pose = self.get_random_pose(env, size)
    return pose

  # def reset(self, env, last_info=None):
  #   self.num_steps = 1
  #   self.goal = {'places': {}, 'steps': []}

  #   # Add L-shaped block.
  #   block_size = (0.1, 0.1, 0.04)
  #   block_urdf = 'assets/insertion/ell.urdf'
  #   block_pose = self.get_random_pose(env, block_size)
  #   block_id = env.add_object(block_urdf, block_pose)
  #   self.goal['steps'].append({block_id: (2 * np.pi, [0])})

  #   # Add L-shaped target pose, but without actually adding it.
  #   if self.goal_cond_testing:
  #     assert last_info is not None
  #     self.goal['places'][0] = self._get_goal_info(last_info)
  #     # print('\nin insertion reset, goal: {}'.format(self.goal['places'][0]))
  #   else:
  #     hole_pose = self.get_random_pose(env, block_size)
  #     self.goal['places'][0] = hole_pose
  #     # print('\nin insertion reset, goal: {}'.format(hole_pose))

  # def _get_goal_info(self, last_info):
  #   """Used to determine the goal given the last `info` dict."""
  #   position, rotation, _ = last_info[4]  # block ID=4
  #   return (position, rotation)

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

"""Environment wrapper around the maze navigation environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from . import simple_maze
import cv2
import numpy as np


class Environment(object):
  """Wrapper around the Simple maze environment."""

  def __init__(self, difficulty=None):
    """Initialize the environment with the specified difficulty."""
    self.difficulty = difficulty
    self._sim_env = simple_maze.navigate(difficulty=difficulty)
    self.stepcount = 0

  def reset(self):
    """Resets the environment."""
    self.stepcount = 0
    time_step = self._sim_env.reset()
    return time_step

  def get_goal_im(self):
    """Computes and returns the goal image."""
    currp = copy.deepcopy(self._sim_env.physics.data.qpos[:])
    currv = copy.deepcopy(self._sim_env.physics.data.qvel[:])
    self._sim_env.task.dontreset = True
    tg = copy.deepcopy(self._sim_env.physics.named.data.geom_xpos['target'][:2])
    self._sim_env.physics.data.qpos[:] = tg
    self._sim_env.physics.data.qvel[:] = 0
    self.step([0, 0])
    self._sim_env.physics.data.qpos[:] = tg
    self._sim_env.physics.data.qvel[:] = 0
    self.step([0, 0])
    _, gim = self.get_observation()
    self._sim_env.physics.data.qpos[:] = currp
    self._sim_env.physics.data.qvel[:] = currv
    self.step([0, 0])
    self._sim_env.task.dontreset = False
    return gim

  def get_subgoal_ims(self, numg):
    """Computes and returs the ground truth sub goal images."""
    currp = copy.deepcopy(self._sim_env.physics.data.qpos[:])
    currv = copy.deepcopy(self._sim_env.physics.data.qvel[:])
    self._sim_env.task.dontreset = True
    tg = copy.deepcopy(self._sim_env.physics.named.data.geom_xpos['target'][:2])

    sg = []
    if self.difficulty == 'e':
      if numg == 1:
        self._sim_env.physics.data.qpos[:] = currp + (tg - currp) / 2
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
      elif numg == 2:
        self._sim_env.physics.data.qpos[:] = currp + (tg - currp) / 3
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
        self._sim_env.physics.data.qpos[:] = currp + 2 * (tg - currp) / 3
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
    elif self.difficulty == 'm':
      if numg == 1:
        self._sim_env.physics.data.qpos[:] = [
            self._sim_env.physics.named.model.geom_pos['wall2A', 'x'],
            self._sim_env.physics.named.model.geom_pos['wall2A', 'y'] - 0.25]
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
      elif numg == 2:
        self._sim_env.physics.data.qpos[:] = [
            self._sim_env.physics.named.model.geom_pos['wall2A', 'x'],
            self._sim_env.physics.named.model.geom_pos['wall2A', 'y'] - 0.25]
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
        self._sim_env.physics.data.qpos[:] = [
            self._sim_env.physics.named.model.geom_pos['wall2A', 'x'],
            self._sim_env.physics.named.model.geom_pos['wall2A', 'y'] - 0.25]
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
    elif self.difficulty == 'h':
      if numg == 1:
        self._sim_env.physics.data.qpos[:] = [
            self._sim_env.physics.named.model.geom_pos['wall1A', 'x'],
            self._sim_env.physics.named.model.geom_pos['wall1A', 'y'] - 0.25]
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
      elif numg == 2:
        self._sim_env.physics.data.qpos[:] = [
            self._sim_env.physics.named.model.geom_pos['wall1A', 'x'],
            self._sim_env.physics.named.model.geom_pos['wall1A', 'y'] - 0.25]
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
        self._sim_env.physics.data.qpos[:] = [
            self._sim_env.physics.named.model.geom_pos['wall2A', 'x'],
            self._sim_env.physics.named.model.geom_pos['wall2A', 'y'] - 0.25]
        self._sim_env.physics.data.qvel[:] = 0
        self.step([0, 0])
        _, gim = self.get_observation()
        sg.append(gim)
    sg = np.array(sg)

    self._sim_env.physics.data.qpos[:] = currp
    self._sim_env.physics.data.qvel[:] = currv
    self.step([0, 0])
    self._sim_env.task.dontreset = False
    return sg

  def is_goal(self):
    """Checks if the current state is a goal state."""
    return self._sim_env.task.is_goal(self._sim_env.physics)

  def step(self, action=None):
    """Steps the environment."""
    time_step = self._sim_env.step(action)
    self._sim_env.physics.data.qvel[:] = 0
    return time_step

  def get_observation(self):
    """Return image observation."""
    obs = self._sim_env.task.get_observation(self._sim_env.physics)
    im = self._sim_env.physics.render(256, 256, camera_id='fixed')
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    return obs, im


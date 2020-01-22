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

"""Simple Maze Navigation Environment.

Agent (green cube) must navigate through narrow gaps in walls
to reach goal position
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 50


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  dirname = os.path.dirname(__file__)
  filename = os.path.join(dirname, 'simple_maze.xml')
  with open(filename, 'r') as f:
    data = f.read().replace('\n', '')
  return data, common.ASSETS


def navigate(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None, difficulty=None):
  """Returns instance of the maze navigation task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = SimpleMaze(random=random, difficulty=difficulty)
  environment_kwargs = environment_kwargs or {}

  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=0.5,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation."""

  def tool_to_target(self):
    """Returns the vector from target to finger in global coordinates."""
    return (self.named.data.geom_xpos['target', :2] -
            self.named.data.geom_xpos['toolgeom', :2])

  def tool_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.tool_to_target())


class SimpleMaze(base.Task):
  """A Maze Navigation `Task` to reach the target."""

  def __init__(self, random=None, difficulty=None):
    """Initialize an instance of `MazeNavigation`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      difficulty: Optional, a String of 'e', 'm', 'h', for
      easy, medium or hard difficulty
    """
    super(SimpleMaze, self).__init__(random=random)
    self.difficulty = difficulty
    self.dontreset = False

  def initialize_episode(self, physics, difficulty=None):
    """Sets the state of the environment at the start of each episode."""
    # Sometime don't reset
    if self.dontreset:
      return

    # Reset based on difficulty
    if self.difficulty is None:
      randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    elif self.difficulty == 'e':
      physics.data.qpos[0] = self.random.uniform(0.15, 0.27)
    elif self.difficulty == 'm':
      physics.data.qpos[0] = self.random.uniform(-0.15, 0.15)
    elif self.difficulty == 'h':
      physics.data.qpos[0] = self.random.uniform(-0.27, -0.15)
    physics.data.qpos[1] = self.random.uniform(-0.27, 0.27)

    # Randomize wal positions
    w1 = self.random.uniform(-0.2, 0.2)
    w2 = self.random.uniform(-0.2, 0.2)
    physics.named.model.geom_pos['wall1A', 'y'] = 0.25 + w1
    physics.named.model.geom_pos['wall1B', 'y'] = -0.25 + w1
    physics.named.model.geom_pos['wall2A', 'y'] = 0.25 + w2
    physics.named.model.geom_pos['wall2B', 'y'] = -0.25 + w2

    # Randomize target position
    physics.named.model.geom_pos['target', 'x'] = self.random.uniform(0.2,
                                                                      0.28)
    physics.named.model.geom_pos['target', 'y'] = self.random.uniform(-0.28,
                                                                      0.28)

  def get_observation(self, physics):
    """Returns an observation of the state and positions."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.tool_to_target()
    obs['velocity'] = physics.velocity()
    return obs

  def is_goal(self, physics):
    """Checks if goal has been reached (within 5 cm)."""
    d = physics.tool_to_target_dist()
    if d < 0.05:
      return True
    return False

  def get_reward(self, physics):
    """Returns shaped reward (not used)."""
    d = physics.tool_to_target_dist()
    return -d




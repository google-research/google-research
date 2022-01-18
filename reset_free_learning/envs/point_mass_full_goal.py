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

"""Point Mass environment with the goal observation being the same as the state space."""
import math
import os

from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np


# pylint: disable=missing-docstring
class PointMassEnv(MujocoEnv):

  MODEL_PATH_DEFAULT = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), 'assets', 'point_mass.xml')
  MODEL_PATH_T = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), 'assets',
      'point_mass_t_shaped.xml')
  MODEL_PATH_Y = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), 'assets',
      'point_mass_y_shaped.xml')

  def __init__(self, env_type='default', reward_type='dense'):
    self.goal = np.array([1.0, 1.0, 0.0, 0.0, 0.0,
                          0.0])  # goal should be 6 dimensional
    self.custom_reset_flag = False
    self._use_simulator = True
    self._env_type = env_type
    self._reward_type = reward_type
    if env_type == 'default':
      model_path = self.MODEL_PATH_DEFAULT
    elif env_type == 'y':
      model_path = self.MODEL_PATH_Y
    elif env_type == 't':
      model_path = self.MODEL_PATH_T
    elif env_type == 'skewed_square':
      model_path = self.MODEL_PATH_DEFAULT
    MujocoEnv.__init__(self, model_path=model_path, frame_skip=5)

  def step(self, action):
    if self._use_simulator:
      self.do_simulation(action, self.frame_skip)
    else:
      force = 0.02 * action[0]
      rot = 1.0 * action[1]
      qpos = np.copy(self.sim.data.qpos)
      qpos[2] += rot
      ori = qpos[2]
      dx = math.cos(ori) * force
      dy = math.sin(ori) * force
      qpos[0] = np.clip(qpos[0] + dx, -3.5, 3.5)
      qpos[1] = np.clip(qpos[1] + dy, -3.5, 3.5)
      qvel = np.copy(self.sim.data.qvel)
      self.set_state(qpos, qvel)

    ob = self._get_obs()
    dist = np.linalg.norm(self.sim.data.qpos.flat[:2] - self.goal[:2])
    if self._reward_type == 'dense':
      reward = -10.0 * dist
      reward += 10.0 * np.exp(
          -(dist**2) / 0.01)  # bonus for being near the goal
    elif self._reward_type == 'sparse':
      reward = 10.0 * (dist <= 1.0)
    done = False
    return ob, reward, done, None

  def _get_obs(self):
    new_obs = [self.sim.data.qpos.flat, self.sim.data.qvel.flat, self.goal]
    return np.concatenate(new_obs).astype('float32')

  def get_next_goal(self):
    if self._env_type == 'default':
      extreme_pos = 7.0
      goal_positions = np.array([
          [extreme_pos, extreme_pos],
          [extreme_pos, -extreme_pos],
          [-extreme_pos, extreme_pos],
          [-extreme_pos, -extreme_pos],
          # [0.0, 0.0],
      ])
    elif self._env_type == 'y':
      goal_positions = np.array([
          [6.0, -4.0],
          [-6.0, -4.0],
          # [0.0, 8.0],
      ])
    elif self._env_type == 't':
      extreme_pos = 8.0
      goal_positions = np.array([
          [extreme_pos, 0.0],
          [-extreme_pos, 0.0],
          [0.0, extreme_pos],
      ])
    elif self._env_type == 'skewed_square':
      extreme_pos = 7.0
      goal_positions = np.array([
          [extreme_pos, extreme_pos],
          [-extreme_pos, extreme_pos],
      ])
    goal_idx = np.random.randint(len(goal_positions))
    goal = np.concatenate([goal_positions[goal_idx], np.zeros(4)])

    return goal

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()

    self.goal = goal
    bp = self.model.body_pos.copy()

    # reset goal
    target_id = self.model.body_name2id('target')
    bp[target_id, :2] = self.goal[:2]
    self.model.body_pos[:] = bp

  def reset_model(self):
    if not self.custom_reset_flag:
      if self._env_type == 'default':
        qpos = self.init_qpos + np.append(
            self.np_random.uniform(low=-.01, high=.01, size=2),
            self.np_random.uniform(-np.pi, np.pi, size=1))
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
      elif self._env_type == 't':
        qpos = self.init_qpos + np.append(
            self.np_random.uniform(low=-.01, high=.01, size=2),
            self.np_random.uniform(-np.pi, np.pi, size=1))
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
      elif self._env_type == 'y':
        qpos = self.init_qpos + np.append(
            self.np_random.uniform(low=-.01, high=.01, size=2) +
            np.array([0.0, 8.0]), self.np_random.uniform(-np.pi, np.pi, size=1))
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
      elif self._env_type == 'skewed_square':
        qpos = self.init_qpos + np.append(
            self.np_random.uniform(low=-.01, high=.01, size=2) + np.array(
                [0.0, -8.0]), self.np_random.uniform(-np.pi, np.pi, size=1))
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01

      self.reset_goal()
      self.set_state(qpos, qvel)
      return self._get_obs()
    else:
      self.custom_reset_flag = False
      return self.custom_reset()

  def do_custom_reset(self, pos):
    self.custom_reset_flag = True
    self.custom_reset_pos = pos

  def custom_reset(self, pos=None):
    if pos is None:
      pos = self.custom_reset_pos
    qpos = np.append(pos[:2], 0) + np.append(
        self.np_random.uniform(low=-.01, high=.01, size=2),
        self.np_random.uniform(-np.pi, np.pi, size=1))
    qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
    self.set_state(qpos, qvel)
    self.reset_goal()
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.trackbodyid = 0
    self.viewer.cam.lookat[:3] = [0, 0, 0]
    self.viewer.cam.distance = 30
    self.viewer.cam.elevation = -90
    self.viewer.cam.azimuth = 0
    self.viewer.cam.trackbodyid = -1

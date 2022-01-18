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

"""Simplified pusher environment."""
import os

from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np


class PusherEnv(MujocoEnv):

  FILE = 'pusher2d_simple.xml'
  MODEL_PATH = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), 'assets', FILE)

  def __init__(self):
    self.goal_pos = np.array([2.0, 0.])
    MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5)
    self.model.stat.extent = 10

  def reset_goal(self, goal=None):
    if goal is None:
      extreme_pos = 1.2
      goal_positions = np.array([
          [extreme_pos, extreme_pos],
          # [extreme_pos, -extreme_pos],
          # [-extreme_pos, extreme_pos],
          # [-extreme_pos, -extreme_pos]
      ])
      goal_idx = np.random.randint(len(goal_positions))
      self.goal_pos = goal_positions[goal_idx]
    else:
      self.goal_pos = goal

    bp = self.model.body_pos.copy()
    # reset goal
    target_id = self.model.body_name2id('target')
    bp[target_id, :2] = self.goal_pos
    self.model.body_pos[:] = bp

  def reset(self):
    qpos = np.zeros((5,))
    qvel = np.zeros((5,))
    qpos[0] = -0.25
    qpos[1] = 0.0
    self.set_state(np.array(qpos), np.array(qvel))

    # reset goal
    self.reset_goal()

    self.sim.forward()
    obs = self._get_obs()
    return obs

  def _get_obs(self):
    return np.concatenate([
        self.sim.data.qpos.flat[:3], self.sim.data.geom_xpos[-2:-1, :2].flat,
        self.sim.data.qvel.flat, self.goal_pos
    ]).reshape(-1)

  def step(self, action):
    self.do_simulation(action, self.frame_skip)
    next_obs = self._get_obs()

    curr_block_xidx = 3
    curr_block_yidx = 4
    curr_gripper_pos = self.sim.data.site_xpos[0, :2]
    curr_block_pos = np.array(
        [next_obs[curr_block_xidx], next_obs[curr_block_yidx]])
    dist_to_block = np.linalg.norm(curr_gripper_pos - curr_block_pos)
    block_dist = np.linalg.norm(self.goal_pos - curr_block_pos)

    reward = -dist_to_block - 10 * block_dist
    reward += 10 * np.exp(-(block_dist**2) / 0.01)

    done = False
    return next_obs, reward, done, {}

  def viewer_setup(self):
    self.viewer.cam.trackbodyid = 0
    self.viewer.cam.lookat[:3] = [0, 0, 0]
    self.viewer.cam.distance = 5.5
    self.viewer.cam.elevation = -90
    self.viewer.cam.azimuth = 0
    self.viewer.cam.trackbodyid = -1


if __name__ == '__main__':
  env = PusherEnv()
  env.reset()
  for _ in range(1000):
    env.step(np.random.uniform(-1, 1, size=(3,)))
  env.reset()
  for _ in range(1000):
    env.step(np.random.uniform(-1, 1, size=(3,)))

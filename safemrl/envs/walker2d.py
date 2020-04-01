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

"""Walker2D mujoco env with target velocity.

Implements Walker2D mujoco environment with variable target velocity
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import gin.tf
from gym.envs import registration
from gym.envs.mujoco.walker2d import Walker2dEnv
import numpy as np
import tensorflow.compat.v1 as tf


tf.compat.v1.enable_v2_behavior()


@gin.configurable
class Walker2dMrlEnv(Walker2dEnv):
  """Walker2D environment with goal velocity."""

  def __init__(self,
               goal_min,
               goal_max,
               goal_vel,
               *args,
               **kwargs):
    self._goal_vel_min = goal_min
    self._goal_vel_max = goal_max
    self._goal_vel = goal_vel
    super(Walker2dMrlEnv, self).__init__(*args, **kwargs)

  def reset(self):
    if self._goal_vel_min is not None and self._goal_vel_max is not None:
      self._goal_vel = np.random.uniform(self._goal_vel_min, self._goal_vel_max)
    return super(Walker2dMrlEnv, self).reset()

  def step(self, a):
    posbefore = self.physics.data.qpos[0]
    self.do_simulation(a, self.frame_skip)
    posafter, height, ang = self.physics.data.qpos[0:3]
    alive_bonus = 0.
    if self._goal_vel is not None:
      reward_motion = -np.abs((
          (posafter - posbefore) / self.dt) - self._goal_vel)
    else:
      reward_motion = ((posafter - posbefore) / self.dt)
    reward_ctrl = -1e-3 * np.square(a).sum()
    if not height > 0.8:
      reward_fall = -5
      fallen = True
    else:
      reward_fall = 0
      fallen = False
    reward = reward_motion + reward_ctrl + alive_bonus + reward_fall
    done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
    ob = self._get_obs()
    return ob, reward, done, dict(
        reward_motion=reward_motion,
        reward_ctrl=reward_ctrl,
        reward_alive=alive_bonus,
        reward_fall=reward_fall,
        fallen=fallen)


registration.register(
    id='DM-Walker2dMrl-v1',
    entry_point=Walker2dMrlEnv,
    max_episode_steps=1000,
)

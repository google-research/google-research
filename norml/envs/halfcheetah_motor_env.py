# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""The Halfcheetah environment used in the NoRML paper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gym.envs.mujoco import half_cheetah
import numpy as np

CONTROL_WEIGHT = 0.5
ALIVE_BONUS = 1.
IMU_LIMIT = 0.8


class HalfcheetahMotorEnv(half_cheetah.HalfCheetahEnv):
  """The Halfcheetah Environment used in the NoRML paper.

  This environment is derived from the HalfCheetah environment in OpenAI Gym.
  To simulate a wiring error, the actions to the front and rear hip joints could
  be swapped.

  For details about the environment, refer to experiment section of
  https://arxiv.org/abs/1903.01063
  """
  metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 100}

  def __init__(self, swap_action=False):
    """Initializes the environment.

    Args:
      swap_action: whether the action dimensions of the two leg joints are
        swapped. If True, all actions passed into step() will have dimension 0
        and 3 swapped (corresponding to front and rear hip joint).
    """
    self._swap_action = swap_action
    super(HalfcheetahMotorEnv, self).__init__()

  def step(self, action):
    """Steps the environment forward.

    Args:
      action: a 6-dimensional vector specifying the torques to each joint. Joint
        index 0 and index 3 corresponds to the front and rear hip joint. To make
        the task harder, we also force an episode to terminate early if the body
        tilts beyond limit.

    Returns:
      observation: the next observation
      reward: reward gained in the current timestep
      done: whether the current episode is completed
      info: other information
    """
    if self._swap_action:
      action[0], action[3] = action[3], action[0]
    obs, reward, done, info = super(HalfcheetahMotorEnv, self).step(action)
    # Add reward for staying alive
    reward = info[
        'reward_run'] + info['reward_ctrl'] * CONTROL_WEIGHT + ALIVE_BONUS
    # Prevent falling or abnormal walking behavior by adding falling detection
    pitch_angle = self.sim.data.qpos.flat[2]
    done = done or np.cos(pitch_angle) < IMU_LIMIT
    return obs, reward, done, info

  def _get_obs(self):
    """Returns the observation at the current timestep.

    Different from the original cheetah environment, we removed the x,y position
    and linear velocity from the observation space, as these observations aren't
    generally available and requires precise, real-time tracking systems.

    Returns:
    observation: a 14-dimensional array containing follows:
      - current pitch angle (1-d)
      - joint positions (6-d)
      - current pitch rate (1-d)
      - joint velocities (6-d)
    """
    return np.concatenate([
        self.sim.data.qpos.flat[2:],
        self.sim.data.qvel.flat[2:],
    ])

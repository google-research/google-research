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

r"""Simple Reinforcement Learning test environment.


Kinematic point mass environment similar to the one from the MAML paper.
"""
import gym
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np


class MovePointEnv(gym.Env):
  """Simple point mass gym environment.

  The goal is to move an agent (point) from a start location to a goal
  position. Each time step, the agent can move in any direction. In addition,
  each action to the point could be rotated by a fixed angle.

  The agent is limited to a [-2, 2] range in X and Y dimensions.

  Args:
    start_pos: Starting position of the point.
    end_pos: Ending position of the point.
    goal_reached_distance: The episode terminates early if the agent is within
      this distance to end_pos.
    trial_length: Maximum length of the episode.
    action_rotation: The degree to rotate the action by, used to test NoRML.
    sparse_reward: If true, the reward is -1 until the episode terminates,
      otherwise the reward is the negative distance to end_pos.
  """

  def __init__(self,
               start_pos,
               end_pos,
               goal_reached_distance=0.1,
               trial_length=100,
               action_rotation=0.,
               sparse_reward=False):
    self._start_pos = np.array(start_pos).reshape((-1, 2))
    self._current_pos = self._start_pos
    self._end_pos = np.array(end_pos).reshape((-1, 2))
    self._action_rotation = action_rotation
    self._sparse_reward = sparse_reward
    if np.abs(self._start_pos).max() > 2:
      raise ValueError('Start position out of bounds.')
    if np.abs(self._end_pos).max() > 2:
      raise ValueError('End position out of bounds.')
    self._positions_log = [self._start_pos]
    self._goal_reached_distance = goal_reached_distance
    self._trial_length = trial_length
    self._step = 0
    self.action_space = gym.spaces.Box(
        -np.ones(2), np.ones(2), dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        np.ones(2) * -2, np.ones(2) * 2, dtype=np.float32)

  def reset(self):
    self._current_pos = self._start_pos
    self._positions_log = [self._start_pos]
    self._step = 0
    return self._get_observation()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: displacement vector.

    Returns:
      observations: The new position of the robot after the action.
      reward: The reward for the current state-action pair (negative distance to
        goal).
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.
    """
    self._step += 1
    rot_matrix = np.array(
        [[np.cos(self._action_rotation), -np.sin(self._action_rotation)],
         [np.sin(self._action_rotation),
          np.cos(self._action_rotation)]])
    new_pos = self._current_pos + rot_matrix.dot(action)
    new_pos = np.clip(new_pos,
                      np.ones(new_pos.shape) * -2,
                      np.ones(new_pos.shape) * 2)
    distance = np.sqrt(np.sum((new_pos - self._end_pos)**2))
    reward = -1. if self._sparse_reward else -distance
    self._current_pos = new_pos
    self._positions_log.append(new_pos)

    done = (distance < self._goal_reached_distance) or (self._step >=
                                                        self._trial_length)

    return self._get_observation(), reward, done, {}

  def _get_observation(self):
    return np.copy(self._current_pos).reshape((-1, 2))

  def render(self, mode='rgb_array', margin=0.1, limits=((-2, 2), (-2, 2))):
    if mode != 'rgb_array':
      raise ValueError('Only rgb_array is supported.')
    fig = plt.figure()

    pos = np.vstack(self._positions_log)
    plt.plot(pos[:, 0], pos[:, 1], 'b.-')
    plt.plot(self._start_pos[:, 0], self._start_pos[:, 1], 'r+')
    plt.plot(self._end_pos[:, 0], self._end_pos[:, 1], 'g+')

    plt.xlim(limits[0])
    plt.ylim(limits[1])

    plt.gca().set_aspect('equal', adjustable='box')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return data

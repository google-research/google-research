# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utility for loading the 2D navigation environments."""
from typing import Optional

import gym
import numpy as np
import scipy


WALLS = {
    'Small':  # max_goal_dist = 3
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':  # max_goal_dist = 9
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'FourRooms':  # max_goal_dist = 14
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U':  # max_goal_dist = 14
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Spiral11x11':  # max_goal_dist = 45
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Maze11x11':  # max_goal_dist = 49
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls


class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               walls = None, resize_factor = 1):
    """Initialize the point environment.

    Args:
      walls: (str or array) binary, H x W array indicating locations of walls.
        Can also be the name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = 0.01
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0, 0, 0, 0]),
        high=np.array([height, width, height, width]),
        dtype=np.float32)
    self.reset()

  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=float)
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    return state

  def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    self.goal = self._sample_empty_state()
    self.state = self._sample_empty_state()
    return self._get_obs()

  def _discretize_state(self, state, resolution=1.0):
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    assert len(state) == 2
    if (np.any(state < self.observation_space.low[:2])
        or np.any(state > self.observation_space.high[:2])):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    action = action.copy()
    if not self.action_space.contains(action):
      print('WARNING: clipping invalid action:', action)
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise, (2,))
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state

    done = False
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    rew = float(dist < 2.0)
    return obs, rew, done, {}

  @property
  def walls(self):
    return self._walls


class PointImage(PointEnv):
  """An image-based 2D navigation environment."""

  def __init__(self, *args, **kwargs):
    self._dist = []
    self._dist_vec = []
    super(PointImage, self).__init__(*args, **kwargs)
    self.observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)

  def reset(self):
    self._dist_vec = []
    self._dist = []
    self.goal = self._sample_empty_state()
    self._goal_img = self._get_img(self.goal)
    self.state = self._sample_empty_state()
    self._dist.append(np.linalg.norm(self.state - self.goal))
    return self._get_obs()

  def step(self, action):
    super(PointImage, self).step(action)
    dist = np.linalg.norm(self.state - self.goal)
    self._dist.append(dist)
    s = self._get_obs()
    r = float(dist < 2.0)
    done = False
    info = {}
    return s, r, done, info

  def _get_img(self, state):
    scale = 30
    img = resize_walls(self.walls, scale)
    img = 0.5 * (1 - img)
    radius = 10
    low_i, low_j = np.clip((state * scale).astype(int) - radius,
                           [0, 0], img.shape)
    high_i, high_j = np.clip((state * scale).astype(int) + radius,
                             [0, 0], img.shape)
    img[low_i:high_i, low_j:high_j] = 1
    (h, w) = img.shape
    img = (255 * img).astype(np.uint8)
    img = scipy.ndimage.zoom(img, (64 / h, 64 / w), order=0)
    img = np.stack([img, img, img], axis=-1)
    return img

  def _get_obs(self):
    return np.concatenate([
        self._get_img(self.state).flatten(),
        self._goal_img.flatten()
    ])

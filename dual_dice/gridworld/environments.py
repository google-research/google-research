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

"""Simple grid-world environment.

The task here is to walk to the (max_x, max_y) position in a square grid.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from typing import Any, Dict, Tuple, Union


class GridWalk(object):
  """Walk on grid to target location."""

  def __init__(self, length, tabular_obs = True):
    """Initializes the environment.

    Args:
      length: The length of the square gridworld.
      tabular_obs: Whether to use tabular observations. Otherwise observations
        are x, y coordinates.
    """
    self._length = length
    self._tabular_obs = tabular_obs
    self._x = np.random.randint(length)
    self._y = np.random.randint(length)
    self._n_state = length ** 2
    self._n_action = 4
    self._target_x = length - 1
    self._target_y = length - 1

  def reset(self):
    """Resets the agent to a random square."""
    self._x = np.random.randint(self._length)
    self._y = np.random.randint(self._length)
    return self._get_obs()

  def _get_obs(self):
    """Gets current observation."""
    if self._tabular_obs:
      return self._x * self._length + self._y
    else:
      return np.array([self._x, self._y])

  def get_tabular_obs(self, xy_obs):
    """Gets tabular observation given non-tabular (x,y) observation."""
    return self._length * xy_obs[Ellipsis, 0] + xy_obs[Ellipsis, 1]

  def get_xy_obs(self, state):
    """Gets (x,y) coordinates given tabular observation."""
    x = state // self._length
    y = state % self._length
    return np.stack([x, y], axis=-1)

  def step(self, action):
    """Perform a step in the environment.

    Args:
      action: A valid action (one of 0, 1, 2, 3).

    Returns:
      next_obs: Observation after action is applied.
      reward: Environment step reward.
      done: Whether the episode has terminated.
      info: A dictionary of additional environment information.

    Raises:
      ValueError: If the input action is invalid.
    """
    if action == 0:
      if self._x < self._length - 1:
        self._x += 1
    elif action == 1:
      if self._y < self._length - 1:
        self._y += 1
    elif action == 2:
      if self._x > 0:
        self._x -= 1
    elif action == 3:
      if self._y > 0:
        self._y -= 1
    else:
      raise ValueError('Invalid action %s.' % action)
    taxi_distance = (np.abs(self._x - self._target_x) +
                     np.abs(self._y - self._target_y))
    reward = np.exp(-2 * taxi_distance / self._length)
    done = False
    return self._get_obs(), reward, done, {}

  @property
  def num_states(self):
    return self._n_state  # pytype: disable=bad-return-type  # bind-properties

  @property
  def num_actions(self):
    return self._n_action

  @property
  def state_dim(self):
    return 1 if self._tabular_obs else 2

  @property
  def action_dim(self):
    return self._n_action

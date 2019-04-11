# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock environment for testing reinforcement learning code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import gym.spaces
import numpy as np


class MockEnvironment(object):
  """Generate random agent input and keep track of statistics."""

  def __init__(self, observ_shape, action_shape, min_duration, max_duration):
    """Generate random agent input and keep track of statistics.

    Args:
      observ_shape: Shape for the random observations.
      action_shape: Shape for the action space.
      min_duration: Minimum number of steps per episode.
      max_duration: Maximum number of steps per episode.

    Attributes:
      steps: List of actual simulated lengths for all episodes.
      durations: List of decided lengths for all episodes.
    """
    self._observ_shape = observ_shape
    self._action_shape = action_shape
    self._min_duration = min_duration
    self._max_duration = max_duration
    self._random = np.random.RandomState(0)
    self.steps = []
    self.durations = []

  @property
  def observation_space(self):
    low = np.zeros(self._observ_shape)
    high = np.ones(self._observ_shape)
    return gym.spaces.Box(low, high)

  @property
  def action_space(self):
    low = np.zeros(self._action_shape)
    high = np.ones(self._action_shape)
    return gym.spaces.Box(low, high)

  @property
  def unwrapped(self):
    return self

  def step(self, action):
    assert self.action_space.contains(action)
    assert self.steps[-1] < self.durations[-1]
    self.steps[-1] += 1
    observ = self._current_observation()
    reward = self._current_reward()
    done = self.steps[-1] >= self.durations[-1]
    info = {}
    return observ, reward, done, info

  def reset(self):
    duration = self._random.randint(self._min_duration, self._max_duration + 1)
    self.steps.append(0)
    self.durations.append(duration)
    return self._current_observation()

  def _current_observation(self):
    return self._random.uniform(0, 1, self._observ_shape)

  def _current_reward(self):
    return self._random.uniform(-1, 1)

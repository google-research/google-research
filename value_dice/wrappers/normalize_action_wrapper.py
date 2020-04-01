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

"""Wrapper to normalize gyn.spaces.Box actions in [-1, 1].
"""

from absl import logging
import gym
from gym import spaces
import numpy as np


class NormalizeBoxActionWrapper(gym.ActionWrapper):
  """Rescale the action space of the environment."""

  def __init__(self, env):
    if not isinstance(env.action_space, spaces.Box):
      raise ValueError('env %s does not use spaces.Box.' % str(env))
    super(NormalizeBoxActionWrapper, self).__init__(env)
    self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

  def action(self, action):
    # rescale the action
    low, high = self.env.action_space.low, self.env.action_space.high
    scaled_action = low + (action + 1.0) * (high - low) / 2.0
    scaled_action = np.clip(scaled_action, low, high)

    return scaled_action

  def reverse_action(self, scaled_action):
    low, high = self.env.action_space.low, self.env.action_space.high
    action = (scaled_action - low) * 2.0 / (high - low) - 1.0
    return action


def check_and_normalize_box_actions(env):
  """Wrap env to normalize actions if [low, high] != [-1, 1]."""
  low, high = env.action_space.low, env.action_space.high

  if isinstance(env.action_space, spaces.Box):
    if (np.abs(low + np.ones_like(low)).max() > 1e-6 or
        np.abs(high - np.ones_like(high)).max() > 1e-6):
      logging.info('Normalizing environment actions.')
      return NormalizeBoxActionWrapper(env)

  # Environment does not need to be normalized.
  return env

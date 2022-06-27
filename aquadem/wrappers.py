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

"""AQuaDem environment wrappers."""

from typing import Any, Dict, Tuple

from acme import wrappers
import dm_env
import gym
import numpy as np


class AdroitSparseRewardWrapper(gym.Wrapper):
  """Wrapper for Adroit replacing the reward with the sparse goal_achieved flag."""

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    # The goal_achieved condition of the environment is consistent with the
    # reward thresholds we use to sparsify the demonstrations in utils.py
    return observation, float(info['goal_achieved']), done, dict(reward=reward)


class SuccessRewardWrapper(wrappers.base.EnvironmentWrapper):
  """Wrapper for Adroit replacing the reward with a success reward."""

  def __init__(self, environment, success_threshold):
    """Initializes a new SuccessRewardWrapper.

    This wrapper replaces the reward based on whether a threshold of
    return has been reached during an episode, in which case a reward of 1
    is returned (only the first time the threshold has been reached).
    Therefore when evaluating on this reward, the return is equal to
    1 if the threshold is reached, 0 otherwise.

    Args:
      environment: Environment to wrap.
      success_threshold: Minimum return for success.
    """
    super().__init__(environment)
    self._success_threshold = success_threshold

  def _convert_timestep(self, action,
                        timestep):

    if timestep.reward >= self._success_threshold and not self._success:
      success_reward = np.float32(1)
      self._success = True
    else:
      success_reward = np.float32(0)

    return timestep._replace(reward=success_reward)

  def step(self, action):
    return self._convert_timestep(action, self._environment.step(action))

  def reset(self):
    timestep = self._environment.reset()
    self._success = False
    return timestep

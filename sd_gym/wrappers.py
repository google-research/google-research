# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Wrappers for creating gym environments."""

import gym
import numpy as np

from sd_gym import core
from sd_gym import env as env_lib


class FlattenScaleAction(gym.ActionWrapper):
  """Action wrapper that flattens and scales the action."""

  def __init__(self, env, a, b):
    super(FlattenScaleAction, self).__init__(env)
    # flatten
    self.flattened_space = gym.spaces.utils.flatten_space(self.env.action_space)
    # rescale
    assert isinstance(
        self.flattened_space, gym.spaces.Box
    ), "expected Box action space, got {}".format(type(self.flattened_space))
    assert np.less_equal(a, b).all(), (a, b)

    self.a = np.zeros(self.flattened_space.shape,
                      dtype=self.flattened_space.dtype) + a
    self.b = np.zeros(self.flattened_space.shape,
                      dtype=self.flattened_space.dtype) + b
    self.action_space = gym.spaces.Box(
        low=a, high=b,
        shape=self.flattened_space.shape,
        dtype=self.flattened_space.dtype
    )

  def action(self, action):
    # rescale
    assert np.all(np.greater_equal(action, self.a)), (action, self.a)
    assert np.all(np.less_equal(action, self.b)), (action, self.b)
    low = self.flattened_space.low
    high = self.flattened_space.high
    action = low + (high - low) * ((action - self.a) / (self.b - self.a))
    action = np.clip(action, low, high)
    # unflatten
    return gym.spaces.utils.unflatten(self.env.action_space, action)

  def reverse_action(self, action):
    pass
    # Implement reverse action:


def make_sd_env(env_params):
  """Makes an environment, flattens action and observation spaces for agents."""
  env = env_lib.SDEnv(env_params)
  env = gym.wrappers.FlattenObservation(env)
  env = FlattenScaleAction(env, -1, 1)
  return env

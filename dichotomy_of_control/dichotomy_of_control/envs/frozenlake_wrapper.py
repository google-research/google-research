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

import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np


class FrozenLakeWrapper(FrozenLakeEnv):

  def __init__(self, num_states):
    super().__init__()
    self.num_states = num_states

  def step(self, action):
    action = np.argmax(action, axis=-1)
    obs, reward, done, info = super().step(action)
    obs_onehot = np.zeros(self.num_states, np.float32)
    obs_onehot[obs] = 1.

    return obs_onehot, reward, done, info

  def reset(self):
    obs = super().reset()
    obs_onehot = np.zeros(self.num_states, np.float32)
    obs_onehot[obs] = 1.
    return obs_onehot

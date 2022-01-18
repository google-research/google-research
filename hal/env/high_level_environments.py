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

# Lint as: python3
"""Non-hierarchical high-level environment."""

from __future__ import absolute_import
from __future__ import division

from clevr_robot_env import ClevrEnv
import numpy as np

MAX_STARTING = -8  # reset until reward smaller than this value
MIN_REWARD = 0  # reward will be set to mininum is smaller than this value


class HighLevelEnv:
  """High-level task for sorting objects.

  Attributes:
    env: environment on which the high level env is built
    sparse: use sparse reward
    start_far: start far away from goal state
    max_starting: maximum reward of the starting state
  """

  def __init__(self, base_env, sparse=True, start_far=False):
    """Initialize high level environment.

    Args:
      base_env: environment on which the high level env is built
      sparse: whether the reward is sparse
      start_far: start far away from potential goal state
    """
    assert isinstance(base_env, ClevrEnv), 'base_env is not a ClevrEnv'
    assert not base_env.variable_scene_content, 'only supports fixed object'
    self.env = base_env
    self.sparse = sparse
    self.start_far = start_far
    self.max_starting = MAX_STARTING if start_far else -1.

  def __getattr__(self, attr):
    return getattr(self.env, attr)

  def step(self, a):
    obs, _, done, _ = self.env.step(a)
    rew = self.reward()
    done = self._is_done(rew)
    return obs, rew, done, None

  def reset(self, max_reset=50):
    """Resets the environment."""
    obs = self.env.reset()
    r = self._complete()
    rep = 0
    while r > self.max_starting and rep < max_reset:
      obs = self.env.reset()
      r = self._complete()
      rep += 1
    return obs

  def reward(self):
    """Computes reward of current environment."""
    curr_state = np.array([self.get_body_com(name) for name in self.obj_name])
    pair_sorted = 0
    for i, coord in enumerate(curr_state[:-1]):
      obj_sorted = curr_state[i+1][0] < coord[0] - 0.01
      obj_sorted = obj_sorted and abs(curr_state[i+1][1] - coord[1]) < 0.4
      pair_sorted += float(obj_sorted)
    if self.sparse:
      reward = 0. if pair_sorted == len(curr_state) - 1 else -10.
    else:
      reward = pair_sorted * 2 - 10
    return reward

  def _complete(self):
    curr_state = np.array([self.get_body_com(name) for name in self.obj_name])
    num_pair_sorted = 0
    for i, coord in enumerate(curr_state[:-1]):
      pair_sorted = curr_state[i+1][0] < coord[0]-0.01
      pair_sorted = pair_sorted and abs(curr_state[i+1][1]-coord[1]) < 0.4
      num_pair_sorted += float(pair_sorted)
    return 10 if num_pair_sorted > 1 else -10

  def _is_done(self, reward):
    return reward == 0.

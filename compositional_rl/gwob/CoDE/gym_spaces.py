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

"""Discrete Gym space with configurable dtype parameter."""
# For more information,
# visit https://github.com/openai/gym/blob/master/gym/spaces/discrete.py

import gym
import gym.spaces
import numpy as np


class Discrete(gym.spaces.Discrete):
  """A discrete space in gym with configurable dtype.

    Default discrete space in gym doesn't allow changing dtype of which the
    default value (np.int64) could be a problem for an RL framework.

    Example:
        >>> Discrete(2, dtype=np.int32)
  """

  def __init__(self, n, dtype=np.int64):
    if dtype != np.int64 and dtype != np.int32:
      raise ValueError(
          "Data type of the discrete space should be 32bit or 64bit integer but found {}"
          .format(dtype))
    super().__init__(n)
    self.dtype = dtype

  def sample(self):
    return self.np_random.randint(self.n, dtype=self.dtype)

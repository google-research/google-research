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

"""Reward functions for SDEnv.

These transforms are used to extract scalar rewards from state variables.
"""
import numpy as np

from sd_gym import core


class NullReward(core.RewardFn):
  """Reward is always 0."""

  def __call__(self, obs):
    del obs  # Unused.
    return 0


class ScalarDeltaReward(core.RewardFn):
  """Computes reward from a positive change in a scalar state variable."""

  def __init__(self, dict_key_or_custom_fn=None, scaling_fn=None):
    """Initializes ScalarDeltaReward.

    Args:
      dict_key_or_custom_fn: String key for the observation used to compute the
          reward, or a custom function.
      scaling_fn: Optional function to rescale reward value.
    """
    self._dict_key = None
    self._custom_fn = None

    if isinstance(dict_key_or_custom_fn, str):
      self._dict_key = dict_key_or_custom_fn
    elif callable(dict_key_or_custom_fn):
      self._custom_fn = dict_key_or_custom_fn
    else:
      raise AttributeError('dict_key_or_custom_fn must be a string or function')
    self._scaling_fn = scaling_fn
    self.is_reset = False

  def _compute(self, obs):
    if self._custom_fn:
      rew = self._custom_fn(obs)
      if isinstance(rew, np.ndarray):
        if rew.ndim == 0:
          return float(rew.item())
        if rew.ndim == 1 and rew.shape[0] == 1:
          return float(rew.item(0))
      if isinstance(rew, float):
        return rew

      raise ValueError('custom function should return a single item')

    return float(obs[self._dict_key])

  def __call__(self, obs):
    if not self.is_reset:
      raise AssertionError('Reward must be reset before simulation')

    current_val = self._compute(obs)
    retval = current_val - self.last_val
    self.last_val = current_val
    if self._scaling_fn:
      return np.sign(retval) * self._scaling_fn(abs(retval))
    return retval

  def reset(self, obs):
    self.is_reset = True
    self.last_val = self._compute(obs)
    return self


class NegativeScalarDeltaReward(ScalarDeltaReward):
  """Computes reward from a negative change in a scalar state variable."""

  def __call__(self, obs):
    delta = super(NegativeScalarDeltaReward, self).__call__(obs)
    return -delta


class BinarizedScalarDeltaReward(ScalarDeltaReward):
  """Computes reward from the sign of a positive change in a state variable."""

  def __call__(self, obs):
    delta = super(BinarizedScalarDeltaReward, self).__call__(obs)
    if delta == 0:
      return None
    return int(delta > 0)


class NegativeBinarizedScalarDeltaReward(ScalarDeltaReward):
  """Computes reward from the sign of a negative change in a state variable."""

  def __call__(self, obs):
    delta = super(NegativeBinarizedScalarDeltaReward, self).__call__(obs)
    if delta == 0:
      return None
    return int(delta < 0)

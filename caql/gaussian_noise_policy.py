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

"""Gaussian noise policy."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np

from caql import policy


class GaussianNoisePolicy(policy.BasePolicy):
  """Implementation for gaussian noise policy."""

  def __init__(self, greedy_policy, sigma, sigma_decay, sigma_min):
    """Creates an epsilon greedy policy.

    Args:
      greedy_policy: policy.BasePolicy. The policy that is used to compute a
        greedy action.
      sigma: float. Standard deviation for a gaussian distribution.
      sigma_decay: float. Decay rate for the sigma.
      sigma_min: float. The minimum value of the sigma.
    """
    if not 0 <= sigma <= 1.0:
      raise ValueError('sigma should be in [0.0, 1.0]')

    self._greedy_policy = greedy_policy
    self._sigma = sigma
    self._sigma_decay = sigma_decay
    self._sigma_min = sigma_min

  @property
  def sigma(self):
    return self._sigma

  def _action(self, state, use_action_function, batch_mode=False):
    mean_action = self._greedy_policy.action(state, use_action_function,
                                             batch_mode)
    if mean_action is None:
      return None
    batch_action_dim = np.shape(mean_action)
    # Match the scale of noise value to action value.
    noise_exploration = (
        self._sigma * self._greedy_policy.action_spec.maximum *
        np.random.randn(*batch_action_dim))
    return mean_action + noise_exploration

  def _update_params(self):
    self._sigma = max(self._sigma * self._sigma_decay, self._sigma_min)

  def _params_debug_str(self):
    return 'sigma: %.3f' % self._sigma

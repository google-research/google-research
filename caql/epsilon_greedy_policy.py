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

"""Epsilon-greedy policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from caql import policy


class EpsilonGreedyPolicy(policy.BasePolicy):
  """Implementation for epsilon-greedy policy."""

  def __init__(self, greedy_policy, epsilon, epsilon_decay, epsilon_min):
    """Creates an epsilon greedy policy.

    Args:
      greedy_policy: policy.BasePolicy. The policy that is used to compute
        a greedy action.
      epsilon: float. The chance of random action.
      epsilon_decay: float. Decay rate for the epsilon.
      epsilon_min: float. The minimum value of the epsilon.
    """
    if not 0 <= epsilon <= 1.0:
      raise ValueError('epsilon should be in [0.0, 1.0]')

    self._greedy_policy = greedy_policy
    self._epsilon = epsilon
    self._epsilon_decay = epsilon_decay
    self._epsilon_min = epsilon_min

  @property
  def epsilon(self):
    return self._epsilon

  def _action(self, state, use_action_function, batch_mode=False):
    if np.random.random() < self._epsilon:
      if self._greedy_policy.continuous_action:
        return np.random.uniform(self._greedy_policy.action_spec.minimum,
                                 self._greedy_policy.action_spec.maximum,
                                 self._greedy_policy.action_spec.shape)
      else:
        # Discrete action.
        return np.random.randint(self._greedy_policy.action_spec.minimum,
                                 self._greedy_policy.action_spec.maximum + 1)
    return self._greedy_policy.action(state, use_action_function, batch_mode)

  def _update_params(self):
    self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)

  def _params_debug_str(self):
    return 'epsilon: %.3f' % self._epsilon

# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Basic utilities for off-policy policy evaluation algorithms."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import numpy as np
import six

from typing import Any, Callable

import dual_dice.policy as policy_lib
import dual_dice.transition_data as transition_data_lib


@six.add_metaclass(abc.ABCMeta)
class BaseAlgo(object):
  """Abstract class representing algorithm for off-policy corrections."""

  @abc.abstractmethod
  def solve(self, data, target_policy):
    """Trains or solves for policy evaluation given experience and policy."""

  @abc.abstractmethod
  def estimate_average_reward(self, data, target_policy):
    """Estimates value (average per-step reward) of policy."""

  def close(self):
    pass


def estimate_value_from_state_ratios(
    data,
    target_policy,
    gamma,
    state_ratio_fn):
  """Estimates value of policy given data and state density ratios.

  Args:
    data: The experience data to base the estimate on.
    target_policy: The policy whose value to estimate.
    gamma: Discount to use in the value calculation.
    state_ratio_fn: A function taking in batches of states and returning
      estimates of the ratio d^pi(s) / d^D(s), where d^pi(s) is the discounted
      occupancy of the target policy at state s and d^D(s) is the probability
      with which state s appears in the experience data.

  Returns:
    Estimated average per-step reward of the target policy.
  """
  all_data = data.get_all()
  state_density_ratio = state_ratio_fn(all_data.state)
  policy_ratio = policy_lib.get_policy_ratio(
      data.policy, target_policy,
      all_data.state, all_data.action)
  state_action_density_ratio = state_density_ratio * policy_ratio
  # Multiply by discount to account for discounted behavior policy.
  weights = state_action_density_ratio * gamma ** all_data.time_step
  return np.sum(all_data.reward * weights) / np.sum(weights)


def estimate_value_from_state_action_ratios(
    data,
    gamma,
    state_action_ratio_fn):
  """Estimates value of policy given data and state-action density ratios.

  Args:
    data: The experience data to base the estimate on.
    gamma: Discount to use in the value calculation.
    state_action_ratio_fn: A function taking in batches of states and actions
      and returning estimates of the ratio d^pi(s, a) / d^D(s, a), where
      d^pi(s, a) is the discounted occupancy of the target policy at
      state-action (s, a) and d^D(s, a) is the probability with which
      state-action pair (s, a) appears in the experience data.

  Returns:
    Estimated average per-step reward of the target policy.
  """
  all_data = data.get_all()
  state_action_density_ratio = state_action_ratio_fn(
      all_data.state, all_data.action)
  # Multiply by discount to account for discounted behavior policy.
  weights = state_action_density_ratio * gamma ** all_data.time_step
  return np.sum(all_data.reward * weights) / np.sum(weights)

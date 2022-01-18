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

"""Tabular approximatation of density ratio using DualDICE.

Based on the paper `DualDICE: Behavior-Agnostic Estimation of Discounted
Stationary Distribution Corrections' by Ofir Nachum, Yinlam Chow, Bo Dai,
and Lihong Li. See https://arxiv.org/abs/1906.04733
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import dual_dice.algos.base as base_algo
import dual_dice.policy as policy_lib
import dual_dice.transition_data as transition_data_lib


class TabularDualDice(base_algo.BaseAlgo):
  """Approximate the density ratio using exact matrix solves."""

  def __init__(self, num_states, num_actions, gamma,
               solve_for_state_action_ratio = False):
    """Initializes the solver.

    Args:
      num_states: The number of discrete states in the environment.
      num_actions: The number of discrete actions in the environment.
      gamma: The discount to use.
      solve_for_state_action_ratio: Whether to solve for state-action density
        ratio. Defaults to False, which instead solves for state density ratio.
        Although the estimated policy value should be the same, approximating
        using the state density ratio is much faster (especially in large
        environments) and more accurate (especially in low-data regimes).
    """
    self._num_states = num_states
    self._num_actions = num_actions
    self._gamma = gamma
    self._solve_for_state_action_ratio = solve_for_state_action_ratio
    self._dimension = (self._num_states * self._num_actions
                       if self._solve_for_state_action_ratio
                       else self._num_states)
    self._nu = np.zeros([self._dimension])
    self._zeta = np.zeros([self._dimension])

  def _get_index(self, state, action):
    if self._solve_for_state_action_ratio:
      return state * self._num_actions + action
    else:
      return state

  def solve(self,
            data,
            target_policy,
            regularizer = 1e-8):
    """Solves for density ratios and then approximates target policy value.

    Args:
      data: The transition data store to use.
      target_policy: The policy whose value we want to estimate.
      regularizer: A small constant to add to matrices before inverting them or
        to floats before taking square root.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    td_residuals = np.zeros([self._dimension, self._dimension])
    total_weights = np.zeros([self._dimension])
    initial_weights = np.zeros([self._dimension])
    for transition in data.iterate_once():
      nu_index = self._get_index(transition.state, transition.action)
      weight = self._gamma ** transition.time_step

      td_residuals[nu_index, nu_index] += weight
      total_weights[nu_index] += weight

      next_probs = target_policy.get_probabilities(transition.next_state)
      policy_ratio = policy_lib.get_policy_ratio(data.policy, target_policy,
                                                 transition.state,
                                                 transition.action)

      # Need to weight next nu by importance weight.
      next_weight = (weight if self._solve_for_state_action_ratio else
                     policy_ratio * weight)
      for next_action, next_prob in enumerate(next_probs):
        next_nu_index = self._get_index(transition.next_state, next_action)
        td_residuals[next_nu_index, nu_index] += (
            -next_prob * self._gamma * next_weight)

      initial_probs = target_policy.get_probabilities(transition.initial_state)
      for initial_action, initial_prob in enumerate(initial_probs):
        initial_nu_index = self._get_index(transition.initial_state,
                                           initial_action)
        initial_weights[initial_nu_index] += weight * initial_prob

    td_residuals /= np.sqrt(regularizer + total_weights)[None, :]
    td_errors = np.dot(td_residuals, td_residuals.T)
    self._nu = np.linalg.solve(
        td_errors + regularizer * np.eye(self._dimension),
        (1 - self._gamma) * initial_weights)
    self._zeta = np.dot(self._nu,
                        td_residuals) / np.sqrt(regularizer + total_weights)
    return self.estimate_average_reward(data, target_policy)

  def estimate_average_reward(self, data, target_policy):
    """Estimates value (average per-step reward) of policy.

    The estimation is based on solved values of zeta, so one should call
    solve() before calling this function.

    Args:
      data: The transition data store to use.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    if self._solve_for_state_action_ratio:
      return base_algo.estimate_value_from_state_action_ratios(
          data, self._gamma,
          lambda state, action: self._zeta[self._get_index(state, action)])
    else:
      return base_algo.estimate_value_from_state_ratios(
          data, target_policy, self._gamma,
          lambda state: self._zeta[self._get_index(state, None)])

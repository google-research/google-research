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

"""Common utility functions."""

import collections

from absl import logging
import numpy as np

from rl_metrics_aaai2021 import bisimulation
from rl_metrics_aaai2021 import d_delta
from rl_metrics_aaai2021 import d_delta_star
from rl_metrics_aaai2021 import discrete_bisimulation
from rl_metrics_aaai2021 import discrete_lax_bisimulation
from rl_metrics_aaai2021 import lax_bisimulation


MetricData = collections.namedtuple('metric_data', ['constructor', 'label'])


MDPStats = collections.namedtuple(
    'MDPStats', ['time', 'num_iterations', 'min_gap', 'avg_gap', 'max_gap'])


# Dictionary mapping metric name to constructor and LaTeX label.
METRICS = {
    'bisimulation':
        MetricData(bisimulation.Bisimulation, r'$d^{\sim}$'),
    'discrete_bisimulation':
        MetricData(discrete_bisimulation.DiscreteBisimulation, r'$e^{\sim}$'),
    'lax_bisimulation':
        MetricData(lax_bisimulation.LaxBisimulation, r'$d^{\sim_{lax}}$'),
    'discrete_lax_bisimulation':
        MetricData(discrete_lax_bisimulation.DiscreteLaxBisimulation,
                   r'$e^{\sim_{lax}}$'),
    'd_delta_1':
        MetricData(d_delta.DDelta1, r'$d_{\Delta1}$'),
    'd_delta_5':
        MetricData(d_delta.DDelta5, r'$d_{\Delta5}$'),
    'd_delta_10':
        MetricData(d_delta.DDelta10, r'$d_{\Delta10}$'),
    'd_delta_15':
        MetricData(d_delta.DDelta15, r'$d_{\Delta15}$'),
    'd_delta_20':
        MetricData(d_delta.DDelta20, r'$d_{\Delta20}$'),
    'd_delta_50':
        MetricData(d_delta.DDelta50, r'$d_{\Delta50}$'),
    'd_delta_100':
    MetricData(d_delta.DDelta100, r'$d_{\Delta100}$'),
    'd_delta_500':
    MetricData(d_delta.DDelta500, r'$d_{\Delta500}$'),
    'd_delta_1000':
    MetricData(d_delta.DDelta1000, r'$d_{\Delta1000}$'),
    'd_delta_5000':
    MetricData(d_delta.DDelta5000, r'$d_{\Delta5000}$'),
    'd_Delta_star':
        MetricData(d_delta_star.DDeltaStar, r'$d_{\Delta^*}$'),
}


def value_iteration(env, tolerance, verbose=False):
  """Run value iteration on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.
    tolerance: float, error tolerance used to exit loop.
    verbose: bool, whether to print verbose messages.

  Returns:
    Numpy array with V* and Q*.
  """
  values = np.zeros(env.num_states)
  q_values = np.zeros((env.num_states, env.num_actions))
  error = tolerance * 2
  i = 0
  while error > tolerance:
    new_values = np.copy(values)
    for s in range(env.num_states):
      for a in range(env.num_actions):
        q_values[s, a] = (
            env.rewards[s, a] +
            env.gamma * np.matmul(env.transition_probs[s, a, :], values))
      new_values[s] = np.max(q_values[s, :])
    error = np.max(abs(new_values - values))
    values = new_values
    i += 1
    if i % 1000 == 0 and verbose:
      logging.info('Error after %d iterations: %f', i, error)
  if verbose:
    logging.info('Found V* in %d iterations', i)
    logging.info(values)
  return values, q_values


def q_value_iteration(env, tolerance):
  """Run q value iteration on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.
    tolerance: float, error tolerance used to exit loop.

  Returns:
    Numpy array with V* and Q*.
  """
  q_values = np.zeros((env.num_states, env.num_actions))
  error = tolerance * 2
  i = 0
  while error > tolerance:
    for s in range(env.num_states):
      for a in range(env.num_actions):
        old_q_values = np.copy(q_values[s, a])
        q_values[s, a] = (
            env.rewards[s, a] + env.gamma *
            np.matmul(env.transition_probs[s, a, :], np.max(q_values, axis=1)))
        error = np.max(abs(old_q_values - q_values[s, a]))
    i += 1
  return q_values


def policy_iteration(env, tolerance, verbose=False):
  """Run policy iteration on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.
    tolerance: float, evaluation stops when the value function change is less
    than the tolerance.
    verbose: bool, whether to print verbose messages.

  Returns:
    Numpy array with V*
  """
  values = np.zeros(env.num_states)
  # Random policy
  policy = np.ones((env.num_states, env.num_actions)) / env.num_actions
  policy_stable = False
  i = 0
  while not policy_stable:
    # Policy evaluation
    while True:
      delta = 0.
      for s in range(env.num_states):
        v = np.sum(env.rewards[s, :] * policy[s, :] + env.gamma * policy[s, :] *
                   np.matmul(env.transition_probs[s, :, :], values))
        delta = max(delta, abs(v - values[s]))
        values[s] = v
      if delta < tolerance:
        break
    # Policy improvement
    policy_stable = True
    for s in range(env.num_states):
      old = policy[s].copy()
      g = np.zeros(env.num_actions, dtype=float)
      for a in range(env.num_actions):
        g[a] = (
            env.rewards[s, a] +
            env.gamma * np.matmul(env.transition_probs[s, a, :], values))
      greed_actions = np.argwhere(g == np.amax(g))
      for a in range(env.num_actions):
        if a in greed_actions:
          policy[s, a] = 1 / len(greed_actions)
        else:
          policy[s, a] = 0
      if not np.array_equal(policy[s], old):
        policy_stable = False

    i += 1
    if i % 1000 == 0 and verbose:
      logging.info('Error after %d iterations: %f', i, delta)
  if verbose:
    logging.info('Found V* in %d iterations', i)
    logging.info(values)
  return values


def q_policy_iteration(env, tolerance, verbose=False):
  """Run policy iteration on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.
    tolerance: float, evaluation stops when the value function change is less
    than the tolerance.
    verbose: bool, whether to print verbose messages.

  Returns:
    Numpy array with V*
  """
  q_values = np.zeros((env.num_states, env.num_actions))
  # Random policy
  policy = np.ones((env.num_states, env.num_actions)) / env.num_actions
  policy_stable = False
  i = 0
  while not policy_stable:
    # Policy evaluation
    while True:
      delta = 0.
      for s in range(env.num_states):
        v = env.rewards[s, :] + env.gamma * np.matmul(
            env.transition_probs[s, :, :], np.sum(q_values * policy, axis=1))
        delta = max(delta, np.max(abs(v- q_values[s])))
        q_values[s] = v
      if delta < tolerance:
        break
    # Policy improvement
    policy_stable = True
    for s in range(env.num_states):
      old = policy[s].copy()
      greedy_actions = np.argwhere(q_values[s] == np.amax(q_values[s]))
      for a in range(env.num_actions):
        if a in greedy_actions:
          policy[s, a] = 1 / len(greedy_actions)
        else:
          policy[s, a] = 0
      if not np.array_equal(policy[s], old):
        policy_stable = False

    i += 1
    if i % 1000 == 0 and verbose:
      logging.info('Error after %d iterations: %f', i, delta)
  if verbose:
    logging.info('Found V* in %d iterations', i)
    logging.info(q_values)
  return q_values

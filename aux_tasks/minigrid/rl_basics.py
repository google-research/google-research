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

"""Reusable implementation of basic RL algorithms."""

from absl import logging
import numpy as np


def get_state_xy(idx, num_cols):
  """Given state index this method returns its equivalent coordinate (x,y).

  Args:
    idx: index uniquely identifying a state
    num_cols: number of colums

  Returns:
  values x, y describing the state's location in the grid
  """
  y = int(idx % num_cols)
  x = int((idx - y) / num_cols)
  return x, y


def get_state_idx(x, y, num_cols):
  """Given state (x,y),  returns the index that uniquely identifies this state.

  Args:
    x: value of the coordinate x
    y: value of the coordinate y
    num_cols: number of colums

  Returns:
    unique index identifying a position in the grid
  """
  idx = y + x * num_cols
  return idx


def policy_random(env):
  r"""Random policy on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.

  Returns:
    Numpy array S \times A: random policy
  """
  return np.ones((env.num_states, env.num_actions)) / env.num_actions


def policy_eps_suboptimal(env, optimal_policy, epsilon=0):
  r"""Epsilon suboptimal policy.

  Takes random action with probability epsilon and
  optimal action with prob 1 - epsilon on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.
    optimal_policy: Numpy array S \times A with optimal policy
    epsilon: float in [0, 1]

  Returns:
    Numpy array S \times A: policy followed by the agent
  """
  return epsilon * policy_random(env) + (1 - epsilon) * optimal_policy


def policy_iteration(env, gamma=0.99, tolerance=1e-5, verbose=False):
  """Run policy iteration on env.

  Args:
    env: a MiniGrid environment, including the MDPWrapper.
    gamma: float, discount factor.
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
        v = np.sum(env.rewards[s, :] * policy[s, :] + gamma * policy[s, :] *
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
            gamma * np.matmul(env.transition_probs[s, a, :], values))
      action = np.argmax(g)
      for a in range(env.num_actions):
        if a == action:
          policy[s, a] = 1.
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
  return values, policy

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

"""Saved policies for grid-world domains."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import Union

from dual_dice.gridworld import environments
import dual_dice.policy as policy


def get_gridwalk_policy(grid_env,
                        tabular_obs = True,
                        epsilon_explore = 0.0):
  """Creates an optimal policy for gridwalk with some exploration.

  Args:
    grid_env: The gridwalk environment.
    tabular_obs: Whether the environment returns tabular observations.
    epsilon_explore: Probability between 0 and 1 with which to explore.
      Default to 0 (no exploration).

  Returns:
    An optimal policy for the environment, with uniform exploration occurring
      some amount of time (as determined by epsilon_explore).
  """

  def _policy_fn(state):
    """Gets optimal action distribution mixed with uniform exploration."""
    if not tabular_obs:
      state = grid_env.get_tabular_obs(state)

    # Optimal policy takes shortest path to (max_x, max_y) point in grid.
    xy = grid_env.get_xy_obs(state)
    x, y = xy[Ellipsis, 0], xy[Ellipsis, 1]
    actions = np.where(x <= y, 0, 1)  # Increase x or increase y by 1.
    probs = np.zeros((actions.size, 4))
    probs[np.arange(actions.size), actions] = 1
    probs = probs.reshape(list(actions.shape) + [4])

    return (probs * (1 - epsilon_explore) +
            0.25 * np.ones([4]) * epsilon_explore)

  return policy.DiscreteFunctionPolicy(_policy_fn)


def get_behavior_gridwalk_policy(grid_env,
                                 tabular_obs = True,
                                 alpha = 0.5):
  """Gets behavior policy for off-policy experiments.

  Args:
    grid_env: Gridwalk environment.
    tabular_obs: Whether the environment uses tabular observations.
    alpha: How close to the optimal policy.  Higher alpha corresponds to more
      optimal policy.

  Returns:
    A sub-optimal policy for the environment.
  """
  return get_gridwalk_policy(grid_env, tabular_obs,
                             epsilon_explore=0.1 + 0.6 * (1 - alpha))


def get_target_gridwalk_policy(
    grid_env,
    tabular_obs = True):
  """Gets target policy for off-policy experiments.

  Args:
    grid_env: Gridwalk environment.
    tabular_obs: Whether the environment uses tabular observations.

  Returns:
    A near-optimal policy for the environment.
  """
  return get_gridwalk_policy(grid_env, tabular_obs, epsilon_explore=0.1)

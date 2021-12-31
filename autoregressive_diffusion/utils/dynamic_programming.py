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

"""Contains dynamic programming routines.
"""
from typing import Tuple

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np


Array = jnp.ndarray


def get_nelbo_matrix(kl_per_t):
  """Computes nelbo matrix so that nelbos[s, t] is going to contain the value logp(x_s | x_t)."""
  num_timesteps = len(kl_per_t)

  # Somewhat unintuitive code to create this mask, looking at a print is way
  # more intuitive. In the case of num_timesteps = 3, it would look like:
  # [0 1 2 3]
  # [0 0 1 2]
  # [0 0 0 1]
  # [0 0 0 0], i.e. to build nelbos[s, t] = -log(x_s | x_t)
  triu = np.triu(np.ones((num_timesteps, num_timesteps)))
  triu = np.cumsum(triu[::-1], axis=0)[::-1]

  # Compute nelbos[s, t] is going to contain the value logp(x_s | x_t). Only
  # considering entries where s > t.
  nelbos_ = kl_per_t[:, None] * triu
  nelbos = np.zeros((num_timesteps + 1, num_timesteps + 1))
  # Last row / first column are zero to match up with costs/dimensions
  # matrices.
  nelbos[:-1, 1:] = nelbos_

  return nelbos


@jax.jit
def jitted_inner_cost_and_dimension_loop(
    nelbos, first_cost):
  """Inner jax-loop that computes the cost and dimension matrices."""
  num_timesteps = first_cost.shape[0] - 1

  def compute_next_cost(prev_cost, _):
    bpds = prev_cost[:, None] + nelbos
    new_dimension = jnp.argmin(bpds, axis=0)
    new_cost = jnp.min(bpds, axis=0)
    return new_cost, (new_cost, new_dimension)

  # Putting the loop inside a lax.scan is very important to get the speedup,
  # otherwise it is slower than numpy. For a more intuitive understanding
  # check out the numpy version get_cost_and_dimension_matrices_np method.
  _, (costs, dimensions) = jax.lax.scan(
      compute_next_cost, init=first_cost, xs=jnp.arange(1, num_timesteps+1))

  return costs, dimensions


def get_cost_and_dimension_matrices(kl_per_t):
  """Compute cost and assignment matrices, in jax."""
  num_timesteps = len(kl_per_t)
  kl_per_t = jnp.array(kl_per_t)

  # costs[k, t] is going to contain the cost to generate t steps but limited to
  # a policy with length k.
  first_cost = np.full((num_timesteps + 1,), np.inf, dtype=np.float)
  first_cost[0] = 0
  first_cost = jnp.array(first_cost)

  # dimensions[k, t] is going to contain the optimal previous t for D[k - 1].
  # First row just contains -1 and is never used, but this way it aligns with
  # the cost matrix.
  first_dimension = jnp.full((num_timesteps + 1), -1, dtype=np.int32)

  # nelbos[s, t] is going to contain the value logp(x_s | x_t)
  nelbos = jnp.array(get_nelbo_matrix(kl_per_t))

  costs, dimensions = jitted_inner_cost_and_dimension_loop(nelbos, first_cost)

  # Concatenate first rows to the matrices.
  costs = jnp.concatenate([first_cost[None, :], costs], axis=0)
  dimensions = jnp.concatenate([first_dimension[None, :], dimensions], axis=0)

  costs = np.array(costs)
  dimensions = np.array(dimensions)

  return costs, dimensions


def get_cost_and_dimension_matrices_np(kl_per_t):
  """Compute cost and assignment matrices in numpy."""
  num_timesteps = len(kl_per_t)

  # costs[k, t] is going to contain the cost to generate t steps but limited to
  # a policy with length k.
  costs = np.full(
      (num_timesteps + 1, num_timesteps + 1), np.inf, dtype=np.float)
  costs[0, 0] = 0

  # dimensions[k, t] is going to contain the optimal previous t for D[k - 1].
  dimensions = np.full(
      (num_timesteps + 1, num_timesteps + 1), -1, dtype=np.int32)

  # nelbos[s, t] is going to contain the value logp(x_s | x_t)
  nelbos = get_nelbo_matrix(kl_per_t)

  # Compute cost and assignment matrices.
  for k in range(1, num_timesteps+1):
    # More efficient, we only have to consider costs <=k:
    bpds = costs[k-1, :k+1, None] + nelbos[:k+1, :]
    dimensions[k] = np.argmin(bpds, axis=0)
    # Use argmin to get minimum to save time, equiv to calling np.amin.
    amin = bpds[dimensions[k], np.arange(num_timesteps+1)]
    costs[k] = amin

    # # Easier to interpret but more expensive, leaving it here for clarity:
    # bpds = costs[k-1, :, None] + nelbos
    # dimensions[k] = np.argmin(bpds, axis=0)
    # # Use argmin to get minimum to save time, equiv to calling np.amin.
    # amin = bpds[dimensions[k], np.arange(num_timesteps+1)]
    # costs[k] = amin

  return costs, dimensions


def get_optimal_path_with_budget(budget, costs, dimensions):
  """Compute optimal path."""
  t = costs.shape[0] - 1
  path = []
  opt_cost = costs[budget, t]
  for k in reversed(range(1, budget+1)):
    t = dimensions[k, t]
    path.append(t)

  # Path is reversed, reverse back.
  path = list(reversed(path))
  path = jnp.asarray(path, jnp.int32)

  return path, opt_cost


def compute_cost_with_policy(kl_per_t, policy):
  """Computes the cost for a specific policy.

  This is particularly useful if you want to extract a policy using training
  data, but you want to compute the expected cost for validation data.

  Args:
    kl_per_t: array with kl cost per time step.
    policy: policy (array with steps) to traverse through the generative
      process.

  Returns:
    cost: float

  """
  nelbos = get_nelbo_matrix(kl_per_t)
  cost = 0
  kprev = 0

  policy = list(policy) + [len(kl_per_t)]

  for k in policy:
    cost += nelbos[kprev, k]
    kprev = k

  return cost


def compute_fixed_budget(kl_per_t, budgets):
  """Computes given an array kl_per_t and multiple budgets the optimal policy."""
  msg = f'Computing cost / dimension matrices with size {len(kl_per_t)}.'
  logging.info(msg)

  cost_matrix, dimension_matrix = get_cost_and_dimension_matrices(kl_per_t)

  policies = []
  costs = []
  for budget in budgets:
    path, opt_cost = get_optimal_path_with_budget(
        budget, cost_matrix, dimension_matrix)
    policies.append(path)
    costs.append(opt_cost)

  return policies, costs

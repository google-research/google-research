# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Implements sequential policy mechanisms (Gaussian and Greedy).

Code is adapted from the following Github Repos:
https://github.com/heyyjudes/differentially-private-set-union/tree/master
https://github.com/ricardocarvalhods/diff-private-set-union/blob/main/dpsu_gw.py

The implementations of the policy methods from these repos are broken down into
three seperate functions:
1. A descent step (either L1, L2 norm, or greedy)
2. A function to compute the policy histogram by calling the descent step
sequentially on the user itemsets
3. The overall mechanism code which calls the above two functions, adds noise to
the policy histogram, thresholds the noisy counts, and returns the output set.
"""

import collections
import enum

import numpy as np

from dp_mm_domain import utils
from dp_mm_domain import weighted_gaussian


Enum = enum.Enum


class Descent(Enum):
  """Enum class for policy descent metrics."""
  L1 = 1
  L2 = 2


class Policy(Enum):
  """Enum class for set union policies."""
  GAUSSIAN_L1 = 1
  GAUSSIAN_L2 = 2
  GREEDY = 3


class OrderedCounter(collections.Counter, collections.OrderedDict):
  pass


def run_lnorm_descent_step(policy_hist, items, threshold, descent=Descent.L1):
  """Runs either an L1 or L2 norm descent update on the policy histogram.

  Based on Algorithms 7 and 8 from https://arxiv.org/pdf/2002.09745.pdf.

  Args:
    policy_hist: A dictionary mapping items to their policy weighted frequency.
    items: A list of items belonging to a single user to consider for the
      descent update.
    threshold: The threshold for the policy.
    descent: The descent metric to use for updating the policy histogram.
      Currently supports Descent.L1 (default) and Descent.L2.

  Returns:
    A dictionary mapping items to their updated policy weighted frequency.
  """
  selected_items = [item for item in items if policy_hist[item] < threshold]

  if selected_items:
    if descent == Descent.L2:
      # The code for the L2 descent step is taken from Lines 346-356 in
      # https://github.com/heyyjudes/differentially-private-set-union/blob/master/histogram.py
      diff_arr = np.asarray(
          [threshold - policy_hist[item] for item in selected_items]
      )
      z = np.linalg.norm(diff_arr, ord=2)

      for i, item in enumerate(selected_items):
        policy_hist[item] += min(1.0, z) * diff_arr[i] / (z)
    else:
      # The code for the L1 descent step is taken from Lines 400-423 in
      # https://github.com/heyyjudes/differentially-private-set-union/blob/master/histogram.py
      gap_dict = {}

      for item in selected_items:
        if policy_hist[item] < threshold:
          gap_dict[item] = threshold - policy_hist[item]

      sorted_gap_dict = sorted(gap_dict.items(), key=lambda x: x[0])
      sorted_gap_keys = [k for k, _ in sorted_gap_dict]

      user_budget = 1.0
      total_items = len(sorted_gap_keys)

      for i, w in enumerate(sorted_gap_keys):
        cost = gap_dict[w] ** 2 * (total_items - i)
        if cost < user_budget:
          policy_hist[w] = threshold
          user_budget -= gap_dict[w] ** 2
        else:
          for j in range(i, total_items):
            add_item = sorted_gap_keys[j]
            policy_hist[add_item] += user_budget / np.sqrt(total_items - i)
          break

  return policy_hist


def run_greedy_descent_step(policy_hist, items, threshold):
  """Runs a greedy descent update on the policy histogram.

  Code is taken from Lines 28-61 in
  https://github.com/ricardocarvalhods/diff-private-set-union/blob/main/dpsu_gw.py.

  Args:
    policy_hist: A dictionary mapping items to their policy weighted frequency.
    items: A list of items belonging to a single user to consider for the
      descent update.
    threshold: The threshold for the policy.

  Returns:
    A dictionary mapping items to their updated policy weighted frequency.
  """

  selected_items = [item for item in items if policy_hist[item] < threshold]

  if selected_items:
    all_item_counter = OrderedCounter(items)
    rho_dict = {}
    for _, w in enumerate(selected_items):
      if policy_hist[w] < threshold:
        rho_dict[w] = all_item_counter[w]

    sorted_rho_dict = sorted(
        rho_dict.items(), key=lambda item: item[1], reverse=True
    )
    sorted_rho_keys = [k for k, _ in sorted_rho_dict]

    user_budget = 1.0
    for _, w in enumerate(sorted_rho_keys):
      cost = threshold - policy_hist[w]

      if cost < user_budget:
        policy_hist[w] += cost
        user_budget -= cost
      else:
        policy_hist[w] += user_budget
        break

  return policy_hist


def get_policy_hist(input_data, threshold, policy):
  """Computes the policy histogram for a given policy.

  Args:
    input_data: A list of lists, where each inner list contains items (e.g.,
      ngrams) contributed by a single user.
    threshold: The threshold for the policy.
    policy: The policy to use for computing the policy histogram. Currently
      supports Policy.GAUSSIAN_L1, Policy.GAUSSIAN_L2, and Policy.GREEDY.

  Returns:
    A dictionary mapping items to their policy weighted frequency.
  Raises:
    ValueError: If the policy is not supported.
  """
  policy_hist = collections.defaultdict(float)
  if policy == Policy.GAUSSIAN_L1:
    for items in input_data:
      policy_hist = run_lnorm_descent_step(
          policy_hist, items, threshold, Descent.L1
      )
  elif policy == Policy.GAUSSIAN_L2:
    for items in input_data:
      policy_hist = run_lnorm_descent_step(
          policy_hist, items, threshold, Descent.L2
      )
  elif policy == Policy.GREEDY:
    for items in input_data:
      policy_hist = run_greedy_descent_step(policy_hist, items, threshold)
  else:
    raise ValueError(f"Unsupported policy: {policy}")
  return policy_hist


def policy_gaussian_mechanism(
    input_data, l0_bound, eps, delta, descent=Descent.L1, alpha=3
):
  """Runs a policy to privately select items based on user contributions.

  This function implements the Policy Gaussian mechanism by Gopi et al. (2020)
  from https://arxiv.org/pdf/2002.09745.pdf.

  Args:
    input_data: A list of lists, where each inner list contains items (e.g.,
      ngrams) contributed by a single user.
    l0_bound: The maximum number of items a single user can contribute.
    eps: The epsilon privacy parameter for the mechanism.
    delta: The delta privacy parameter for the mechanism.
    descent: The descent metric to use for updating the policy histogram.
      Currently supports Descent.L1 (default) and Descent.L2.
    alpha: A scaling factor used in calculating the policy threshold.

  Returns:
    A set of items whose noisy weights are above the calculated threshold.
  """
  clipped_users = utils.l0_bound_users(input_data, l0_bound)

  delta = delta / 2

  # Uses the same sigma and threshold as the weighted gaussian mechanism. Note
  # that we halve the delta parameter before calling the WGM threshold function.
  sigma, wgm_threshold = (
      weighted_gaussian.get_weighted_gaussian_sigma_and_threshold(
          eps, delta, l0_bound
      )
  )

  policy_threshold = wgm_threshold + alpha * sigma

  if descent == Descent.L1:
    policy = Policy.GAUSSIAN_L1
  elif descent == Descent.L2:
    policy = Policy.GAUSSIAN_L2
  else:
    raise ValueError(f"Unsupported descent: {descent}")

  policy_hist = get_policy_hist(clipped_users, policy_threshold, policy)

  output = set()
  for item, weighted_count in policy_hist.items():
    nval = weighted_count + np.random.normal(0.0, sigma)
    if nval > wgm_threshold:
      output.add(item)

  return output


def policy_greedy_mechanism(input_data, eps, delta, alpha=3):
  """Runs a greedy policy to privately select items based on user contributions.

  This function implements the Greedy updates Without sampling (GW) mechanism
  by Carvalho et al. (2022) from
  https://cdn.aaai.org/ojs/21183/21183-13-25196-1-2-20220628.pdf.

  Args:
    input_data: A list of lists, where each inner list contains items (e.g.,
      ngrams) contributed by a single user.
    eps: The epsilon privacy parameter for the mechanism.
    delta: The delta privacy parameter for the mechanism.
    alpha: A scaling factor used in calculating the policy threshold.

  Returns:
    A set of items whose noisy weights are above the calculated threshold.
  """

  # The following parameters are taken from Lines 15-17 in
  # https://github.com/ricardocarvalhods/diff-private-set-union/blob/main/dpsu_gw.py
  l_param = 1 / eps
  l_rho = 1 - (1 / eps) * np.lib.scimath.log(2 * delta)

  # policy_threshold needs to be at least 1.0 in order to not have a dependence
  # on the l0_bound. See Theorem 1 in the above paper.
  policy_threshold = max(l_rho + alpha * l_param, 1.0)

  policy_hist = get_policy_hist(input_data, policy_threshold, Policy.GREEDY)

  output = set()
  for item, weighted_count in policy_hist.items():
    nval = weighted_count + np.random.laplace(0, l_param)
    if nval > l_rho:
      output.add(item)

  return output

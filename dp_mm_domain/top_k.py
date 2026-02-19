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

"""Implements Limited Domain mechanism for private top-k."""

import functools

import numpy as np

from dp_mm_domain import utils

partial = functools.partial


def compute_limited_domain_eps(local_eps, delta_prime, k):
  """Computes overall epsilon for Limited Domain mechanism given local epsilon and delta_prime.

  Based on Theorem 1 in https://arxiv.org/pdf/1905.04273.

  Args:
    local_eps: The epsilon parameter for each individual mechanism.
    delta_prime: The delta_prime parameter in Theorem 1 in
      https://arxiv.org/pdf/1905.04273.
    k: The number of elements to release.

  Returns:
    The overall epsilon parameter for the Limited Domain mechanism.
  """
  term1 = k * local_eps
  term2 = k * local_eps * (
      (np.exp(local_eps) - 1) / (np.exp(local_eps) + 1)
  ) + local_eps * np.sqrt(2 * k * np.log(1 / delta_prime))
  term3 = (k * (local_eps**2)) / 2 + local_eps * np.sqrt(
      (1 / 2) * k * np.log(1 / delta_prime)
  )
  return min(term1, term2, term3)


def get_local_eps_delta(eps, delta, delta_prime, k, tolerance=1e-6):
  """Computes the largest local epsilon and local delta for overall mechanism to be (eps, delta)-DP.

  Args:
    eps: The overall privacy guarantee will be (eps, delta)-DP.
    delta: The overall privacy guarantee will be (eps, delta)-DP.
    delta_prime: The delta_prime parameter in Theorem 1 in
      https://arxiv.org/pdf/1905.04273.
    k: Number of items to release.
    tolerance: Float accuracy for computed value. Note that this errs on the
      side of being conservative.

  Returns:
    The largest local epsilon and local delta parameters such that the limited
    domain mechanism is (eps, delta)-DP.

  Raises:
    ValueError: If delta_prime >= delta.
  """

  if delta_prime >= delta:
    raise ValueError(
        'delta_prime must be smaller than delta.'
    )

  binary_search_function = partial(
      compute_limited_domain_eps, delta_prime=delta_prime, k=k
  )

  local_eps = utils.binary_search(
      function=binary_search_function,
      threshold=eps,
      tolerance=tolerance,
      decreasing=False,
  )

  local_delta = delta - delta_prime

  return local_eps, local_delta


def get_hbot(sorted_hist, k_bar, delta, eps, l0_bound):
  """Computes the h-bot parameter for the Limited Domain mechanism.

  Args:
    sorted_hist: A sorted list of (item, count) pairs, sorted in decreasing
      order of count.
    k_bar: An upper bound on the number of distinct items in the input.
    delta: The delta privacy parameter.
    eps: The epsilon privacy parameter.
    l0_bound: An upper bound on the number of unique items each user
      contributes.

  Returns:
    The computed h-bot parameter as a float.
  """

  if k_bar >= len(sorted_hist):
    return 1 + np.log(min(l0_bound, k_bar) / delta) / eps
  else:
    return (
        sorted_hist[k_bar][1] + 1 + np.log(min(l0_bound, k_bar) / delta) / eps
    )


def limited_domain_mechanism(input_data, k, k_bar, eps, delta, l0_bound):
  """Implements the Limited Domain mechanism (Algorithm 1) from https://arxiv.org/abs/1905.04273.

  Args:
    input_data: A list of lists, where each inner list contains elements from
      the domain.
    k: The number of top items to release.
    k_bar: A bound on the number of distinct items in the input.
    eps: The total epsilon privacy parameter.
    delta: The total delta privacy parameter.
    l0_bound: An upper bound on the number of unique items each user
      contributes.

  Returns:
    A list containing the released top-k items.
  """
  clipped_data = utils.l0_bound_users(input_data, l0_bound)

  # We split the delta budget between computing the local epsilon and the noisy
  # threshold.
  delta_prime = delta / 2
  local_eps, local_delta = get_local_eps_delta(eps, delta, delta_prime, k)

  freq_hist = utils.get_hist(clipped_data)
  sorted_hist = sorted(freq_hist.items(), key=lambda x: x[1], reverse=True)

  h_bot = get_hbot(sorted_hist, k_bar, local_delta, local_eps, l0_bound)
  v_bot = h_bot + np.random.gumbel(scale=1 / local_eps)

  noisy_top_kbar_items = []
  for i in range(k_bar):
    noisy_top_kbar_items.append((
        sorted_hist[i][0],
        sorted_hist[i][1] + np.random.gumbel(scale=1 / local_eps),
    ))

  sorted_noisy_top_kbar_items = sorted(
      noisy_top_kbar_items, key=lambda x: x[1], reverse=True
  )
  output = []
  for i in range(k):
    if sorted_noisy_top_kbar_items[i][1] < v_bot:
      break
    else:
      output.append(sorted_noisy_top_kbar_items[i][0])

  return output

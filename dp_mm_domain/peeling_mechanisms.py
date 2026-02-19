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

"""Implements Peeling mechanisms for unknown domains."""


import numpy as np

from dp_mm_domain import utils
from dp_mm_domain import weighted_gaussian


def cdp_peeling_mechanism(freq_hist, domain, k, eps, delta):
  """Computes (epsilon, delta)-DP top-k counts using the CDP peeling mechanism.

  The peeling  mechanism (https://arxiv.org/pdf/1905.04273.pdf) adaptively uses
  the counts as a utility function for the exponential mechanism. Once an item
  is selected, the item is removed from the item set and we repeat this
  procedure with the remaining items until k items are selected. Here we use the
  Gumbel trick to simulate the mechanism. The Gumbel trick adds Gumbel noise to
  the counts with parameter epsilon' and returns the indices of items with
  highest noisy counts. epsilon' is computed using CDP composition of the
  exponential mechanism (see function utils.em_epsilon_cdp) so that the overall
  mechanism is (eps, delta)-DP.

  Code adapted from
  https://github.com/google-research/google-research/blob/master/dp_topk/baseline_mechanisms.py

  Args:
    freq_hist: A dictionary mapping items to their frequencies (counts).
    domain: A set of elements that are in the domain.
    k: The number of top elements to return.
    eps: The epsilon parameter for (eps, delta)-DP.
    delta: The delta parameter for (eps, delta)-DP.

  Returns:
    A list containing the top-k elements according to the noisy counts.
  """
  local_eps = utils.em_epsilon_cdp(eps, delta, k)

  noisy_hist = []
  for item in domain:
    noisy_freq = freq_hist.get(item, 0) + np.random.gumbel(
        scale=1 / local_eps
    )
    noisy_hist.append((item, noisy_freq))

  sorted_noisy_hist = sorted(noisy_hist, key=lambda x: x[1], reverse=True)
  return [item[0] for item in sorted_noisy_hist[:k]]


def user_peeling_mechanism(input_data, domain, k, eps, delta):
  """Implements a private mechanism for selecting a hitting set using user peeling.

  This mechanism iteratively selects items based on a noisy count of users
  containing them, and then "peels" away those users from consideration
  in subsequent iterations.

  Args:
    input_data: A list of lists, where each inner list contains items
      contributed by a single user.
    domain: A set of elements that are in the domain.
    k: The number of items to return.
    eps: The epsilon parameter for (eps, delta)-DP.
    delta: The delta parameter for (eps, delta)-DP.

  Returns:
    A set of k items selected by the mechanism.
  """
  n = len(domain)
  domain = set(domain)
  output = set()
  items_to_users = utils.get_items_to_users(input_data)
  freq_hist = utils.get_hist(input_data)

  for _ in range(min(k, n)):
    if not items_to_users:
      break
    # Despite picking the item with the highest noisy count, we still need to
    # pass k to the cdp_peeling_mechanism in order to account for the privacy
    # loss due to k-fold composition.
    max_item = cdp_peeling_mechanism(freq_hist, domain, k, eps, delta)[0]
    output.add(max_item)
    domain = domain - set([max_item])
    items_to_users, freq_hist = utils.remove_users_with_item(
        items_to_users, freq_hist, input_data, max_item
    )

  return output


def wgm_then_peel_mechanism(
    input_data, k, eps_schedule, delta_schedule, l0_bound, peel_users=False
):
  """Applies a two-stage mechanism: WGM for domain reduction, then peeling.

  This mechanism first uses the Weighted Gaussian Mechanism (WGM) to privately
  select a smaller domain of elements. Then, it applies a known-domain peeling
  mechanism on the input data, restricted to the selected domain, to output k
  items. The privacy budget (epsilon, delta) is split between the two stages.

  Args:
    input_data: A list of lists, where each inner list contains items
      contributed by a single user.
    k: The number of elements to return.
    eps_schedule: A list or tuple containing two epsilon values, [eps_1, eps_2].
      eps_1 is used for the WGM stage, and eps_2 is used for the peeling stage.
    delta_schedule: A list or tuple containing two delta values, [delta_1,
      delta_2]. delta_1 is used for the WGM stage, and delta_2 is used for the
      peeling stage.
    l0_bound: An upper bound on the number of unique elements in each inner list
      of `input_data`. Used by the WGM.
    peel_users: If True, the peeling mechanism is applied to the users, rather
      than the items.

  Returns:
    A list containing the k elements found by the peeling mechanism
    on the domain selected by WGM.
  """

  delta_1 = delta_schedule[0]
  delta_2 = delta_schedule[1]
  eps_1 = eps_schedule[0]
  eps_2 = eps_schedule[1]

  domain = set(
      weighted_gaussian.weighted_gaussian_mechanism(
          input_data, l0_bound, eps_1, delta_1
      ).keys()
  )

  if peel_users:
    return user_peeling_mechanism(input_data, domain, k, eps_2, delta_2)
  else:
    freq_hist = utils.get_hist(input_data)
    return cdp_peeling_mechanism(freq_hist, domain, k, eps_2, delta_2)

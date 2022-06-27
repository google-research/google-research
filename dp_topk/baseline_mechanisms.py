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

"""Baseline mechanisms for (differentially private) top-k selection.

This file implements a non-private, exact top-k mechanism, the peeling mechanism
(https://dl.acm.org/doi/10.1145/1835804.1835869),
the Laplace top-k mechanism (https://arxiv.org/abs/2105.08233), and the infinity
norm mechanism (https://arxiv.org/pdf/1501.06095.pdf, see Theorem 4.1.).
"""

import numpy as np
from dp_topk.differential_privacy import NeighborType


def sorted_top_k(item_counts, k):
  """Returns indices of top-k items' counts.

  Indices are sorted in decreasing order by corresponding item count (i.e., the
  index of the item with the largest count comes first).

  Args:
    item_counts: Array of integers defining item counts.
    k: An integer indicating the number of desired items.
  """
  # Partitioning runs in O(d) time.
  top_k_unsorted = np.argpartition(-item_counts, k - 1)[:k]
  # Sorting highest k counts runs in O(k * log(k)) time.
  sorting_order = np.argsort(item_counts[top_k_unsorted])[::-1]
  return top_k_unsorted[sorting_order]


def em_epsilon_cdp(epsilon, delta, k):
  """Computes local epsilon for exp mech to achieve global (epsilon, delta)-DP.

  The tightest known privacy analysis of the peeling mechanism uses concentrated
  differential privacy (CDP). It combines the exponential mechanism's specific
  CDP guarantee with the generic conversion from CDP to approximate DP.

  In more detail, an eps'-DP exponential mechanism is eps'^2/8-CDP
  (Lemmas 3.2 and 3.4 in https://arxiv.org/abs/2004.07223, or more
  explicitly in Corollary 7 of
  https://differentialprivacy.org/exponential-mechanism-bounded-range/), so k
  applications are k * eps'^2/8-CDP (Lemma 1.7 in BS16,
  https://arxiv.org/abs/1605.02065). By Proposition 1.3 in BS16, this is
  (k * eps'^2/8 + eps' * sqrt(k log(1/delta)/2}, delta)-DP for any delta > 0.
  Setting the first term equal to eps, we get
  k * eps'^2/8 + eps' * sqrt(k log(1/delta)/2} - eps = 0.
  Substitute y = eps' * sqrt{k} to get
  (1/8) * y^2 + sqrt{log(1/delta)/2} * y - eps = 0. Then by the quadratic
  formula, eps' * sqrt{k} = y
  = -sqrt{log(1/delta) / 2} + sqrt{log(1/delta) / 2 + eps / 2} / (1/4). Divide
  by sqrt{k} and rearrange to get
  eps' = sqrt{(8 log(1/delta) + 8eps) / k} - sqrt{8 log(1/delta) / k}.


  Args:
    epsilon: The overall privacy guarantee will be (epsilon, delta)-DP.
    delta: The overall privacy guarantee will be (epsilon, delta)-DP.
    k: Number of compositions of the exponential mechanism.

  Returns:
    Parameter local_epsilon such that k local_epsilon-DP applications of the
    exponential mechanism overall satisfy (epsilon, delta)-DP.
  """
  if delta <= 0:
    return epsilon / k
  else:
    log_delta = np.log(1 / delta)
    return max(
        epsilon / k,
        np.sqrt((8 * log_delta + 8 * epsilon) / k) -
        np.sqrt(8 * log_delta / k))


def cdp_peeling_mechanism(item_counts, k, epsilon, delta):
  """Computes (epsilon, delta)-DP top-k counts using the CDP peeling mechanism.

  The peeling  mechanism (https://arxiv.org/pdf/1905.04273.pdf) adaptively uses
  the counts as a utility function for the exponential mechanism. Once an item
  is selected, the item is removed from the item set and we repeat this
  procedure with the remaining items until k items are selected. Here we use the
  Gumbel trick to simulate the mechanism. The Gumbel trick adds Gumbel noise to
  the counts with parameter epsilon' and returns the indices of items with
  highest noisy counts. epsilon' is computed using CDP composition of the
  exponential mechanism (see function em_epsilon_cdp) so that the overall
  mechanism is (epsilon, delta)-DP. Contribution bound c and NeighborType are
  not input parameters because the peeling mechanism has the same definition
  regardless of c or the neighboring relation used.

  Args:
    item_counts: Array of integers defining item counts.
    k: An integer indicating the number of desired items.
    epsilon: The overall privacy guarantee will be (epsilon, delta)-DP.
    delta: The overall privacy guarantee will be (epsilon, delta)-DP.

  Returns:
    An array containing the indices of the items with the top-k noisy counts.
  """
  local_epsilon = em_epsilon_cdp(epsilon, delta, k)
  noisy_counts = item_counts + np.random.gumbel(
      scale=1 / local_epsilon, size=item_counts.shape)
  noisy_sorted_items = sorted_top_k(noisy_counts, k)
  return noisy_sorted_items


def pnf_peeling_mechanism(item_counts, k, epsilon):
  """Computes epsilon-DP top-k counts by the permute-and-flip peeling mechanism.

  The peeling  mechanism (https://arxiv.org/pdf/1905.04273.pdf) adaptively uses
  the counts as a utility function for the exponential mechanism. Once an item
  is selected, the item is removed from the item set and we repeat this
  procedure with the remaining items until k items are selected. Here we use
  permute-and-flip as a replacement for the exponential mechanism as
  permute-and-flip dominates it for pure DP (https://arxiv.org/abs/2010.12603).
  We further use the exponential noise implementation of permute-and-flip for
  speed (https://arxiv.org/abs/2105.07260). Contribution bound c and
  NeighborType are not input parameters because the peeling mechanism has the
  same definition regardless of c or the neighboring relation used.

  Args:
    item_counts: Array of integers defining item counts.
    k: An integer indicating the number of desired items.
    epsilon: The overall mechanism will be epsilon-DP.

  Returns:
    An array containing the indices of the items with the top-k noisy counts.
  """
  mask = np.zeros(item_counts.size, dtype=bool)
  selected_items = np.empty(k, dtype=int)
  for i in range(k):
    noisy_counts = item_counts + np.random.exponential(
        scale=k / epsilon, size=item_counts.size)
    masked_noisy_counts = np.ma.array(noisy_counts, mask=mask)
    selected_items[i] = np.argmax(masked_noisy_counts)
    mask[int(selected_items[i])] = True
  return selected_items


def laplace_mechanism(item_counts, k, c, epsilon, neighbor_type):
  """Computes epsilon-DP top-k counts using the Laplace mechanism.

  The Laplace top-k mechanism (http://www.vldb.org/pvldb/vol13/p293-ding.pdf) is
  a pure-DP algorithm for computing top-k queries. It adds Laplace noise to the
  histogram of counts with parameter k/epsilon. Letting c be the number of item
  counts each user can contribute to, we use c/epsilon when c<k, since in this
  case the regular Laplace mechanism provides pure DP.

  Args:
    item_counts: Array of integers defining item counts.
    k: An integer indicating the number of desired items.
    c: An integer denoting the maximum number of items that each user can
      contribute to.
    epsilon: A float defining privacy budget.
    neighbor_type: Available neighbor types are defined in the NeighborType
      enum.

  Returns:
    An array containing the indices of the items with the top-k noisy counts.
  """
  if neighbor_type is NeighborType.SWAP:
    sensitivity = min(2 * k, 2 * c)
  else:
    sensitivity = min(k, c)
  noisy_counts = item_counts + np.random.laplace(
      scale=sensitivity / epsilon, size=item_counts.shape)
  noisy_sorted_items = sorted_top_k(noisy_counts, k)
  return noisy_sorted_items


def gamma_mechanism(item_counts, k, epsilon):
  """Computes DP top-k items from counts using the infinity norm as utility.

  This mechanism, introduced in Theorem 4.1. in
  https://arxiv.org/pdf/1501.06095.pdf, uses the exponential mechanism with the
  infinity norm as utility function. It can be efficiently implemented by adding
  noise coming from a gamma distribution. Contribution bound c and NeighborType
  are not input parameters because the infinity norm mechanism has the same
  definition regardless of c or the neighboring relation used.

  Args:
    item_counts: Array of integers defining item counts.
    k: An integer indicating the number of desired items.
    epsilon: A float defining privacy budget.

  Returns:
    An array containing the indices of the items with the top-k noisy counts.
  """
  d = len(item_counts)
  radius = np.random.gamma(shape=d + 1, scale=1 / epsilon)
  noise = radius * np.random.uniform(low=-1, high=1, size=d)
  return sorted_top_k(item_counts + noise, k)

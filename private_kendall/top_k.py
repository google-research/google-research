# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Differentially private top-k.

This code is adapted from
https://github.com/google-research/google-research/blob/master/dp_topk/baseline_mechanisms.py.
"""


import numpy as np


def sorted_top_k(item_counts, k):
  """Returns indices of top-k items' counts.

  Indices are sorted in decreasing order by corresponding item count (i.e., the
  index of the item with the largest count comes first).

  Args:
    item_counts: Array of numbers defining item counts.
    k: An integer indicating the number of desired items.
  """
  top_k = np.argsort(item_counts)[-k:][::-1]
  return top_k


def basic_peeling_mechanism(item_counts, k, epsilon, l_inf_sensitivity,
                            monotonic):
  """Computes epsilon-DP top-k on item_counts using basic composition.

  Args:
    item_counts: Array of numbers defining item counts.
    k: An integer indicating the number of desired items.
    epsilon: The output will be epsilon-DP.
    l_inf_sensitivity: A bound on the l_inf sensitivity of item_counts under the
      addition or removal of one user.
    monotonic: Whether or not item_counts is monotonic, i.e., True if and only
      if adding a user does not decrease any count in item_counts.

  Returns:
    A sorted array of the indices of the top k items. See, e.g., Lemmas 4.1 and
    4.2 of https://arxiv.org/abs/1905.04273 for details.
  """
  local_epsilon = epsilon / k
  if not monotonic:
    local_epsilon = local_epsilon / 2
  noisy_counts = item_counts + np.random.gumbel(
      scale=l_inf_sensitivity / local_epsilon, size=item_counts.shape
  )
  return sorted_top_k(noisy_counts, k)

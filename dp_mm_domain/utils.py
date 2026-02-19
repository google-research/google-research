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

"""Helper functions for various mechanisms.

This code is partially taken from
https://github.com/google-research/google-research/blob/master/dp_l2/utils.py
"""

import collections
import itertools
import numpy as np


Counter = collections.Counter


def binary_search(function, threshold, tolerance=1e-3, decreasing=False):
  """Returns minimum or maximum value such that function(value) <= threshold.

  The minimum (maximum) is returned if increasing (decreasing), and the search
  is performed over interval [0, 1].

  Args:
    function: A real-valued function.
    threshold: Float threshold for function.
    tolerance: Float accuracy for computed value. Note that this errs on the
      side of being conservative.
    decreasing: If True, the function is assumed to be decreasing. Otherwise,
      the function is assumed to be increasing.
  """
  left_input = 0
  right_input = 1
  if decreasing:
    while function(right_input) > threshold:
      right_input = 2 * right_input
      left_input = right_input / 2
    while right_input - left_input > tolerance:
      mid_input = (left_input + right_input) / 2
      if function(mid_input) <= threshold:
        right_input = mid_input
      else:
        left_input = mid_input
    return right_input
  else:
    while function(right_input) < threshold:
      left_input = right_input
      right_input = 2 * right_input
    while right_input - left_input > tolerance:
      mid_input = (left_input + right_input) / 2
      if function(mid_input) < threshold:
        left_input = mid_input
      else:
        right_input = mid_input

    return left_input


def l0_bound_users(input_data, l0_bound):
  """Clips users to have at most l0_bound number of items.

  This function takes a list of lists (representing users and their items)
  and limits each user to at most `l0_bound` items by drawing a random subset
  without replacement.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.
    l0_bound: The maximum number of items allowed for each user.

  Returns:
    list of lists: A new dataset where each user's items have been subsampled
                   to have at most `l0_bound` items.
  """

  new_dataset = []
  for elements in input_data:
    elements = list(set(elements))
    s = min(len(elements), l0_bound)
    selected_elements = np.random.choice(
        elements, size=s, replace=False
    ).tolist()
    new_dataset.append(selected_elements)
  return new_dataset


def get_hist(input_data):
  """Computes the histogram mapping each item to its frequency in input_data.

  Removes duplicates in each user's list before computing the histogram.

  Args:
    input_data: A list where each inner list represents a user's items. For
      example, `[[1, 2, 3], [4, 5]]`.

  Returns:
    A dictionary mapping each item to its frequency.
  """
  unique_items_per_user = [set(user_items) for user_items in input_data]
  flattened_data = list(itertools.chain.from_iterable(unique_items_per_user))
  hist = Counter(flattened_data)
  return hist


def remove_elements_from_list_of_lists(list_of_lists, elements_to_remove):
  """Removes elements from a list of lists.

  Args:
    list_of_lists: A list of lists, where each inner list contains elements.
    elements_to_remove: A set of elements to remove from the list of lists.

  Returns:
    A new list of lists, where each inner list contains the elements from the
    original list that are not in the `elements_to_remove` set.
  """
  remove_set = set(elements_to_remove)

  new_list_of_lists = []
  for inner_list in list_of_lists:
    new_inner_list = []
    for element in inner_list:
      if element not in remove_set:
        new_inner_list.append(element)
    new_list_of_lists.append(new_inner_list)
  return new_list_of_lists


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


  Code taken from
  https://github.com/google-research/google-research/blob/master/dp_topk/baseline_mechanisms.py


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
        np.sqrt((8 * log_delta + 8 * epsilon) / k) - np.sqrt(8 * log_delta / k),
    )


def remove_users_with_item(items_to_users, freq_hist, input_data, hitting_item):
  """Removes users containing a specific item from the dataset.

  Updates the frequency histogram and the item to user map.

  Args:
    items_to_users: A map from item to user indices who have the item.
    freq_hist: A Counter mapping each item to its frequency in the current
      dataset.
    input_data: A list where each inner list represents a user's items.
    hitting_item: The item whose associated users should be removed from
      item_to_users.

  Returns:
    A tuple containing the updated item_to_users map and the updated frequency
    histogram.
  """
  users_to_remove = items_to_users[hitting_item]
  for user in users_to_remove:
    for item in set(input_data[user]):
      freq_hist[item] -= 1
      remaining_users = items_to_users[item] - users_to_remove
      items_to_users[item] = remaining_users

  return items_to_users, freq_hist


def get_items_to_users(input_data):
  """Returns a map from item to the set of users who have the item.

  Args:
    input_data: A list of lists, where each inner list contains items
      contributed by a single user.
  """
  items_to_hit_users = collections.defaultdict(set)
  for user_index, user in enumerate(input_data):
    for item in user:
      items_to_hit_users[item].add(user_index)
  return items_to_hit_users

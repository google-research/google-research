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

"""Implements Hitting Set mechanisms for unknown domains."""

from dp_mm_domain import utils


def greedy_hitting_set(input_data, k):
  """Returns a set of k items selected by the non-private greedy hitting set algorithm.

  The greedy hitting set algorithm iteratively selects the item held by the most
  users, and then removes those users from consideration in subsequent
  iterations.

  Instead of just tracking the frequency histogram of the items, we track the
  set of users who have each item. This allows us to efficiently look up the
  users who have a given item, allowing us to remove those users from
  consideration when selecting the next item.

  Args:
    input_data: A list of lists, where each inner list is the collection of
      items belonging to a user.
    k: The number of items to release.

  Returns:
    A set of k items representing the greedy hitting set.
  """

  items_to_users = utils.get_items_to_users(input_data)
  freq_hist = utils.get_hist(input_data)
  output_set = set()
  for _ in range(k):
    max_item = max(
        freq_hist,
        key=lambda x: freq_hist[x],
    )
    output_set.add(max_item)
    items_to_users, freq_hist = utils.remove_users_with_item(
        items_to_users, freq_hist, input_data, max_item
    )
  return output_set

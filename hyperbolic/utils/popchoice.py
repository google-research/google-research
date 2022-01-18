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

# Lint as: python3
"""Popular choice helper functions."""


import random
import numpy as np


def degree_of_item(train):
  """Calculates degree of items from user-item pairs."""
  item_to_deg = {}
  for pair in train:
    _, item = pair
    if item in item_to_deg:
      item_to_deg[item] += 1
    else:
      item_to_deg[item] = 1
  return item_to_deg


def sorted_by_degrees(item_to_deg):
  """Sorts items from highest degree to lowest."""
  deg_to_item = {}
  for item in item_to_deg:
    deg = item_to_deg[item]
    if deg in deg_to_item:
      deg_to_item[deg].append(item)
    else:
      deg_to_item[deg] = [item]
  return sorted(deg_to_item.items(), reverse=True)


def rank_all_by_pop(sorted_by_deg):
  """Scores items by degrees. Scores of same degree items are randomized."""
  ranked_list = []
  for _, items in sorted_by_deg:
    if len(items) == 1:
      ranked_list.append(items[0])
    else:
      random.shuffle(items)
      ranked_list.extend(items)
  n_items = len(ranked_list)
  scores_of_items = np.zeros(n_items)
  for i, item in enumerate(ranked_list):
    scores_of_items[item] = n_items -i
  return scores_of_items


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

"""Base class for sampling methods.

Provides interface to sampling methods that allow same signature
for get_scores and select_batch_to_label.
"""

import abc
import numpy as np


class SamplingMethod(object):
  """Base class for sampling methods."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, n, debug_info):
    self.n = n
    self.all_indices = np.arange(self.n, dtype=np.int64)
    self.scores = None
    self.debug_info = debug_info

  @abc.abstractmethod
  def get_scores(
      self,
      already_selected_indices,
      label_budget
  ):
    """Gets scores of the test data for sampling."""
    return np.zeros(self.n, dtype=np.float32)

  def select_batch_to_label(
      self,
      already_selected_indices,
      label_budget,
      update_scores = True,
  ):
    """Returns the indices of batch of samples to label.

    Args:
      already_selected_indices: index of datapoints already selected
      label_budget: labeling budget
      update_scores: whether to update the scores

    Returns:
      indices of samples selected to label
    """
    if (self.scores is None) or update_scores:
      self.scores = self.get_scores(already_selected_indices, label_budget)
    remain_indices = np.setdiff1d(self.all_indices, already_selected_indices)
    sorted_index = np.argsort(self.scores[remain_indices])
    remain_indices = remain_indices[sorted_index]
    newly_selected_indices = remain_indices[:label_budget]
    selected_indices = np.concatenate(
        (already_selected_indices, newly_selected_indices), axis=0
    )
    if self.debug_info:
      min_score = np.min(self.scores[newly_selected_indices])
      max_score = np.max(self.scores[newly_selected_indices])
      print(
          f'Min selected scores: {min_score}, max selected score: {max_score}'
      )
    return selected_indices

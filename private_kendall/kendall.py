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

"""Kendall differentially private feature selection."""


import numpy as np
from scipy import stats

from private_kendall import top_k


def kendall(features, labels):
  """Computes Kendall rank correlation coefficients between features and labels.

  Args:
    features: Matrix of feature vectors. Each row is an example. Assumed to not
      have intercept feature.
    labels: Vector of labels.

  Returns:
    Array of Kendall rank coefficients, where each entry represents the
    correlation between a column of features and labels.
  """
  n, d = features.shape
  coefficients = np.asarray([stats.kendalltau(features[:, idx], labels,
                                              method='asymptotic', variant='b',
                                              alternative='two-sided')[0]
                             for idx in range(d)])
  # Scale coefficients using the definition in the paper
  coefficients = np.abs((n /2) * coefficients)
  return coefficients


def dp_kendall_feature_selection(features, labels, k, epsilon):
  """Runs the DPKendall algorithm from the paper.

  Args:
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
    k: Number of features to select.
    epsilon: The algorithm is epsilon-DP.

  Returns:
    Array of k indices privately selected to have the largest Kendall rank
    correlations with labels.
  """
  # Remove intercept feature
  features = features[:, :-1]
  _, d = features.shape
  split_eps = epsilon / k
  selected_indices = []
  # selected_correlations tracks correlations between features and previously
  # selected features
  selected_correlations = np.zeros((k - 1, d))
  label_coefficients = kendall(features, labels)
  # Sensitivity is 3/2 in the first round and 3 after
  sensitivity = 3/2
  for j in range(k):
    diffs = label_coefficients
    if selected_indices:
      last_idx = selected_indices[-1]
      old_feature = features[:, last_idx]
      selected_correlations[j - 1, :] = kendall(features, old_feature)
      diffs = diffs - (
          np.sum(selected_correlations, axis=0) / len(selected_indices)
      )
      sensitivity = 3
    new_idx = top_k.basic_peeling_mechanism(
        item_counts=diffs,
        k=1,
        epsilon=split_eps,
        l_inf_sensitivity=sensitivity,
        monotonic=False,
    )[0]
    selected_indices.append(new_idx)
    label_coefficients[new_idx] = -np.inf
  # Add back intercept feature
  selected_indices.append(d)
  return np.asarray(selected_indices)

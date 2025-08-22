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

"""Lasso differentially private feature selection.

This file implements the Lasso algorithm (Samp-Agg in
http://proceedings.mlr.press/v23/kifer12/kifer12.pdf) for epsilon-differentially
private feature selection.
"""


import numpy as np

from private_kendall import random_shuffle
from private_kendall import regression
from private_kendall import top_k


def lasso_features(features, labels, k):
  """Returns the k indices with the largest coefficients from a lasso model.

  Args:
    features: Matrix of feature vectors. Each row is an example. Assumed to have
      intercept feature.
    labels: Vector of labels.
    k: Number of largest coefficients to select.

  Returns:
    Array of k feature indices.
  """
  model = regression.nondp(True, features, labels)
  _, d = features.shape
  nonzero_coefficients = np.zeros(d)
  # Ignore intercept feature, which will be added back in dp_lasso_features
  nonzero_coefficients[top_k.sorted_top_k(np.abs(model.flatten[:, -1]), k)] = 1
  return nonzero_coefficients


def dp_lasso_features(features, labels, k, m, epsilon):
  """Applies lasso_features to m random subsets and aggregates using DP top-k.

  Args:
    features: Matrix of feature vectors. Each row is an example. Assumed to have
      intercept feature.
    labels: Vector of labels.
    k: Number of largest features to select.
    m: The data will be partitioned into m random subsets of equal size.
    epsilon: The algorithm is epsilon-DP.

  Returns:
    Vector of k+1 feature indices, where the intercept index is always included.
  """
  n, d = features.shape
  features, labels = random_shuffle.random_shuffle(features, labels)
  batch_size = int(n / m)
  subsample_indices = np.empty((m, d - 1))
  for idx in range(m):
    batch_features = features[batch_size * idx : batch_size * idx + batch_size]
    batch_labels = labels[batch_size * idx : batch_size * idx + batch_size]
    subsample_indices[idx] = lasso_features(batch_features, batch_labels, k)
  summed_indices = np.sum(subsample_indices, axis=0)
  answer_indices = top_k.basic_peeling_mechanism(summed_indices, k, epsilon, 1,
                                                 monotonic=False)
  # Add back intercept
  return np.append(answer_indices, d - 1)

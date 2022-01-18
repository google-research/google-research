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

"""TODO(tsitsulin): add headers, tests, and improve style."""

import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def precision(y_true, y_pred):
  true_positives, false_positives, _, _ = _compute_counts(y_true, y_pred)
  return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
  true_positives, _, false_negatives, _ = _compute_counts(y_true, y_pred)
  return true_positives / (true_positives + false_negatives)


def accuracy_score(y_true, y_pred):
  true_positives, false_positives, false_negatives, true_negatives = _compute_counts(
      y_true, y_pred)
  return (true_positives + true_negatives) / (
      true_positives + false_positives + false_negatives + true_negatives)


def _compute_counts(y_true, y_pred):  # TODO(tsitsulin): add docstring pylint: disable=missing-function-docstring
  contingency = contingency_matrix(y_true, y_pred)
  same_class_true = np.max(contingency, 1)
  same_class_pred = np.max(contingency, 0)
  diff_class_true = contingency.sum(axis=1) - same_class_true
  diff_class_pred = contingency.sum(axis=0) - same_class_pred
  total = contingency.sum()

  true_positives = (same_class_true * (same_class_true - 1)).sum()
  false_positives = (diff_class_true * same_class_true * 2).sum()
  false_negatives = (diff_class_pred * same_class_pred * 2).sum()
  true_negatives = total * (
      total - 1) - true_positives - false_positives - false_negatives

  return true_positives, false_positives, false_negatives, true_negatives


def modularity(adjacency, clusters):
  degrees = adjacency.sum(axis=0).A1
  m = degrees.sum()
  result = 0
  for cluster_id in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
    degrees_submatrix = degrees[cluster_indices]
    result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / m
  return result / m


def conductance(adjacency, clusters):  # TODO(tsitsulin): add docstring pylint: disable=missing-function-docstring
  inter = 0
  intra = 0
  cluster_idx = np.zeros(adjacency.shape[0], dtype=np.bool)
  for cluster_id in np.unique(clusters):
    cluster_idx[:] = 0
    cluster_idx[np.where(clusters == cluster_id)[0]] = 1
    adj_submatrix = adjacency[cluster_idx, :]
    inter += np.sum(adj_submatrix[:, cluster_idx])
    intra += np.sum(adj_submatrix[:, ~cluster_idx])
  return intra / (inter + intra)

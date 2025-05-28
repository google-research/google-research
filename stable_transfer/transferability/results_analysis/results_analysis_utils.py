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

"""Utils to analyse results with correlation scores and other measurements."""

import numpy as np
from scipy import stats


def get_correlation_scores(
    actual_performances, transferability_scores, metric='w-kendall'):
  """Return a correlation score, according to the metric."""

  assert metric in ['w-kendall', 'kendall', 'pearson']
  if metric == 'w-kendall':
    return stats.weightedtau(actual_performances, transferability_scores)[0]
  if metric == 'kendall':
    return stats.kendalltau(actual_performances, transferability_scores)[0]
  if metric == 'pearson':
    return stats.pearsonr(actual_performances, transferability_scores)[0]


def best_in_top_n(
    actual_performances, transferability_scores,
    transferability_scores_higher_is_better=True, n=1):
  """Returns whether the best model is within the top-n best transfer scores."""
  assert n > 0
  if n < 1:  # n is used as percentage
    n = int(len(actual_performances) * n)
  best_from_transfer = np.argsort(transferability_scores)
  if transferability_scores_higher_is_better:
    best_from_transfer = best_from_transfer[::-1]
  return np.argmax(actual_performances) in best_from_transfer[:n]


def relative_top_accuracy(
    actual_performances, transferability_scores,
    transferability_scores_higher_is_better=True):
  """Returns the accuracy ratio between the best model and the selected one."""
  if transferability_scores_higher_is_better:
    best_from_transfer = np.argmax(transferability_scores)
  else:
    best_from_transfer = np.argmin(transferability_scores)
  best_perf_from_transfer = actual_performances[best_from_transfer]
  best_perf = np.max(actual_performances)
  return best_perf_from_transfer / best_perf


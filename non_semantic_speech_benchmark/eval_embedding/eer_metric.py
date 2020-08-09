# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Equal Error Rate (EER) metric."""

from typing import Any, Iterable, Tuple

import numpy as np


def calculate_eer(scores, labels):
  """Returns the equal error rate for a binary classifier.

  EER is defined as the point on the DET curve where the false positive and
  false negative rates are equal.

  Args:
    scores: Regression scores for each data point. A score of 1 indicates a
      classification of label 1.
    labels: Ground truth labels for each data point.

  Returns:
    eer: The Equal Error Rate.
  """
  fpr, fnr = calculate_det_curve(scores, labels)
  min_diff_idx = np.argmin(np.abs(fpr - fnr))
  return np.mean((fpr[min_diff_idx], fnr[min_diff_idx]))


def calculate_det_curve(scores,
                        labels):
  """Calculates the false positive and negative rate at each score.

  The DET curve is related to the ROC curve, except it plots false positive rate
  against false negative rate.
  See https://en.wikipedia.org/wiki/Detection_error_tradeoff for a full
  description of the DET curve.

  Args:
    scores: Regression scores for each data point. A score of 1 indicates a
      classification of label 1. Should be in range (0, 1).
    labels: Ground truth labels for each data point.

  Returns:
    fpr, fnr
    All returned values are numpy arrays with the same length as scores.
    fpr: False positive rate at a given threshold value.
    fnr: False negative rate at a given threshold value.
  """

  scores = np.asarray(scores, dtype=float)
  labels = np.asarray(labels, dtype=float)
  indices = np.argsort(scores)
  labels = labels[indices]
  fnr = np.cumsum(labels) / np.sum(labels)
  fnr = np.insert(fnr, 0, 0)

  negative_labels = 1 - labels
  fpr = np.cumsum(negative_labels[::-1])[::-1]
  fpr /= np.sum(negative_labels)
  fpr = np.append(fpr, 0)

  return fpr, fnr

# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Evaluation functions for constrained KLD optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

DELTA = 1e-10


def cross_entropy_loss(features, labels, model_weights, threshold):
  """Returns cross entropy loss for deterministic model."""
  predictions = np.dot(features, model_weights) + threshold
  loss = np.mean(np.maximum(predictions, 0) - predictions * labels +
                 np.log(1 + np.exp(-np.abs(predictions))))
  return loss


def expected_error_rate(features, labels, models, probabilities):
  """Returns expected error for stochastic model."""
  er = 0.0
  for i in range(len(models)):
    predictions = np.dot(features, models[i][0]) + models[i][1]
    predicted_labels = 1.0 * (predictions > 0)
    er += probabilities[i] * np.mean(predicted_labels != labels)
  return er


def expected_group_klds(features, labels, groups, models, probabilities):
  """Returns expected KLD(p, hat{p}_G) for stochastic model, for G = 0, 1."""
  p = np.mean(labels > 0)  # Overall proportion of positives.
  p0 = 0  # Positive prediction rate for group 0.
  p1 = 0  # Positive prediction rate for group 1.

  for i in range(len(models)):
    predictions = np.dot(features, models[i][0]) + models[i][1]
    pos_preds0 = predictions[groups == 0] > 0
    if np.any(pos_preds0):
      p0 += probabilities[i] * np.mean(pos_preds0)
    pos_preds1 = predictions[groups == 1] > 0
    if np.any(pos_preds1):
      p1 += probabilities[i] * np.mean(pos_preds1)

  # Expected KLD for group 0 and 1.
  kld0 = p * np.log((p + DELTA) / (p0 + DELTA)) + (1 - p) * np.log(
      (1 - p + DELTA) / (1 - p0 + DELTA))
  kld1 = p * np.log((p + DELTA) / (p1 + DELTA)) + (1 - p) * np.log(
      (1 - p + DELTA) / (1 - p1 + DELTA))

  return kld0, kld1

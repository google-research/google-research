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

"""Evaluation functions for constrained F-measure optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


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


def expected_group_errors(features, labels, groups, models, probabilities):
  """Returns expected groups-specific errors for stochastic model."""
  er0 = expected_error_rate(features[groups == 0, :], labels[groups == 0],
                            models, probabilities)
  er1 = expected_error_rate(features[groups == 1, :], labels[groups == 1],
                            models, probabilities)
  return er0, er1


def expected_fmeasure(features, labels, models, probabilities):
  """Returns expected F-measure for stochastic model."""
  tp = 0.0
  fn = 0.0
  fp = 0.0
  for i in range(len(models)):
    predictions = np.dot(features, models[i][0]) + models[i][1]
    tp += probabilities[i] * np.sum(predictions[labels == 1] > 0)
    fn += probabilities[i] * np.sum(predictions[labels == 1] <= 0)
    fp += probabilities[i] * np.sum(predictions[labels == 0] > 0)
  return 2 * tp * 1. / (2 * tp + fp + fn)


def expected_group_fmeasures(features, labels, groups, models, probabilities):
  """Returns expected group-specific F-measures for stochastic model."""
  fmeasure0 = expected_fmeasure(features[groups == 0, :], labels[groups == 0],
                                models, probabilities)
  fmeasure1 = expected_fmeasure(features[groups == 1, :], labels[groups == 1],
                                models, probabilities)
  return fmeasure0, fmeasure1


def fmeasure_for_predictions(labels, predictions):
  """Returns F-measure for given labels and predictions."""
  tp = np.sum(predictions[labels == 1] > 0)
  fn = np.sum(predictions[labels == 1] <= 0)
  fp = np.sum(predictions[labels == 0] > 0)
  return 2 * tp * 1. / (2 * tp + fp + fn)

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

"""Custom acquisition functions for Makita with ModAL."""

import math

from modAL.models import base
from modAL.utils import data as mod_data
import numpy as np


def greedy(optimizer,
           features,
           n_instances = 1):
  """Takes the best instances by inference value sorted in ascending order.

  Args:
    optimizer: BaseLearner. Model to use to score instances.
    features: modALinput. Featurization of the instances to choose from.
    n_instances: Integer. The number of instances to select.

  Returns:
    Indices of the instances chosen.
  """
  return np.argpartition(optimizer.predict(features), n_instances)[:n_instances]


def half_sample(optimizer,
                features,
                n_instances = 1,
                alpha = 1.0):
  """Chooses the instances with the highest uncertainty.

  Args:
    optimizer: BaseLearner. Model to use to score instances.
    features: modALinput. Featurization of the instances to choose from.
    n_instances: Integer. The number of instances to select.
    alpha: Float. Half sampling weighting parameter. Higher weights will bias
      towards better inference values.

  Returns:
    Indices of the instances chosen.
  """
  predictions = optimizer.predict(features)

  half_differences = (
      (np.sign(predictions[::2]) * np.abs(predictions[::2])**alpha -
       np.sign(predictions[1::2]) * np.abs(predictions[1::2])**alpha) / 2).T

  num_splits = half_differences.shape[1]
  delta = half_differences / np.sqrt(num_splits)
  picked_indices = []
  big_h = np.eye(num_splits)

  selectable = np.ones(len(delta))

  for _ in range(n_instances):
    delta_transpose_delta = np.matmul(np.transpose(delta), delta)
    variances = np.sum(delta * delta, axis=1)

    candidate_score = selectable * np.sum(
        np.matmul(delta, delta_transpose_delta) * delta,
        axis=1) / (1 + variances)

    best_score = np.argmax(candidate_score)
    picked_indices.append(best_score)
    selectable[best_score] = 0

    best_delta = delta[best_score, :] / math.sqrt(variances[best_score])

    best_lambda = 1 - 1 / math.sqrt(1 + variances[best_score])
    update = np.eye(num_splits) - best_lambda * np.outer(best_delta, best_delta)

    delta = np.matmul(delta, update)

    big_h = np.matmul(big_h, update)

  return np.array(picked_indices)

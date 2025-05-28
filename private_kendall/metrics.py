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

"""Code for R^2 metric used in experiments.

Adapted from
https://github.com/google-research/google-research/blob/master/dp_regression/experiment.py.
The functions are simplified to work with a single model at a time and are
otherwise unchanged.
"""


import numpy as np


def r_squared_from_predictions(predictions, labels):
  """Returns R^2 value for given predictions on labels.

  Args:
    predictions: Vector of predictions.
    labels: Vector of labels.
  """
  sum_squared_residuals = np.sum(np.square(predictions - labels))
  total_sum_squares = np.sum(np.square(labels - np.mean(labels)))
  return 1 - (sum_squared_residuals / total_sum_squares)


def r_squared_from_model(model, features, labels):
  """Returns R^2 value for given model on features and labels.

  Args:
    model: Vector of coefficients for a regression model on features.
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
  """
  predictions = np.matmul(features, model)
  return r_squared_from_predictions(predictions.flatten, labels)

# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Various metric functions to evaluate locally interpretable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def fidelity_metrics(test_y_hat, test_y_fit, metric):
  """Computes fidelity metrics.

  Fidelity is defined as the differences between black-box model.
  predictions (test_y_hat) and locally interpretable model predictions
  (test_y_fit).
  Different metrics can be used such as mae, mse, rmse, r2 score.

  Args:
    test_y_hat: black-box model predictions
    test_y_fit: locally interpretable model predictions
    metric: metric to estimate the fidelity (mae, mse, rmse, r2 score)

  Returns:
    fidelity: fidelity result
  """

  # Mean Absolute Error
  if metric == 'mae':
    fidelity = metrics.mean_absolute_error(test_y_hat, test_y_fit)
  # Mean Squared Error
  elif metric == 'mse':
    fidelity = metrics.mean_squared_error(test_y_hat, test_y_fit)
  # Root Mean Squared Error
  elif metric == 'rmse':
    fidelity = np.sqrt(metrics.mean_squared_error(test_y_hat, test_y_fit))
  # R2 Score
  elif metric == 'r2':
    fidelity = metrics.r2_score(test_y_hat, test_y_fit)

  return fidelity


def overall_performance_metrics(test_y, test_y_fit, metric):
  """Computes overall performance metrics.

  Overall performance is defined as the differences between ground truth labels
  (test_y) and locally interpretable model predictions (test_y_fit).
  Different metrics can be used such as mae, mse, rmse, auc, accuracy.

  Args:
    test_y: ground truth labels
    test_y_fit: locally interpretable model predictions
    metric: metric to estimate the fidelity (mae, mse, rmse, auc, accuracy)

  Returns:
    overall_perf: overall prediction performance result
  """

  # Mean Absolute Error
  if metric == 'mae':
    overall_perf = metrics.mean_absolute_error(test_y, test_y_fit)
  # Mean Squared Error
  elif metric == 'mse':
    overall_perf = metrics.mean_squared_error(test_y, test_y_fit)
  # Root Mean Squared Error
  elif metric == 'rmse':
    overall_perf = np.sqrt(metrics.mean_squared_error(test_y, test_y_fit))
  # Area Under ROC Curve
  elif metric == 'auc':
    overall_perf = metrics.roc_auc_score(test_y, test_y_fit)
  # Accuracy
  elif metric == 'accuracy':
    overall_perf = metrics.accuracy_score(np.argmax(test_y, axis=1),
                                          np.argmax(test_y_fit, axis=1))

  return overall_perf


def awd_metric(test_c, test_coef):
  """Computes absolute weight difference (AWD) metric.

  Absolute weight difference (AWD) is defined as the differences between
  ground truth local dynamics (test_c) and estimated local dynamics (test_coef).

  Args:
    test_c: ground truth local dynamics
    test_coef: estimated local dynamics by locally interpretable model

  Returns:
    awd: absolute weight difference (AWD) performance result
  """

  # Only for non-zero coefficients
  test_c_nonzero = 1*(test_c > 0)

  # Sum of absolute weight difference
  awd_sum = np.sum(np.abs((test_c * test_c_nonzero) - \
                          (test_coef[:, 1:] * test_c_nonzero)))

  # Mean of absolute weight difference
  awd = awd_sum / np.sum(test_c_nonzero)

  return awd


def plot_result(x_test, data_name, test_y_hat, test_y_fit,
                test_c, test_coef, metric, criteria):
  """Plots various fidelity performances.

  This module plots fidelity or AWD results with respect to
  distance from the boundary where the local dynamics change (in percentile).

  Args:
    x_test: features in testing set
    data_name: Syn1, Syn2 or Syn3
    test_y_hat: black-box model predictions
    test_y_fit: locally interpretable model predictions
    test_c: ground truth local dynamics
    test_coef: estimated local dynamics by locally interpretable model
    metric: metrics for computing fidelity
    criteria: 'Fidelity' or 'AWD';
  """

  # Order of testing set index based on the
  # distance from the boundary where the local dynamics change
  if data_name == 'Syn1':
    test_idx = np.argsort(np.abs(x_test[:, 9]))
  elif data_name == 'Syn2':
    test_idx = np.argsort(np.abs(x_test[:, 9] + np.exp(x_test[:, 10]) - 1))
  elif data_name == 'Syn3':
    test_idx = np.argsort(np.abs(x_test[:, 9] + np.power(x_test[:, 10], 3)))

  # Determines x in terms of percentile
  division = 10
  x = [(1.0/(2*division)) + (1.0/division)*i for i in range(division)]

  # Initializes output
  output = np.zeros([division,])

  # Parameters
  thresh = (1.0/division)
  test_no = len(test_idx)

  # For each division (distance from the decision boundary)
  for i in range(division):
    # Samples in each division
    temp_idx = test_idx[int(test_no*thresh*i):int(test_no*thresh*(i+1))]

    if criteria == 'Fidelity':
      # Computes fidelity
      output[i] = fidelity_metrics(test_y_hat[temp_idx],
                                   test_y_fit[temp_idx],
                                   metric)
    elif criteria == 'AWD':
      # Computes AWD
      output[i] = awd_metric(test_c[temp_idx, :], test_coef[temp_idx, :])

  # Plots
  plt.figure(figsize=(6, 4))
  plt.plot(x, output, 'o-')
  plt.xlabel('Distance from the boundary (percentile)', size=16)
  if criteria == 'Fidelity':
    plt.ylabel(metric, size=16)
  elif criteria == 'AWD':
    plt.ylabel('AWD', size=16)
  plt.grid()
  plt.legend(['RL-LIM - ' + criteria], prop={'size': 16})
  plt.title(data_name + ' Dataset', size=16)
  plt.show()

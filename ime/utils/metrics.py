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

"""Implementation of different metrics used by to evaluate the model."""
import numpy as np


def RSE(pred, true):
  """Defines the root square error.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output:  A float representing root square error between predicted
              values and true ones
  """
  return np.sqrt(np.sum(
      (true - pred)**2)) / np.sqrt(np.sum((true - true.mean())**2))


def CORR(pred, true):
  """Defines the correlation.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: a float representing correlation between predicted
              values and true ones
  """

  u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
  d = np.sqrt(((true - true.mean(0))**2 * (pred - pred.mean(0))**2).sum(0))
  return (u / d).mean(-1)


def MAE(pred, true):
  """Defines mean absolute error.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: a float repsenting mean absolute error between predicted
    values and true ones
  """

  return np.mean(np.abs(pred - true))


def MSE(pred, true):
  """Defines the mean square error.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: a float representing mean square error between predicted values and
    true ones
  """

  return np.mean((pred - true)**2)


def RMSE(pred, true):
  """Defines the root mean square error.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: a float representing root mean square error between predicted values
    and true ones
  """
  return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
  """Defines the mean absolute percentage error.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: a float representing mean absolute percentage error  between
     predicted values and true ones
  """
  return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
  """Defines the mean squared percentage error.


  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: a float representing mean squared percentage error  between
    predicted values and true ones
  """
  return np.mean(np.square((pred - true) / true))


def Metric(pred, true):
  """Defines all regression metrics.

  Args:
    pred: A numpy array of arbitrary shape
    true: A numpy array of arbitrary shape

  Returns:
    output: floats for mean absoulte error, mean squared error, root mean
    square error, mean absolute percentage error and mean squared percentage
    error
  """

  mae = MAE(pred, true)
  mse = MSE(pred, true)
  rmse = RMSE(pred, true)
  mape = MAPE(pred, true)
  mspe = MSPE(pred, true)

  return mae, mse, rmse, mape, mspe

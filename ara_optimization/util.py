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

"""Helper functions for ARA evaluations."""

from collections.abc import Sequence
import numpy as np
import pandas as pd
from ara_optimization import metrics


def randomized_round(x):
  """Perform randomized rounding between the nearest two integers."""
  return np.floor(x + np.random.random(len(x)))


def randomized_snap_row(value, max_value,
                        contribution_cap):
  """Round input to bounded discrete range.

  The input is discretized to one of contribution_cap + 1 possible values that
  are evenly spaced between 0 and max_value, inclusive. That is, the inputs are
  rounded to the nearest integer multiple of (max_value / contribution_cap) that
  is between 0 and max_value.

  Args:
    value: the series of inputs to round
    max_value: the largest possible value after rounding
    contribution_cap: the granularity of the discretization

  Returns:
    the discretized series of values.
  """
  value = value.clip(lower=0, upper=max_value)
  value = value * (contribution_cap / max_value)
  value = randomized_round(value)
  return value * max_value / contribution_cap


def snap_contribution_bounds(bounds, total):
  """Scale and round contribution bounds to integers summing to total.

  Each range must be an integer that is at least 1.

  Args:
    bounds: desired relative lengths of the ranges
    total: the desired sum of lengths

  Returns:
    a list of strictly positive integers summing to total, with ratios as close
    as possible to the input bounds.
  """
  scaled_bounds = [x * total / sum(bounds) for x in bounds]
  rounded_bounds = np.rint(scaled_bounds).astype(int).tolist()

  while sum(rounded_bounds) < total:
    i = np.argmax([x - y for x, y in zip(scaled_bounds, rounded_bounds)])
    rounded_bounds[i] += 1
  while sum(rounded_bounds) > total:
    i = np.argmax([y - x for x, y in zip(scaled_bounds, rounded_bounds)])
    rounded_bounds[i] -= 1
  for i in range(len(rounded_bounds)):
    if rounded_bounds[i] == 0:
      j = np.argmax([y - x if y > 1 else -np.inf
                     for x, y in zip(scaled_bounds, rounded_bounds)])
      rounded_bounds[j] -= 1
      rounded_bounds[i] += 1
  return rounded_bounds


def rmsre_tau_error_metrics(df, value_columns,
                            mult_factor = 5.0
                            ):
  """Choose suitable tau parameters for RMSRE error metric for given dataset.

  The values are chosen by multiplying the median of each column by a
  multiplicative scaling factor.

  Args:
    df: the dataset
    value_columns: the columns for which parameters are needed
    mult_factor: the scaling factor to use for all columns

  Returns:
    A list of error metrics
  """
  error_metrics = []
  for col in value_columns:
    tau = mult_factor * df[col].median()
    error_metrics.append(metrics.RMSRETauMetric(tau))
  error_metrics.append(metrics.RMSRETauMetric(mult_factor))
  return error_metrics


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

"""Helper functions for manipulating DataFrames of trial measurements."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import pandas as pd


def get_index_values(df, level):
  """Gets index values from a DataFrame MultiIndex.

  Args:
    df: A DataFrame.
    level: The integer position of the level in the MultiIndex, or the name of
      the level.

  Returns:
    Vector of index values.
  """
  return df.index.get_level_values(level).values


def apply_step_budget(measurements, step_budget):
  """Filters measurements to those satisfying a budget in terms of steps."""
  return measurements[get_index_values(measurements, "step") <= step_budget]


def apply_example_budget(measurements, example_budget):
  """Filters measurements to those satisfying a budget in terms of examples."""
  batch_size = get_index_values(measurements, "batch_size")
  num_steps = get_index_values(measurements, "step")
  return measurements[batch_size * num_steps <= example_budget]


def compute_steps_to_result(measurements,
                            objective_col_name,
                            threshold,
                            maximize=False,
                            group_by="batch_size"):
  """Returns the measurements that reached the threshold in the fewest steps.

  Args:
    measurements: DataFrame of measurements indexed by at least (trial_id,
      step).
    objective_col_name: Column name of the objective metric.
    threshold: Target value of the objective metric.
    maximize: Whether the goal is to maximize (as opposed to minimize) the
      objective metric.
    group_by: Any valid first argument to DataFrame.groupby, for example a
      column name or list of column names. If empty or None, the operation is
      performed over the entire measurements table.

  Returns:
    A DataFrame with either 0 or 1 row per group, which is the measurement that
    reached the threshold in the fewest steps for that group (if any).
  """
  if group_by:
    # For each group, recursively call this function without group_by and
    # concatenate the results.
    return pd.concat([
        compute_steps_to_result(grouped_measurements, objective_col_name,
                                threshold, maximize, None)
        for _, grouped_measurements in measurements.groupby(group_by)
    ])

  # Sort measurements by step, breaking ties by trial id.
  measurements = measurements.sort_index(level=["step", "trial_id"])

  # Select all rows satisfying the threshold.
  comparator = operator.gt if maximize else operator.lt
  good_measurements = measurements[comparator(measurements[objective_col_name],
                                              threshold)]
  if good_measurements.empty:
    return good_measurements  # Return a table with no rows

  # Return the first measurement row satisfying the threshold.
  return good_measurements.iloc[[0]]


def get_best_measurement(measurements,
                         objective_col_name,
                         maximize=False,
                         group_by="batch_size"):
  """Returns the measurement corresponding to the best objective value.

  Args:
    measurements: DataFrame of measurements.
    objective_col_name: Column name of the objective metric.
    maximize: Whether the goal is to maximize (as opposed to minimize) the
      objective metric.
    group_by: Any valid first argument to DataFrame.groupby, for example a
      column name or list of column names. If empty or None, the operation is
      performed over the entire measurements table.

  Returns:
    A DataFrame with 1 row per group, which is the measurement corresponding to
    the best objective value for that group.
  """
  if group_by:
    # For each group, recursively call this function without group_by and
    # concatenate the results.
    return pd.concat([
        get_best_measurement(grouped_measurements, objective_col_name, maximize,
                             None)
        for _, grouped_measurements in measurements.groupby(group_by)
    ])

  # Sort measurements by objective and return the first one.
  measurements = measurements.sort_values(
      objective_col_name, ascending=not maximize)
  return measurements.iloc[[0]]

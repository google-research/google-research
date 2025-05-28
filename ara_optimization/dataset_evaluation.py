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

"""Error evaluation on a dataset.

Evaluation of the error of Attribution Reporting API (ARA) summary reports
mechanism with specified hyperparameters on a dataset.
"""

from collections.abc import Sequence
import pandas as pd
from scipy import stats
from ara_optimization import metrics
from ara_optimization import util

DEFAULT_TOTAL_CONTRIBUTION_BUDGET = 2**16


class ARADataset:
  """A dataset used for Attribution Reporting API (ARA) analysis."""

  def __init__(self,
               df,
               slice_columns,
               value_columns,
               count_column = "Count",
               total_contribution_budget =
               DEFAULT_TOTAL_CONTRIBUTION_BUDGET):
    """Initialize the dataset for error evaluation of the ARA mechanism.

    Args:
      df: the dataset
      slice_columns: columns specifying the groups to aggregate over
      value_columns: columns specifying the queries to estimate for each group
      count_column: a column that counts the number of input records
      total_contribution_budget: an upper bound on the sum of contributions.
    """
    self.df = df
    self.slice_columns = slice_columns
    self.value_columns = value_columns
    self.count_column = count_column
    self.total_contribution_budget = total_contribution_budget

  def _rescale_contribution_budgets(self, contribution_budgets,
                                    approximate_objective
                                    ):
    """Rescale contribution budgets and discretize if not approximating."""
    if approximate_objective:
      return [x * self.total_contribution_budget / sum(contribution_budgets)
              for x in contribution_budgets]
    else:
      return util.snap_contribution_bounds(contribution_budgets,
                                           self.total_contribution_budget)

  def _set_up_count_column(self, error_metric,
                           separate_count_estimation
                           ):
    """Handle count column as specified by separate_count_estimation flag."""
    if separate_count_estimation:
      value_columns = self.value_columns + [self.count_column]
    else:
      value_columns = self.value_columns
      error_metric = error_metric[:-1]
    return value_columns, list(error_metric)

  def _clip_and_group_data(self, value_columns,
                           clipping_thresholds,
                           contribution_budgets,
                           approximate_objective):
    """Clip and aggregate each column, discretizing if not approximating."""
    df = self.df.copy()
    aggdf_cols = list(self.value_columns + [self.count_column])
    for col, clip_thresh, budget in zip(value_columns,
                                        clipping_thresholds,
                                        contribution_budgets):
      if approximate_objective:
        df[col + "_Clipped"] = df[col].clip(upper=clip_thresh)
      else:
        df[col + "_Clipped"] = util.randomized_snap_row(
            df[col], clip_thresh, budget)
      aggdf_cols.append(col + "_Clipped")

    grouped = df.groupby(by=self.slice_columns)[aggdf_cols]
    aggdf = grouped.sum()

    return aggdf

  def _compute_column_error(self, df, total_privacy_budget,
                            column, privacy_budget,
                            contribution_budget, clip_threshold,
                            metric,
                            approximate_objective):
    """Compute the average error of the specified column."""
    if approximate_objective:
      variance = 2 * clip_threshold**2 / privacy_budget**2
    else:
      variance = (clip_threshold / contribution_budget)**2 * stats.dlaplace(
          total_privacy_budget / self.total_contribution_budget).var()
    df[column + "_Error"] = metric.error(
        df[column] - df[column+"_Clipped"], variance, df[column])

    return metric.avg_error(df[column+"_Error"])

  def _compute_inferred_count_error(self, df,
                                    total_privacy_budget,
                                    count_error_metric
                                    ):
    """Compute error of the count estimate inferred from the other columns."""
    variance = ((1 / self.total_contribution_budget)**2 * (
        len(self.value_columns) + 1) * stats.dlaplace(
            total_privacy_budget / self.total_contribution_budget).var())

    df["count_column_variance"] = variance
    df["count_column_bias"] = 0
    df[self.count_column + "_Error"] = count_error_metric.error(
        df["count_column_bias"], df["count_column_variance"],
        df[self.count_column])
    return count_error_metric.avg_error(df[self.count_column + "_Error"])

  def evaluate_objective(self,
                         error_metric,
                         total_privacy_budget,
                         contribution_budgets,
                         clipping_thresholds,
                         approximate_objective = False,
                         separate_count_estimation = False):
    """Compute error metric for specified hyperparameters.

    Note that if the flag approximate_objective is set to true and
    separate_count_estimation is set to false, then the estimated error will not
    incorporate the estimate of the count column, since this is invariant and is
    not useful for optimization.

    Args:
      error_metric: the error metric for each value column and the count column
      total_privacy_budget: the overall privacy budget used for all queries
      contribution_budgets: the portion of the domain used for each value column
      clipping_thresholds: the threshold used to truncate each value column
      approximate_objective: a flag determining whether to use a smooth
        approximation or the exact error accounting for discretization
      separate_count_estimation: a flag determining whether to treat count as a
        separate feature or to infer it from the other values estimated

    Returns:
      the overall error of estimating each value and the count
    """
    contribution_budgets = self._rescale_contribution_budgets(
        contribution_budgets, approximate_objective)
    privacy_budget_split = [
        x * total_privacy_budget / self.total_contribution_budget
        for x in contribution_budgets]
    count_error_metric = error_metric[-1]
    value_columns, error_metric = self._set_up_count_column(
        error_metric, separate_count_estimation)

    aggdf = self._clip_and_group_data(value_columns, clipping_thresholds,
                                      contribution_budgets,
                                      approximate_objective)

    errors = []
    for col, eps, contr, clip_thresh, metric in zip(value_columns,
                                                    privacy_budget_split,
                                                    contribution_budgets,
                                                    clipping_thresholds,
                                                    error_metric):
      errors.append(self._compute_column_error(aggdf, total_privacy_budget,
                                               col, eps, contr, clip_thresh,
                                               metric, approximate_objective))

    if not separate_count_estimation and not approximate_objective:
      errors.append(self._compute_inferred_count_error(aggdf,
                                                       total_privacy_budget,
                                                       count_error_metric))

    return error_metric[0].avg_error(errors)

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

"""Compute errors of optimized and baseline ARA algorithms on a dataset."""

from collections.abc import Sequence
import pandas as pd
from ara_optimization import dataset_evaluation
from ara_optimization import metrics
from ara_optimization import optimize_ara


def run_experiment(train_df, eval_df,
                   slice_columns, value_columns,
                   count_column, error_metric,
                   privacy_budgets = (1, 2, 4, 8, 16, 32, 64),
                   baseline_quantiles = (95, 99)
                   ):
  """Compute errors of optimized and baseline ARA algorithms on a dataset.

  Args:
    train_df: dataset that is analyzed for setting hyperparameters
    eval_df: separate dataset on which error is reported
    slice_columns: columns specifying the groups to aggregate over
    value_columns: columns specifying the queries to estimate for each group
    count_column: a column that counts the number of input records
    error_metric: the error metric for each value column and the count column
    privacy_budgets: a list of overall privacy budgets for evaluation
    baseline_quantiles: the quantiles to use for baseline clipping thresholds

  Returns:
    A dictionary storing the error of the optimized algorithm and each baseline
    for each total privacy budget. The keys of the dictionary are strings such
    as 'Optimization', '95' or '99' (where the second and third keys correspond
    to baselines), and the values are lists consisting of the error achieved by
    that algorithm for each privacy budget.
  """
  result = {}
  train_dataset = dataset_evaluation.ARADataset(train_df, slice_columns,
                                                value_columns, count_column)
  eval_dataset = dataset_evaluation.ARADataset(eval_df, slice_columns,
                                               value_columns, count_column)
  errors = []
  for budget in privacy_budgets:
    error = optimize_ara.optimize_ara(train_dataset, eval_dataset,
                                      error_metric[:-1], budget).error
    errors.append(error)
  result['Optimization'] = errors

  for quantile in baseline_quantiles:
    errors = []
    for budget in privacy_budgets:
      error = optimize_ara.baseline(train_dataset, eval_dataset,
                                    error_metric, budget, quantile / 100.).error
      errors.append(error)
    result[str(quantile)] = errors

  return result

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

"""Optimization of hyperparameters for attribution reporting API."""

import dataclasses
from absl import logging
import numpy as np
from scipy import optimize
from ara_optimization import dataset_evaluation
from ara_optimization import metrics


@dataclasses.dataclass(frozen=True)
class ARAObjective:
  """Objective function for optimizing ARA hyperparameters."""
  dataset: dataset_evaluation.ARADataset
  error_metrics: list[metrics.ErrorMetric]
  total_privacy_budget: float

  def initial_value(self):
    """Choose initial value of hyperparameters before optimization.

    The initial hyperparameters are an equal split of the contribution budget
    and clipping thresholds equal to the 99% quantile of each column. Note that
    the clipping thresholds are stored in a logarithmic representation.

    Returns:
      A suggested initial hyperparameter setting before optimization
    """
    num_features = len(self.dataset.value_columns)
    contribution_budgets = [1 / num_features] * num_features
    clipping_thresholds = [self.dataset.df[col].quantile(0.99) for
                           col in self.dataset.value_columns]
    return _combine_hyperparameters(contribution_budgets, clipping_thresholds)

  def evaluate_objective(self, hyperparameters):
    """A wrapper evaluating the error for a particular choice of parameters.

    This function may be passed to an optimizer to obtain good hyperparameters.

    Args:
      hyperparameters: a list of hyperparameters. The first half of the elements
        are the contribution budgets for each value column, and the second half
        of the elements are the base-2 logarithm of the clipping thresholds for
        each value column.

    Returns:
      The error of these hyperparameters on the dataset.
    """
    contrib_budgets, clip_thresholds = _split_hyperparameters(hyperparameters)
    return self.dataset.evaluate_objective(self.error_metrics,
                                           self.total_privacy_budget,
                                           contrib_budgets,
                                           clip_thresholds,
                                           approximate_objective=True)


@dataclasses.dataclass(frozen=True)
class ConfigurationError:
  """The error for a particular configuration of hyperparameters."""
  contribution_budgets: list[float]
  clipping_thresholds: list[float]
  error: float


def _split_hyperparameters(hyperparameters
                           ):
  num_features = len(hyperparameters) // 2
  contribution_budgets = hyperparameters[:num_features]
  clipping_thresholds = [2**y for y in hyperparameters[num_features:]]
  return contribution_budgets, clipping_thresholds


def _combine_hyperparameters(contribution_budgets,
                             clipping_thresholds):
  log_clipping_thresholds = np.log2(clipping_thresholds).tolist()
  return contribution_budgets + log_clipping_thresholds


def optimize_ara(train_dataset,
                 eval_dataset,
                 error_metrics,
                 total_privacy_budget
                 ):
  """Optimize contribution budgets and clipping thresholds for given dataset.

  Args:
    train_dataset: dataset that may be analyzed for setting hyperparameters
    eval_dataset: separate dataset on which error is reported
    error_metrics: the error metric for each value column and the count column
    total_privacy_budget: the overall privacy budget used for all queries

  Returns:
    A ConfigurationError object storing the optimized contribution budgets, the
    optimized clipping thresholds, and the error on eval_dataset under these
    optimized hyperparameters.
  """

  objective = ARAObjective(train_dataset, error_metrics, total_privacy_budget)
  initial_point = objective.initial_value()
  num_features = len(objective.dataset.value_columns)
  num_hyperparameters = 2 * num_features
  bounds = [[1e-10, np.inf]] * num_hyperparameters

  result = optimize.minimize(objective.evaluate_objective, x0=initial_point,
                             bounds=bounds)
  if not result.success:
    logging.warning("Optimization failed: %s", result.message)

  contribution_budgets, clipping_thresholds = _split_hyperparameters(result.x)

  error = eval_dataset.evaluate_objective(
      error_metrics, total_privacy_budget, contribution_budgets,
      clipping_thresholds, approximate_objective=False)

  return ConfigurationError(contribution_budgets, clipping_thresholds, error)


def baseline(train_dataset,
             eval_dataset,
             error_metrics,
             total_privacy_budget,
             quantile):
  """Compute error of baseline that uses fixed quantiles to set parameters.

  The baseline uses an equal split of the contribution budget among all queries,
  including the count query. The clipping thresholds are determined by
  considering a fixed quantile of the training dataset.

  Args:
    train_dataset: dataset that may be analyzed for setting hyperparameters
    eval_dataset: separate dataset on which error is reported
    error_metrics: the error metric for each value column and the count column
    total_privacy_budget: the overall privacy budget used for all queries
    quantile: the quantile in [0,1] used to set baseline clipping thresholds

  Returns:
    A ConfigurationError object storing the baseline contribution budgets, the
    baseline clipping thresholds, and the error on eval_dataset under these
    baseline hyperparameters.
  """
  num_features = len(train_dataset.value_columns) + 1
  contribution_budgets = [1 / num_features] * num_features
  quantiles = train_dataset.df.quantile(quantile, numeric_only=True)
  clipping_thresholds = [quantiles[col]
                         for col in train_dataset.value_columns] + [1]

  error = eval_dataset.evaluate_objective(error_metrics, total_privacy_budget,
                                          contribution_budgets,
                                          clipping_thresholds,
                                          approximate_objective=False,
                                          separate_count_estimation=True)

  return ConfigurationError(contribution_budgets, clipping_thresholds, error)

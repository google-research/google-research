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

"""Main methods of the paper (bootstrap, PPI, CRC)."""

import functools
import logging
from typing import Callable

import numpy as np

from high_confidence_ir_eval_using_genai import datasets


def excluded_by_upper_bound(true_values, bound_values):
  return np.sum(np.less(bound_values, true_values))


def excluded_by_lower_bound(true_values, bound_values):
  return np.sum(np.greater(bound_values, true_values))


def find_bounds(
    true_values,
    bound_func,
    coverage = 0.95,
    epsilon = 10**-5,
):
  """Computes bounds using binary search.

  Args:
    true_values: The true values.
    bound_func: A function that takes a threshold and returns the bound values.
    coverage: The desired coverage.
    epsilon: The tolerance for the binary search.

  Returns:
    A tuple of lower and upper degrees ([-1, 1]) that can be used to infer
    bounds.
  """
  num_values = true_values.shape[0]
  if coverage + coverage / num_values > 1:
    logging.warning(
        "Impossible coverage target: %f for sample size: %d, returning maximum"
        " bounds.",
        coverage,
        num_values,
    )
    return -1, 1
  assert (
      coverage + coverage / num_values <= 1
  ), f"Impossible coverage target: {coverage} for sample size: {num_values}"
  # correction to account for sample size
  coverage += coverage / num_values
  # to gaurantee coverage we convert to the number of items to exclude
  # working with percentages can result in small but significant errors
  num_exclude_total = np.floor((1.0 - coverage) * num_values)
  num_exclude_lower_bound = np.floor(num_exclude_total / 2.0)
  # we will do a binary search to find the lower and upper bound thresholds
  # these variables keep track of the high and low values for the bounds
  # (this could be a bit confusing since they are bounding the bound thresholds)
  upper_low = -1
  upper_high = 1
  lower_low = -1
  lower_high = 1
  # start binary search for the lower bound
  while lower_high - lower_low > epsilon:
    middle_value = lower_low + (lower_high - lower_low) / 2.0
    bound_values = bound_func(middle_value)
    cur_excluded = excluded_by_lower_bound(true_values, bound_values)
    # If more values are excluded than necessary by the middle value of lower
    # bound, then it is too high. We can thus update the maximum value with the
    # middle value, and continue searching.
    if cur_excluded > num_exclude_lower_bound:
      lower_high = middle_value
    # vice-versa, if too few are excluded, the middle value is too low
    else:
      lower_low = middle_value
  # to be sure the lower bound is has the coverage, we select the smaller value
  lower_bound = lower_low
  # it could be that our lower bound is excluding a bit more than expected
  # in that case the upper bound can exclude a bit less, therefore, we recompute
  # what the upper bound should exclude
  num_exclude_upper_bound = num_exclude_total - excluded_by_lower_bound(
      true_values, bound_func(lower_bound)
  )
  # start binary search for the upper bound, same as above but other direction
  while upper_high - upper_low > epsilon:
    middle_value = upper_low + (upper_high - upper_low) / 2.0
    bound_values = bound_func(middle_value)
    cur_excluded = excluded_by_upper_bound(true_values, bound_values)
    if cur_excluded <= num_exclude_upper_bound:
      upper_high = middle_value
    else:
      upper_low = middle_value
  upper_bound = upper_high
  return lower_bound, upper_bound


def bootstrap(
    ds_vali,
    ds_test,
    n = 10_000,
):
  """Computes the bootstrap confidence interval.

  Args:
    ds_vali: The validation set.
    ds_test: The test set.
    n: The number of bootstrap samples to use.

  Returns:
    A tuple of lower and upper bounds.
  """
  del ds_test  # Unused
  query_performances = ds_vali.per_query_true_dcg()
  mean_query_performance = np.mean(query_performances)
  samples = ds_vali.rng.choice(
      query_performances, size=(n, query_performances.shape[0]), replace=True
  )
  delta = np.mean(samples, axis=-1) - mean_query_performance
  delta = np.sort(delta, kind="stable")
  upper = mean_query_performance - delta[int(np.ceil(n * 0.025))]
  lower = (
      mean_query_performance
      - delta[np.minimum(int(np.floor(n * 0.975)), n - 1)]
  )
  return lower, upper


def ppi(
    ds_vali,
    ds_test,
):
  """Computes the PPI confidence interval.

  Args:
    ds_vali: The validation set.
    ds_test: The test set.

  Returns:
    A tuple of lower and upper bounds.
  """
  # Concatenate validation and test set.
  ds_concat = ds_vali.concatenate(ds_test)
  # Compute predicted dcg on both validation and test.
  pred_dcgs = ds_concat.per_query_pred_dcg()
  pred_mean = np.mean(pred_dcgs)
  pred_var = np.std(pred_dcgs, ddof=1) ** 2.0
  # Compute error correction only on validation set.
  errors = ds_vali.per_query_pred_dcg() - ds_vali.per_query_true_dcg()
  errors_mean = np.mean(errors)
  errors_var = np.std(errors, ddof=1) ** 2.0
  width = 1.96 * np.sqrt(
      errors_var / ds_vali.num_queries + pred_var / ds_concat.num_queries
  )
  # Compute the lower and upper bounds.
  lower = pred_mean - errors_mean - width
  upper = pred_mean - errors_mean + width
  return lower, upper


def crc_bootstrap(
    ds_vali,
    ds_test,
    n = 10_000,
):
  """Computes conformal risk control using bootstrap estimates.

  Args:
    ds_vali: The validation set.
    ds_test: The test set.
    n: The number of bootstrap samples to use.

  Returns:
    A tuple of lower and upper bounds.
  """
  lower_degree, upper_degree = find_bounds(
      ds_vali.per_query_bootstrap_true_dcg(n=n),
      functools.partial(ds_vali.per_query_bootstrap_pred_dcg, n=n),
  )
  return ds_test.pred_dcg(lower_degree), ds_test.pred_dcg(upper_degree)


def crc_per_query(
    ds_vali,
    ds_test,
):
  """Computes conformal risk control using per-query estimates.

  This method returns **per-query** bounds instead of the overall bounds.

  Args:
    ds_vali: The validation set.
    ds_test: The test set.

  Returns:
    A tuple of per-query lower and upper bounds.
  """
  lower_degree, upper_degree = find_bounds(
      ds_vali.per_query_true_dcg(),
      ds_vali.per_query_pred_dcg,
  )
  return (
      ds_test.per_query_pred_dcg(lower_degree),
      ds_test.per_query_pred_dcg(upper_degree),
  )

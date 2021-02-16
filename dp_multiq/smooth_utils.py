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

"""Smooth sensitivity utils used by both Smooth and CSmooth.

Section 3.1 from "Smooth Sensitivity and Sampling in Private Data Analysis" by
Nissim, Radkhodnikova, and Smith
(https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf) gives
details for compute_log_sensitivity and its helper functions.
"""

import numpy as np


def check_indices(n, lower_idx, upper_idx):
  """Raises an error for indices outside of the [-1, n] range.

  Args:
    n: Right endpoint for valid range.
    lower_idx: Lower bound for idx.
    upper_idx: Upper bound for idx.
  """
  if lower_idx < -1:
    raise ValueError("Index too small: lower_idx < -1.")
  if upper_idx > n:
    raise ValueError("Index too large: upper_idx > n.")


def update_log_smooth_sensitivity(lower_idx1, upper_idx1, lower_idx2,
                                  upper_idx2, data, data_low, data_high, t,
                                  log_smooth_sensitivity):
  """Updates, returns log smooth sensitivity by searching local sensitivities.

  Args:
    lower_idx1: Min value for index i.
    upper_idx1: Max value for index i.
    lower_idx2: Min value for index j.
    upper_idx2: Max value for index j.
    data: User data, sorted in increasing order and clipped to lie in the
      [data_low, data_high] range.
    data_low: Lower limit for differentially private quantile output value.
    data_high: Upper limit for differentially private quantile output value.
    t: Smooth sensitivity parameter.
    log_smooth_sensitivity: Current max log smooth sensitivity, as found by
      previous searches of other index ranges.

  Returns:
    The maximum distance-weighted local sensitivity at any pair of indices
    (i, j) where lower_idx1 <= i <= upper_idx1 and
    lower_idx2 <= j <= upper_idx2. The special indices -1 and n = len(data) are
    allowed and interpreted as indexing values data_low and data_high,
    respectively.
  """
  n = len(data)

  # Sanity checks.
  check_indices(n, lower_idx1, upper_idx1)
  check_indices(n, lower_idx2, upper_idx2)
  if upper_idx2 < lower_idx2:
    raise ValueError("Indices out of order: upper_idx2 < lower_idx2.")

  if upper_idx1 < lower_idx1:
    # Nothing to explore, return current log smooth sensitivity value.
    return log_smooth_sensitivity

  # Find the middle index and set i to this value.
  i = (lower_idx1 + upper_idx1) // 2

  # Scan the eligible indices j in the [lower_idx2, upper_idx2] range.
  js = np.arange(lower_idx2, upper_idx2 + 1)

  # Copy values from data at the indices indicated by js.  (For js that are n,
  # use max_value.)
  j_vals = np.empty(upper_idx2 + 1 - lower_idx2)
  js_lt_n_bool = js < n
  js_lt_n = js[js_lt_n_bool]
  j_vals[js_lt_n_bool] = data[js_lt_n]
  j_vals[np.logical_not(js_lt_n_bool)] = data_high

  # Compute database distances for all the (i, j) pairs.
  database_distances = np.maximum(js - (i + 1), 0)

  # Compute local sensitivities for all the (i, j) pairs.
  base_value = data_low if i == -1 else data[i]
  local_sensitivities = j_vals - base_value

  # Compute log smooth sensitivities:
  #   log(exp(-t*database_distances) * local_sensitivities).
  log_smooth_sensitivities = -t * database_distances + np.log(
      local_sensitivities)

  # Find the largest smooth sensitivity.
  max_smooth_sensitivity_index = np.argmax(log_smooth_sensitivities)
  current_max_log_smooth_sensitivity = log_smooth_sensitivities[
      max_smooth_sensitivity_index]
  max_smooth_sensitivity_index = js[max_smooth_sensitivity_index]

  # Update the input smooth sensitivity if we found a larger one.
  log_smooth_sensitivity = max(log_smooth_sensitivity,
                               current_max_log_smooth_sensitivity)

  # Check the remaining indices.  (All indices in the [lower_idx1, upper_idx1]
  # range that are not equal to the midpoint i value checked above.)
  log_smooth_sensitivity1 = update_log_smooth_sensitivity(
      i + 1, upper_idx1, max_smooth_sensitivity_index, upper_idx2, data,
      data_low, data_high, t, log_smooth_sensitivity)
  log_smooth_sensitivity2 = update_log_smooth_sensitivity(
      lower_idx1, i - 1, lower_idx2, max_smooth_sensitivity_index, data,
      data_low, data_high, t, log_smooth_sensitivity)
  return max(log_smooth_sensitivity1, log_smooth_sensitivity2)


def compute_log_smooth_sensitivity(data, data_low, data_high, true_quantile_idx,
                                   t):
  """Returns log(t-smooth sensitivity) for the given dataset and quantile.

  Args:
    data: User data, sorted in increasing order and clipped to lie in the
      [data_low, data_high] range.
    data_low: Lower limit for differentially private quantile output value.
    data_high: Upper limit for differentially private quantile output value.
    true_quantile_idx: Index into data at the desired quantile location.
    t: Smooth sensitivity parameter.
  """
  n = len(data)
  return update_log_smooth_sensitivity(-1, true_quantile_idx, true_quantile_idx,
                                       n, data, data_low, data_high, t,
                                       -np.inf)

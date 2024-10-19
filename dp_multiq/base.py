# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Basic methods for generating data and computing non-private quantiles."""

import math
import numpy as np


def quantile_index(n, quantile):
  """Returns index of the specified quantile in a sorted dataset of n elements.

  Args:
    n: Size of the sorted dataset.
    quantile: A value in [0, 1] indicating the desired quantile.

  Returns:
    Index of the specified quantile. If the quantile is between points at
    indices i and i+1, returns i.
  """
  return int(math.floor((n - 1) * quantile))


def quantiles(data, qs):
  """Returns quantile estimates for qs.

  Args:
    data: A dataset sorted in increasing order.
    qs: Increasing array of quantiles in [0,1].
  """
  return np.quantile(data, qs, interpolation='lower')


def misclassified_points_error(sorted_data, true_quantiles, est_quantiles):
  """Returns the average number of data points between true and est quantiles.

  Args:
    sorted_data: A dataset sorted in increasing order.
    true_quantiles: Ground truth quantiles.
    est_quantiles: Estimated quantiles.

  Returns:
    The number of data points strictly between true_quantiles[j] and
    est_quantiles[j], averaged over all j.
  """
  total_missed = 0
  num_quantiles = len(true_quantiles)
  for q_idx in range(num_quantiles):
    total_missed += np.abs(
        np.sum(sorted_data > true_quantiles[q_idx]) -
        np.sum(sorted_data > est_quantiles[q_idx]))
  return total_missed / num_quantiles


def distance_error(true_quantiles, est_quantiles):
  """Returns the mean distance between the true and estimated quantiles.

  Args:
    true_quantiles: Ground truth quantiles.
    est_quantiles: Estimated quantiles.
  """
  return np.mean(np.abs(true_quantiles - est_quantiles))


def gen_gaussian(num_samples, mean, stddev):
  """Returns num_samples iid Gaussian samples in increasing order.

  Args:
    num_samples: Number of samples to return.
    mean: Mean of Gaussian distribution to sample.
    stddev: Standard deviation of Gaussian distribution to sample.
  """
  return np.sort(np.random.normal(loc=mean, scale=stddev, size=num_samples))


def gen_uniform(num_samples, data_low, data_high):
  """Returns num_samples iid uniform samples in increasing order.

  Args:
    num_samples: Number of samples to return.
    data_low: Lower bound of uniform distribution to sample.
    data_high: Upper bound of uniform distribution to sample.
  """
  return np.sort(
      np.random.uniform(low=data_low, high=data_high, size=num_samples))

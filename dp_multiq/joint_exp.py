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

"""JointExp method for computing multiple dp quantiles."""

import numpy as np
from scipy import special

from dp_multiq import ind_exp


def compute_intervals(sorted_data, data_low, data_high):
  """Returns array of intervals of adjacent points.

  Args:
    sorted_data: Nondecreasing array of data points, all in the [data_low,
      data_high] range.
    data_low: Lower bound for data.
    data_high: Upper bound for data.

  Returns:
    An array of intervals of adjacent points from [data_low, data_high] in
    nondecreasing order. For example, if sorted_data = [0,1,1,2,3],
    data_low = 0, and data_high = 4, returns
    [[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4]].
  """
  return np.block([[data_low, sorted_data], [sorted_data,
                                             data_high]]).transpose()


def compute_log_phi(data_intervals, qs, eps, swap):
  """Computes multi-dimensional array log_phi.

  Args:
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    qs: Increasing array of quantiles in [0,1].
    eps: Privacy parameter epsilon.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.

  Returns:
    Array log_phi[a,b,c] where a and b index over intervals and c indexes over
    quantiles.
  """
  num_data_intervals = len(data_intervals)
  original_data_size = num_data_intervals - 1
  num_quantiles = len(qs)
  log_phi = np.zeros(
      [num_data_intervals, num_data_intervals, num_quantiles + 1])
  if swap:
    sensitivity = 2.0
  else:
    if len(qs) == 1:
      sensitivity = 2.0 * (1 - min(qs[0], 1 - qs[0]))
    else:
      sensitivity = 2.0 * (1 - np.min(qs[1:] - qs[:-1]))
  eps_term = -(eps / (2.0 * sensitivity))
  data_intervals_log_sizes = np.log(data_intervals[:, 1] - data_intervals[:, 0])
  # Compute log_phi[i1, i2, j] for i1, j = 0.
  diffs = np.abs(np.arange(num_data_intervals) - (qs[0] * original_data_size))
  log_phi[0, :, 0] = (eps_term * diffs) + data_intervals_log_sizes
  # Compute log_phi[i1, i2, j] for 1 <= j < num_quantiles.
  diffs = np.fromfunction(lambda i, j, k: np.maximum(j - i, 0),
                          (num_data_intervals, num_data_intervals, 1))
  diffs = np.abs(diffs - (qs[1:] - qs[:-1]) * original_data_size)
  diffs += np.tril(
      np.full((num_data_intervals, num_data_intervals), np.inf),
      k=-1)[:, :, np.newaxis]
  log_phi[:, :,
          1:-1] = (eps_term * diffs) + data_intervals_log_sizes[np.newaxis, :,
                                                                np.newaxis]
  # Compute log_phi[i1, i2, j] for j = num_quantiles.
  diffs = np.abs(
      np.arange(original_data_size, -1, -1) -
      ((1 - qs[-1]) * original_data_size))
  log_phi[:, -1, num_quantiles] = eps_term * diffs
  return log_phi


def logdotexp(a, b):
  """Multiplies probabilities using two matrices of log probabilities.

  Args:
    a: Matrix of log probabilities.
    b: Matrix of log probabilities.

  Returns:
    For a_ij = log(p_{a, ij}) and b_ij = log(p_{b, ij}) where p denotes a
    probability, returns matrix m where m_ij = log(p_{a, ij} * p_{b, ij})
    without leaving logspace for numerical stability.
  """
  max_a, max_b = np.max(a), np.max(b)
  exp_a, exp_b = a - max_a, b - max_b
  np.exp(exp_a, out=exp_a)
  np.exp(exp_b, out=exp_b)
  c = np.dot(exp_a, exp_b)
  np.log(c, out=c)
  c += max_a + max_b
  return c


def compute_log_alpha(data_intervals, log_phi, qs):
  """Computes three-dimensional array log_alpha.

  Args:
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    log_phi: Array from compute_log_phi.
    qs: Increasing array of quantiles in (0,1).

  Returns:
    Array log_alpha[a, b, c] where a and c index over quantiles and b represents
    interval repeats.
  """
  num_intervals = len(data_intervals)
  num_quantiles = len(qs)
  log_alpha = np.log(np.zeros([num_quantiles, num_intervals, num_quantiles]))
  log_alpha[0, :, 0] = log_phi[0, :, 0]
  for j in range(1, num_quantiles):
    log_hat_alpha = special.logsumexp(log_alpha[j - 1, :, :], axis=1)
    log_alpha[j, :, 0] = logdotexp(
        log_hat_alpha,
        log_phi[:, :, j] + np.diag(np.log(np.zeros(num_intervals))))
    log_alpha[j, :,
              1:j + 1] = np.diag(log_phi[:, :, j])[:, np.newaxis] + log_alpha[
                  j - 1, :, 0:j] - np.log(np.arange(1, j + 1) + 1)
  return log_alpha


def sample_joint_exp(log_alpha, data_intervals, log_phi, qs):
  """Given log_alpha and log_phi, samples final quantile estimates.

  Args:
    log_alpha: Array from compute_log_alpha.
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    log_phi: Array from compute_log_phi.
    qs: Increasing array of quantiles in (0,1).

  Returns:
    Array outputs where outputs[i] is the quantile estimate corresponding to
    quantile q[i].
  """
  num_intervals = len(data_intervals)
  num_quantiles = len(qs)
  outputs = np.zeros(num_quantiles)
  last_i = -1
  j = num_quantiles - 1
  repeats = 0
  while j >= 0:
    log_dist = log_alpha[j, :, :] + log_phi[:, last_i, j + 1][:, np.newaxis]
    # Prevent repeats unless it's the first round.
    if j < num_quantiles - 1:
      log_dist[last_i, :] = -np.inf
    i, k = np.unravel_index(
        ind_exp.racing_sample(log_dist), [num_intervals, num_quantiles])
    repeats += k
    k += 1
    for j2 in range(j - k + 1, j + 1):
      outputs[j2] = np.random.uniform(data_intervals[i, 0], data_intervals[i,
                                                                           1])
    j -= k
    last_i = i
  return outputs


def joint_exp(sorted_data, data_low, data_high, qs, eps, swap):
  """Computes eps-differentially private quantile estimates for qs.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower bound for data.
    data_high: Upper bound for data.
    qs: Increasing array of quantiles in (0,1).
    eps: Privacy parameter epsilon.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.

  Returns:
    Array o where o[i] is the quantile estimate corresponding to quantile q[i].
  """
  clipped_data = np.clip(sorted_data, data_low, data_high)
  data_intervals = compute_intervals(clipped_data, data_low, data_high)
  log_phi = compute_log_phi(data_intervals, qs, eps, swap)
  log_alpha = compute_log_alpha(data_intervals, log_phi, qs)
  return sample_joint_exp(log_alpha, data_intervals, log_phi, qs)

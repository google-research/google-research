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

"""JointExp method for computing multiple dp quantiles."""

import numpy as np
from numpy.fft import irfft
from numpy.fft import rfft
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
  """Computes two-dimensional array log_phi.

  Args:
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    qs: Increasing array of quantiles in [0,1].
    eps: Privacy parameter epsilon.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.

  Returns:
    Array log_phi where log_phi[i-i',j] = log(phi(i, i', j)).
  """
  num_data_intervals = len(data_intervals)
  original_data_size = num_data_intervals - 1
  if swap:
    sensitivity = 2.0
  else:
    if len(qs) == 1:
      sensitivity = 2.0 * (1 - min(qs[0], 1 - qs[0]))
    else:
      sensitivity = 2.0 * (1 - min(qs[0], np.min(qs[1:] - qs[:-1]), 1 - qs[-1]))
  eps_term = -(eps / (2.0 * sensitivity))
  gaps = np.arange(num_data_intervals)
  target_ns = (np.block([qs, 1]) - np.block([0, qs])) * original_data_size
  return eps_term * np.abs(gaps.reshape(-1, 1) - target_ns)


def logdotexp_toeplitz_lt(c, x):
  """Multiplies a log-space vector by a lower triangular Toeplitz matrix.

  Args:
    c: First column of the Toeplitz matrix (in log space).
    x: Vector to be multiplied (in log space).

  Returns:
    Let T denote the lower triangular Toeplitz matrix whose first column is
    given by exp(c); then the vector returned by this function is log(T *
    exp(x)). The multiplication is done using FFTs for efficiency, and care is
    taken to avoid overflow during exponentiation.
  """
  max_c, max_x = np.max(c), np.max(x)
  exp_c, exp_x = c - max_c, x - max_x
  np.exp(exp_c, out=exp_c)
  np.exp(exp_x, out=exp_x)
  n = len(x)
  # Choose the next power of two.
  p = np.power(2, np.ceil(np.log2(2 * n - 1))).astype(int)
  fft_exp_c = rfft(exp_c, n=p)
  fft_exp_x = rfft(exp_x, n=p)
  y = irfft(fft_exp_c * fft_exp_x)[:n]
  np.maximum(0, y, out=y)
  np.log(y, out=y)
  y += max_c + max_x
  return y


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
  data_intervals_log_sizes = np.log(data_intervals[:, 1] - data_intervals[:, 0])
  log_alpha = np.log(np.zeros([num_quantiles, num_intervals, num_quantiles]))
  log_alpha[0, :, 0] = log_phi[:, 0] + data_intervals_log_sizes
  # A handy mask for log_phi.
  disallow_repeat = np.zeros(num_intervals)
  disallow_repeat[0] = -np.inf
  for j in range(1, num_quantiles):
    log_hat_alpha = special.logsumexp(log_alpha[j - 1, :, :], axis=1)
    log_alpha[j, :, 0] = data_intervals_log_sizes + logdotexp_toeplitz_lt(
        log_phi[:, j] + disallow_repeat, log_hat_alpha)
    log_alpha[j, 0, 0] = -np.inf  # Correct possible numerical error.
    log_alpha[j, :, 1:j+1] = \
      (log_phi[0, j] + data_intervals_log_sizes)[:, np.newaxis] \
      + log_alpha[j-1, :, 0:j] - np.log(np.arange(1, j+1) + 1)
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
  last_i = num_intervals - 1
  j = num_quantiles - 1
  repeats = 0
  while j >= 0:
    log_dist = log_alpha[j, :last_i + 1, :] + log_phi[:last_i + 1,
                                                      j + 1][::-1, np.newaxis]
    # Prevent repeats unless it's the first round.
    if j < num_quantiles - 1:
      log_dist[last_i, :] = -np.inf
    i, k = np.unravel_index(
        ind_exp.racing_sample(log_dist), [last_i + 1, num_quantiles])
    repeats += k
    k += 1
    for j2 in range(j - k + 1, j + 1):
      outputs[j2] = np.random.uniform(data_intervals[i, 0], data_intervals[i,
                                                                           1])
    j -= k
    last_i = i
  return np.sort(outputs)


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

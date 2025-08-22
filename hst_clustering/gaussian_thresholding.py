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

"""Utility functions to implement the Gaussian thresholding mechanism."""

from typing import Callable, Tuple

import numpy as np
from scipy import optimize
from scipy import stats


def get_gaussian_thresholding_params(
    max_bucket_contribution,
    max_num_buckets_contributed,
    epsilon,
    delta,
):
  """Returns the parameters for the Gaussian thresholding mechanism.

  Args:
    max_bucket_contribution: The maximum number of contributions a user has per
      bucket. Also known as the L_infinity cap.
    max_num_buckets_contributed: Maximum number of buckets a user contributes in
      a histogram. Also known as the L_0 cap
    epsilon: Epsilon of DP
    delta: Delta of DP

  Returns:
    Standard deviation to add gaussian noise. Threshold for removing partitions
  """

  l2_sensitivity = max_bucket_contribution * np.sqrt(
      max_num_buckets_contributed
  )
  sigma = find_sigma(epsilon, delta / 2, l2_sensitivity)
  threshold = gaussian_threshold(
      max_bucket_contribution, max_num_buckets_contributed, sigma, delta / 2
  )
  return sigma, threshold


def gaussian_cdf(x):
  """Calculates the standard Gaussian CDF.

  Args:
    x: Input to Gaussian CDF

  Returns:
    P(X < x) for P a standard Gaussian measure.
  """
  return stats.norm(0, 1).cdf(x)


def b_plus(v, epsilon):
  """Auxiliary function from Algorithm 1 of the Analytic Gaussian Mechanism.

  Args:
    v: Parameter to evaluate function
    epsilon: Epsilon of differential privacy.

  Returns:
    B plus parameter. See B+ in
    https://arxiv.org/pdf/1805.06530.pdf (Algorithm 1).
  """
  return gaussian_cdf(np.sqrt(v * epsilon)) - np.exp(epsilon) * gaussian_cdf(
      -np.sqrt(epsilon * (v + 2))
  )


def b_minus(v, epsilon):
  """Auxiliary function from Algorithm 1 of the Analytic Gaussian Mechanism.

  Args:
    v: Parameter to evaluate function
    epsilon: Epsilon of differential privacy.

  Returns:
    B minus parameter. See B- in
    https://arxiv.org/pdf/1805.06530.pdf (Algorithm 1).
  """
  return gaussian_cdf(-np.sqrt(v * epsilon)) - np.exp(epsilon) * gaussian_cdf(
      -np.sqrt(epsilon * (v + 2))
  )


def compute_inverse(
    b_func, epsilon, delta
):
  """Computes the inverse of an auxiliary function.

  Finds v such that b_func(v, epsilon) = delta

  Args:
    b_func: Function that takes two arguments (v, epsilon).
    epsilon: Epsilon of DP, it is the second parameter of b_func
    delta: Delta of differential privacy

  Returns:
    The inverse of b_func with respect to its first parameter.
  """
  sol = optimize.root_scalar(
      lambda v: b_func(v, epsilon) - delta, bracket=(0, 5000)
  )
  return sol.root


def find_gaussian_multiplier(epsilon, delta):
  """Find alpha from the analytic guassian mechanism.

  Args:
    epsilon: Epsilon of DP
    delta: Delta of DP

  Returns:
    Alpha (See alpha in Algorithm 1 of https://arxiv.org/pdf/1805.06530.pdf)
  """
  delta_0 = gaussian_cdf(0) - np.exp(epsilon) * gaussian_cdf(
      -np.sqrt(2 * epsilon)
  )
  b_func = b_plus if delta_0 <= delta else b_minus
  v = compute_inverse(b_func, epsilon, delta)
  if delta_0 <= delta:
    return np.sqrt(1 + v / 2) - np.sqrt(v / 2)
  return np.sqrt(1 + v / 2) + np.sqrt(v / 2)


def find_sigma(epsilon, delta, sensitivity):
  """Finds standard deviation of the Gaussian mechanism.

  Args:
    epsilon: Epsilon of DP
    delta: Delta of DP
    sensitivity: L2 sensitivity of mechanism.

  Returns:
    the standard deviation.
  """
  alpha = find_gaussian_multiplier(epsilon, delta)
  return alpha * sensitivity / np.sqrt(2 * epsilon)


def gaussian_threshold(
    max_bucket_contribution,
    max_num_buckets_contributed,
    sigma,
    delta,
):
  """Returns the threshold of the Gaussian mechanism for unknown partitions.

  Args:
    max_bucket_contribution: The maximum number of contributions a user has per
      bucket. Also known as the L_ininifty cap.
    max_num_buckets_contributed: Maximum number of buckets a user contributes in
      a histogram. Also known as the L_0 cap
    sigma: Sigma of the Gaussian Mechanism
    delta: Delta introduced into the DP mechanism by thresholding.

  Returns:
    The threshold to discard buckets
  """
  quantile = 1 - 2 * delta / max_num_buckets_contributed
  tau = stats.norm(0, 1).ppf(quantile)
  return max_bucket_contribution + np.sqrt(2) * sigma * tau

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

"""Functions for the Gaussian mechanism."""

import functools
import numpy as np
from scipy import stats

from dp_l2 import utils


def gaussian_cdf_check(eps, l2_sensitivity, sigma):
  """Returns the difference of CDFs used in the analytic Gaussian mechanism.

  See Theorem 8 of https://arxiv.org/abs/1805.06530 for details.

  Args:
    eps: Float privacy parameter epsilon.
    l2_sensitivity: Float l2 sensitivity of the underlying statistic.
    sigma: Float standard deviation of the Gaussian mechanism.
  """
  first_term = stats.norm.cdf(
      l2_sensitivity / (2 * sigma) - eps * sigma / l2_sensitivity
  )
  second_term = stats.norm.cdf(
      -l2_sensitivity / (2 * sigma) - eps * sigma / l2_sensitivity
  )
  return first_term - np.exp(eps) * second_term


def get_gaussian_sigma(eps, delta, l2_sensitivity, tolerance=1e-3):
  """Returns the minimum Gaussian mechanism sigma satisfying (eps, delta)-DP.

  Uses binary search with the analytic Gaussian mechanism
  (https://arxiv.org/pdf/1805.06530) to lower bound sigma to tolerance.

  Args:
    eps: Float privacy parameter epsilon.
    delta: Float privacy parameter delta.
    l2_sensitivity: Float l2 sensitivity of the underlying statistic.
    tolerance: Float accuracy for computed sigma. Note that this errs on the
      side of being conservative.
  """
  # gaussian_cdf_check is decreasing in sigma, so we can use it with binary
  # search.
  binary_search_function = functools.partial(
      gaussian_cdf_check, eps, l2_sensitivity
  )
  return utils.binary_search(
      function=binary_search_function, threshold=delta, tolerance=tolerance
  )


def get_gaussian_samples(d, sigma, num_samples):
  """Returns samples of shape (num_samples, d) from the Gaussian mechanism.

  Args:
    d: Integer dimension.
    sigma: Float noise scale.
    num_samples: Integer number of samples to generate.
  """
  return np.random.randn(num_samples, d) * sigma


def get_gaussian_mean_squared_l2_error(d, sigma):
  """Returns the mean squared l_2 error of the specified Gaussian mechanism.

  See Lemma 4.4 in the paper for details.

  Args:
    d: Integer dimension.
    sigma: Float noise scale.
  """
  return d * sigma ** 2

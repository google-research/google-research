# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Functions for the Weighted Gaussian Mechanisms.

Implements the Weighted Gaussian Mechanism from
https://arxiv.org/pdf/2002.09745.pdf.
"""

import collections

import numpy as np
from scipy import stats

from dp_mm_domain import gaussian
from dp_mm_domain import utils


def get_weighted_hist(input_data):
  """Computes the weighted item histogram of the input data.

  Args:
    input_data: List of lists of elements, one list per user.

  Returns:
    Weighted histogram of the input data. Duplicate elements are removed and the
    l2 norm of the user's contribution vector is normalized to 1.
  """
  weighted_hist = collections.defaultdict(int)
  for elements in input_data:
    unique_elements = list(set(elements))
    for element in unique_elements:
      weighted_hist[element] += (len(unique_elements)) ** (-1 / 2)

  return weighted_hist


def get_weighted_gaussian_sigma_and_threshold(eps, delta, l0_bound):
  """Returns the threshold for the Weighted Gaussian mechanisms.

  Args:
    eps: Float privacy parameter epsilon.
    delta: Float privacy parameter delta.
    l0_bound: Integer bound on maximum number of elements a user can increment.

  Returns:
    A pair (sigma, threshold) containing the standard deviation and threshold
    for the Weighted Gaussian Mechanism, respectively.
  """

  sigma = gaussian.get_gaussian_sigma(eps, delta, 1.0)
  threshold = max([
      t ** (-1 / 2) + sigma * stats.norm.ppf((1 - delta) ** (1.0 / t))
      for t in range(1, l0_bound + 1)
  ])

  return sigma, threshold


def get_noisy_weighted_hist_above_threshold(input_data, sigma, threshold):
  """Computes the weighted histogram of the input data with Gaussian noise.

  Args:
    input_data: List of lists of elements, one list per user.
    sigma: Float standard deviation of the Gaussian noise.
    threshold: Float threshold for the histogram.

  Returns:
    Weighted histogram of the input data with Gaussian noise.
    Only items with noisy counts above the threshold are included in the
    histogram.
  """
  weighted_hist = get_weighted_hist(input_data)

  noisy_hist = {}
  for item, count in weighted_hist.items():
    noisy_count = count + np.random.normal(0, sigma)
    if noisy_count > threshold:
      noisy_hist[item] = noisy_count
  return noisy_hist


def weighted_gaussian_mechanism(input_data, l0_bound, eps, delta):
  """Computes (eps, delta)-DP set using the Weighted Gaussian Mechanism.

    Returns a set of elements that are in the union of the sets of
    elements held by users.

  Args:
    input_data: List of lists of elements, one list per user.
    l0_bound: Maximum number of elements a user can increment.
    eps: Value for privacy parameter epsilon.
    delta: Value for privacy parameter delta.

  Returns:
    (eps, delta)-DP histogram of the input data with Gaussian noise. Uses the
    parameters from Theorem B.2 in
    https://arxiv.org/pdf/2002.09745.pdf.
  """

  clipped_data = utils.l0_bound_users(input_data, l0_bound)

  # delta is halved to spend half of the budget on the threshold and half on the
  # noise.
  delta = delta / 2

  sigma, threshold = get_weighted_gaussian_sigma_and_threshold(
      eps, delta, l0_bound
  )

  return get_noisy_weighted_hist_above_threshold(clipped_data, sigma, threshold)

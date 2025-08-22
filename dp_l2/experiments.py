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

"""Functions for running privacy, error, and time experiments."""

import functools
import time

import numpy as np

from dp_l2 import gaussian
from dp_l2 import l2
from dp_l2 import laplace
from dp_l2 import utils


def errors_and_times_experiment(
    eps, delta, d_range, num_rs, num_e1_rs, num_samples
):
  """Runs error, time, and sigma experiments for all three mechanisms.

  For each d in d_range, for each of the Laplace, Gaussian, and l_2 mechanisms,
  we do the following:
    1) record the time taken to compute sigma;
    2) compute the mean squared l_2 error; and
    3) record the times for drawing num_samples samples.

  Args:
    eps: Float privacy parameter epsilon.
    delta: Float privacy parameter delta.
    d_range: Integer array of dimensions to evaluate.
    num_rs: Integer grid size for l_2 mechanism upper bound computation.
    num_e1_rs: Integer grid size for l_2 mechanism lower bound computation.
    num_samples: Integer number of samples to draw from mechanisms.

  Returns:
    Dictionaries of arrays sigma_times, mean_errors, and sample_times recording
    the results of the experiment for each d in d_range, where each dictionary
    is keyed by "Laplace", "Gaussian", and "l2"; and an array l2_sigmas
    recording the sigmas computed by the l_2 mechanism for each d in d_range.
  """
  sigma_times = {
      "Laplace": np.zeros(len(d_range)),
      "Gaussian": np.zeros(len(d_range)),
      "l2": np.zeros(len(d_range)),
  }
  mean_errors = {
      "Laplace": np.zeros(len(d_range)),
      "Gaussian": np.zeros(len(d_range)),
      "l2": np.zeros(len(d_range)),
  }
  sample_times = {
      "Laplace": np.zeros(len(d_range)),
      "Gaussian": np.zeros(len(d_range)),
      "l2": np.zeros(len(d_range)),
  }
  l2_sigmas = np.zeros(len(d_range))
  for d_idx, d in enumerate(d_range):
    start = time.time()
    laplace_sigma = laplace.get_laplace_sigma(d, eps, delta)
    end = time.time()
    sigma_times["Laplace"][d_idx] = end - start
    mean_errors["Laplace"][d_idx] = laplace.get_laplace_mean_squared_l2_error(
        d, laplace_sigma
    )
    start = time.time()
    laplace.get_laplace_samples(d, laplace_sigma, num_samples)
    end = time.time()
    sample_times["Laplace"][d_idx] = end - start
    start = time.time()
    gaussian_sigma = gaussian.get_gaussian_sigma(eps, delta, 1)
    end = time.time()
    sigma_times["Gaussian"][d_idx] = end - start
    mean_errors["Gaussian"][d_idx] = (
        gaussian.get_gaussian_mean_squared_l2_error(d, gaussian_sigma)
    )
    start = time.time()
    _ = gaussian.get_gaussian_samples(d, gaussian_sigma, num_samples)
    end = time.time()
    sample_times["Gaussian"][d_idx] = end - start
    start = time.time()
    l2_sigma = l2.get_l2_sigma(d, eps, delta, num_rs, num_e1_rs)
    l2_sigmas[d_idx] = l2_sigma
    end = time.time()
    sigma_times["l2"][d_idx] = end - start
    mean_errors["l2"][d_idx] = l2.get_l2_mean_squared_l2_error(d, l2_sigma)
    start = time.time()
    l2.get_l2_samples(d, l2_sigma, num_samples)
    end = time.time()
    sample_times["l2"][d_idx] = end - start
  return sigma_times, mean_errors, sample_times, l2_sigmas


def empirical_get_l2_plrv_difference(d, eps, num_samples, sigma):
  """Returns an empirical estimate of the l_2 mechanism's PLRV difference.

  We draw num_samples samples from the l_2 mechanisms centered at 0 and 1
  and empirically estimate both terms of the PLRV difference used in the
  necessary and sufficient condition for approximate DP. When this difference
  is smaller than delta, the mechanism is (eps, delta)-DP. This function returns
  the difference, and is used in the binary search for the l_2 mechanism's
  smallest sigma that yields a mechanism satisfying (eps, delta)-DP.

  See Section 4.1 in the paper for details.

  Args:
    d: Integer dimension of the mechanism.
    eps: Float privacy parameter epsilon.
    num_samples: Integer number of samples to draw from the mechanism.
    sigma: Float sigma of the mechanism.
  """
  e1 = np.zeros(d)
  e1[0] = 1
  samples_0 = l2.get_l2_samples(d, sigma, num_samples)
  samples_1 = l2.get_l2_samples(d, sigma, num_samples) + e1
  term_1_diffs = np.linalg.norm(samples_0 - e1, axis=1) - np.linalg.norm(
      samples_0, axis=1
  )
  term_1_plrv_estimate = np.sum(term_1_diffs / sigma >= eps)
  term_2_diffs = np.linalg.norm(samples_1 - e1, axis=1) - np.linalg.norm(
      samples_1, axis=1
  )
  term_2_plrv_estimate = np.sum(term_2_diffs / sigma >= eps)
  return term_1_plrv_estimate - np.exp(eps) * term_2_plrv_estimate


def empirical_get_l2_sigma(d, eps, delta, tolerance=1e-2):
  """Returns estimate of l_2 mechanism's smallest (eps, delta)-DP sigma.

  We use empirical_get_l2_plrv_difference to binary search for the smallest
  sigma that yields an l_2 mechanism satisfying (eps, delta)-DP.

  See Section 4.1 in the paper for details.

  Args:
    d: Integer dimension of the mechanism.
    eps: Float privacy parameter epsilon.
    delta: Float privacy parameter delta.
    tolerance: Float tolerance for binary search.
  """
  num_samples = int(1000 / delta)
  violations_threshold = 1000
  binary_search_function = functools.partial(
      empirical_get_l2_plrv_difference, d, eps, num_samples
  )
  return utils.binary_search(
      function=binary_search_function,
      threshold=violations_threshold,
      tolerance=tolerance,
  )

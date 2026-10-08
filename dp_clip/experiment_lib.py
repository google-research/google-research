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

"""Utilities for noise, histograms, and experiments."""

from collections.abc import Sequence
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dp_clip import dp_clip_lib


def get_gaussian_sigma(rho, l2_sensitivity):
  """Returns sigma such that resulting Gaussian mechanism is rho-CDP.

  Args:
    rho: Float CDP privacy parameter.
    l2_sensitivity: Float L2 sensitivity of the query.

  Returns:
    The float standard deviation of the Gaussian noise.
  """
  return l2_sensitivity / np.sqrt(2 * rho)


def get_cdp_gaussian_sample(
    d, rho, l2_sensitivity
):
  """Returns a sample from the specified d-dimensional Gaussian mechanism.

  Args:
    d: Integer dimension.
    rho: Float CDP privacy parameter.
    l2_sensitivity: Float L2 sensitivity of the query.

  Returns:
    A numpy array of samples of shape (d,).
  """
  return get_gaussian_sigma(rho, l2_sensitivity) * np.random.randn(d)


def get_true_histogram(data_matrix):
  """Computes the unclipped true histogram of a data matrix, i.e., its row sum.

  Args:
    data_matrix: A scipy.sparse matrix where rows represent users and columns
      represent features.

  Returns:
    A 1D numpy array representing the feature sums.
  """
  return data_matrix.sum(axis=0).A1


def l2_bound(data_matrix, bound):
  """Scales row vectors of data_matrix into a ball of radius bound.

  Args:
    data_matrix: A scipy.sparse matrix of user vectors.
    bound: Float maximum L2 norm.

  Returns:
    A scipy.sparse matrix where every row has L2 norm at most bound.
  """
  row_norms = spla.norm(data_matrix, axis=1)
  exceeds_bound = row_norms > bound

  scaling_factor = np.ones_like(row_norms)
  scaling_factor[exceeds_bound] = bound / row_norms[exceeds_bound]

  projected_matrix = sp.diags(scaling_factor) @ data_matrix
  return projected_matrix


def noisy_gaussian_histogram(
    bounded_histogram, rho, l2_sensitivity
):
  """Adds Gaussian noise to a bounded histogram.

  Args:
    bounded_histogram: A numpy array of histogram values.
    rho: Float privacy parameter rho.
    l2_sensitivity: Float L2 sensitivity.

  Returns:
    A numpy array representing the noisy histogram.
  """
  return bounded_histogram + get_cdp_gaussian_sample(
      len(bounded_histogram), rho, l2_sensitivity
  )


def get_vector_mse(vector_1, vector_2):
  """Computes the Mean Squared Error (L2 squared / d) between two vectors.

  Args:
    vector_1: A numpy array.
    vector_2: A numpy array.

  Returns:
    The mean squared error between the two vectors.
  """
  return np.mean((vector_1 - vector_2) ** 2)


def run_experiment(
    beta,
    goal_quantiles,
    clip_rho_range,
    histogram_rho,
    num_trials,
    data,
    nonnegative_data = True,
):
  """Runs experiments for different stochastic methods and bounds estimation.

  Args:
    beta: Float geometric base.
    goal_quantiles: Sequence of quantiles to evaluate.
    clip_rho_range: Numpy array of privacy budgets for clipping.
    histogram_rho: Float privacy parameter for the histogram.
    num_trials: Number of trials to run.
    data: A scipy.sparse matrix representing the dataset.
    nonnegative_data: Whether the dataset vectors are guaranteed to be
      nonnegative.

  Returns:
    A tuple containing:
      - errors: dict of MSE errors for each method.
      - estimates: dict of bounds estimated for each method.
      - mean_times: dict of average execution times.
  """
  _, d = data.shape
  true_histogram = get_true_histogram(data)
  norms = spla.norm(data, axis=1)

  opt_name = dp_clip_lib.PLOT_LABELS[dp_clip_lib.BoundEstimatorMethod.OPTIMAL]
  rc_name = dp_clip_lib.PLOT_LABELS[
      dp_clip_lib.BoundEstimatorMethod.OPTIMIZE_RC
  ]
  hly_name = dp_clip_lib.PLOT_LABELS[
      dp_clip_lib.BoundEstimatorMethod.HLY_UQ
  ]
  base_uq_name = dp_clip_lib.PLOT_LABELS[
      dp_clip_lib.BoundEstimatorMethod.UNBOUNDED_QUANTILE
  ]

  estimators = [
      (
          rc_name,
          lambda rho: dp_clip_lib.optimize_rc(
              data,
              beta,
              histogram_rho,
              rho,
              num_trials=num_trials,
              nonnegative_data=nonnegative_data,
          ),
      ),
      (
          hly_name,
          lambda rho: dp_clip_lib.hly_uq(
              norms,
              d,
              beta,
              histogram_rho,
              rho,
              num_trials=num_trials,
          ),
      ),
  ]

  for q in goal_quantiles:
    q_name = f"{base_uq_name} (q={q})"
    estimators.append((
        q_name,
        lambda rho, current_q=q: dp_clip_lib.unbounded_quantile(
            norms, beta, current_q, rho, num_trials=num_trials
        ),
    ))

  names = [opt_name] + [name for name, _ in estimators]

  errors = {name: np.zeros((len(clip_rho_range), num_trials)) for name in names}
  estimates = {
      name: np.zeros((len(clip_rho_range), num_trials)) for name in names
  }
  mean_times = {name: 0.0 for name in names}

  start_opt = time.perf_counter()
  optimal_clips = dp_clip_lib.compute_optimal_clipping_bound(
      data, histogram_rho, num_trials=num_trials
  )
  mean_times[opt_name] = time.perf_counter() - start_opt

  bounded_sum_cache: dict[float, np.ndarray] = {}

  def get_bounded_sum(c):
    key = float(c)
    if key not in bounded_sum_cache:
      bounded_sum_cache[key] = l2_bound(data, key).sum(axis=0).A1
    return bounded_sum_cache[key]

  for clip_rho_idx, clip_rho in enumerate(clip_rho_range):
    print(f"clip_rho: {clip_rho}")

    estimates[opt_name][clip_rho_idx, :] = optimal_clips
    for trial, c in enumerate(optimal_clips):
      bounded_histogram = get_bounded_sum(c)
      noisy_histogram = noisy_gaussian_histogram(
          bounded_histogram, histogram_rho, 2.0 * c
      )
      errors[opt_name][clip_rho_idx, trial] = get_vector_mse(
          true_histogram, noisy_histogram
      )

    for name, estimator_fn in estimators:
      start_time = time.perf_counter()
      l2_bounds = estimator_fn(clip_rho)
      mean_times[name] += time.perf_counter() - start_time

      estimates[name][clip_rho_idx, :] = l2_bounds

      for trial, c in enumerate(l2_bounds):
        bounded_histogram = get_bounded_sum(c)
        noisy_histogram = noisy_gaussian_histogram(
            bounded_histogram, histogram_rho, 2.0 * c
        )
        errors[name][clip_rho_idx, trial] = get_vector_mse(
            true_histogram, noisy_histogram
        )

  total_calls = num_trials * len(clip_rho_range)
  for name, _ in estimators:
    mean_times[name] /= total_calls

  return errors, estimates, mean_times

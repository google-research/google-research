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

"""Core DP clipping mechanism functions."""

import collections
import enum
import math

import numpy as np
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def compute_optimal_clipping_bound(
    data_vectors,
    rho,
    num_trials = 1,
):
  """Computes the L2 clipping bound that minimizes the MSE of a rho-zCDP sum.

  Args:
    data_vectors: A scipy.sparse CSR matrix of data vectors.
    rho: The sum privacy budget parameter.
    num_trials: The number of trial results to return (this is for consistency
      with other experiment code; as this function is deterministic, it returns
      num_trials copies of one evaluation).

  Returns:
    A numpy array of clipping bounds no smaller than 1.
  """
  _, d = data_vectors.shape

  norms = spla.norm(data_vectors, axis=1)
  true_sum = data_vectors.sum(axis=0).A1

  max_norm = np.max(norms) if norms.size > 0 else 0.0
  if max_norm == 0.0:
    return np.zeros(num_trials)

  def mse_objective(c):
    scales = np.where(norms > c, c / np.maximum(norms, 1e-12), 1.0)
    clipped_sum = (sp.diags(scales) @ data_vectors).sum(axis=0).A1
    squared_bias = np.sum((true_sum - clipped_sum) ** 2)
    variance = 2.0 * d * (c**2) / rho
    return squared_bias + variance

  result = scipy.optimize.minimize_scalar(
      mse_objective,
      bounds=(0, max_norm),
      method="bounded",
      options={"xatol": 0.01},
  )

  return np.full(num_trials, max(1.0, result.x))


def unbounded_quantile(
    data,
    beta,
    quantile,
    rho,
    num_trials = 1,
):
  """Computes a rho-zCDP estimate of a given quantile in data.

  Based on https://arxiv.org/abs/2305.01177 (see Section C).

  Args:
    data: An array or list of data values.
    beta: The geometric base for the logarithm binning.
    quantile: The target quantile to estimate (e.g., 0.9).
    rho: The privacy budget parameter.
    num_trials: The number of trial results to return.

  Returns:
    A numpy array of estimated quantiles.
  """
  scale = 1.0 / np.sqrt(2.0 * rho)

  counts = collections.defaultdict(int)
  for x in data:
    if x <= 1.0:
      i = 0
    else:
      i = int(math.log(x, beta) // 1)
    counts[i] += 1

  results = np.zeros(num_trials)
  for trial in range(num_trials):
    threshold = quantile * len(data) + np.random.exponential(scale=scale)
    current_sum = 0
    i = 0

    while True:
      current_sum += counts[i]
      i += 1
      if current_sum + np.random.exponential(scale=2.0 * scale) > threshold:
        break

    results[trial] = beta**i

  return results


def hly_uq(
    data,
    d,
    beta,
    rho_sum,
    rho_clip,
    num_trials = 1,
):
  """Computes a clipping bound using UnboundedQuantile at the HLY (2021) rank.

  Targets the (n - sqrt(2 * d / rho_sum))-th order statistic of the norms from
  Huang, Liang, & Yi (NeurIPS 2021), estimated via UnboundedQuantile.
  Based on https://arxiv.org/abs/2106.00463.

  Args:
    data: An array of vector L2 norms.
    d: Ambient dimension of the vectors.
    beta: The geometric base for the logarithm binning.
    rho_sum: The privacy budget parameter for the final vector sum.
    rho_clip: The privacy budget parameter for UQ's Above Threshold sequence.
    num_trials: The number of trial results to return.

  Returns:
    A numpy array of estimated clipping bounds.
  """
  quantile = max(0.0, 1.0 - math.sqrt(2.0 * d / rho_sum) / len(data))
  return unbounded_quantile(
      data, beta, quantile, rho_clip, num_trials=num_trials
  )


def precompute_rc(
    data,
    beta,
):
  """Pre-computes residual coherence values for candidate bounds (Algorithm 3).

  Args:
    data: A scipy.sparse CSR matrix of data vectors.
    beta: The geometric base for the sequence of candidate bounds.

  Returns:
    A 1D numpy array of residual coherence values for candidate bounds
    beta**0, beta**1, ..., beta**B, where B is the largest bin index with
    nonzero residual coherence.
  """
  n, d = data.shape
  norms = spla.norm(data, axis=1)
  unit_dirs = sp.diags(1.0 / np.maximum(norms, 1e-9)) @ data

  bins = np.full(n, -1, dtype=np.int32)
  valid = norms > 1.0
  if not np.any(valid):
    return np.array([])
  bins[valid] = np.floor(np.log(norms[valid]) / np.log(beta)).astype(np.int32)

  max_bin = int(np.max(bins))
  valid_indices = np.where(bins >= 0)[0]
  indicator = sp.csr_matrix(
      (np.ones(len(valid_indices)), (bins[valid_indices], valid_indices)),
      shape=(max_bin + 1, n),
  )
  dirs = indicator @ unit_dirs

  tail = np.zeros(d)
  rc_vals = np.zeros(max_bin + 1)
  for k in range(max_bin, -1, -1):
    tail += dirs[k].toarray()[0]
    rc_vals[k] = np.linalg.norm(tail)

  return rc_vals


def optimize_rc(
    data,
    beta,
    rho_sum,
    rho_clip,
    num_trials = 1,
    nonnegative_data = True,
):
  """Computes a clipping bound that privately optimizes residual coherence.

  Args:
    data: A scipy.sparse CSR matrix of data vectors.
    beta: The geometric base for the sequence of candidate bounds.
    rho_sum: The privacy budget parameter for the final vector sum.
    rho_clip: The privacy budget parameter for the Above Threshold sequence.
    num_trials: The number of trial results to return.
    nonnegative_data: Whether the input data vectors are guaranteed to be
      nonnegative. If False, rho_clip is divided by 4 to account for the 2x
      higher sensitivity of residual coherence on general vectors.

  Returns:
    A numpy array of computed clipping bounds.
  """
  _, d = data.shape
  rc_vals = precompute_rc(data, beta)
  num_bins = len(rc_vals)

  base_threshold = -math.sqrt(2.0 * d / rho_sum)
  effective_rho_clip = rho_clip if nonnegative_data else rho_clip / 4.0

  results = np.zeros(num_trials)
  for trial in range(num_trials):
    trial_threshold = base_threshold + np.random.exponential(
        scale=math.sqrt(2.0 / effective_rho_clip)
    )
    i = 0

    while True:
      rc = rc_vals[i] if i < num_bins else 0.0
      noise = np.random.exponential(scale=math.sqrt(8.0 / effective_rho_clip))

      if -rc + noise >= trial_threshold:
        break

      i += 1

    results[trial] = beta**i

  return results


class BoundEstimatorMethod(enum.Enum):
  """Methods for estimating the L2 clipping bound."""

  OPTIMAL = 1
  UNBOUNDED_QUANTILE = 2
  OPTIMIZE_RC = 3
  HLY_UQ = 4


ESTIMATOR_METHODS = {
    BoundEstimatorMethod.OPTIMAL: compute_optimal_clipping_bound,
    BoundEstimatorMethod.UNBOUNDED_QUANTILE: unbounded_quantile,
    BoundEstimatorMethod.OPTIMIZE_RC: optimize_rc,
    BoundEstimatorMethod.HLY_UQ: hly_uq,
}

PLOT_LABELS = {
    BoundEstimatorMethod.OPTIMAL: "Optimal",
    BoundEstimatorMethod.UNBOUNDED_QUANTILE: "UQ",
    BoundEstimatorMethod.OPTIMIZE_RC: "OptimizeRC",
    BoundEstimatorMethod.HLY_UQ: "HLY-UQ",
}

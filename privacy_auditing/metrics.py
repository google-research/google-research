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

"""Additional metrics not yet in jax_privacy.auditing."""

from collections.abc import Sequence
from concurrent import futures

import numpy as np
import scipy.stats

_norm_ppf = scipy.stats.norm.ppf


def _smoothed_rates(
    counts, total, shrinkage
):
  """Computes the smoothed rate using additive smoothing."""
  return (shrinkage + counts) / (2.0 * shrinkage + total)


def _max_mu_jackknife_one_side(
    target_scores_sorted,
    other_scores_sorted,
    min_count,
    shrinkage,
):
  """Computes max_mu leaving out each element of target_scores_sorted.

  Args:
      target_scores_sorted: The scores we are jackknifing (removing one at a
        time).
      other_scores_sorted: The scores that remain fixed.
      min_count: The minimum count threshold for valid statistics.
      shrinkage: Shrinkage factor for rate estimation.

  Returns:
      A numpy array of length len(target_scores_sorted) where the i-th element
      is the max_mu obtained when target_scores_sorted[i] is removed.
  """
  n_target = len(target_scores_sorted)
  n_other = len(other_scores_sorted)

  all_scores = np.union1d(target_scores_sorted, other_scores_sorted)
  global_fn_counts = np.searchsorted(
      target_scores_sorted, all_scores, side='left'
  )
  tn_counts = np.searchsorted(other_scores_sorted, all_scores, side='left')

  tnr = _smoothed_rates(tn_counts, n_other, shrinkage)

  def compute_mus(fn_counts):
    neg_counts = fn_counts + tn_counts
    pos_counts = (n_target - 1 + n_other) - neg_counts

    valid = (
        (np.minimum(pos_counts, neg_counts) >= min_count)
        & (fn_counts >= 0)
        & (fn_counts <= n_target - 1)
    )

    # Initialize with -inf so invalid thresholds are ignored in max.
    mus = np.full(fn_counts.shape, -np.inf)

    fnr = _smoothed_rates(fn_counts, n_target - 1, shrinkage)
    mus[valid] = np.abs(_norm_ppf(tnr[valid]) - _norm_ppf(fnr[valid]))

    return mus

  # When removed_index < global_fn_counts[k], fn_count is reduced by 1.
  scores_drop = compute_mus(global_fn_counts - 1)

  # When removed_index >= global_fn_counts[k], fn_count is unchanged.
  scores_keep = compute_mus(global_fn_counts)

  # The "split point" for threshold k is global_fn_counts[k].
  # If removed_index < split_point: use scores_drop[k]
  # If removed_index >= split_point: use scores_keep[k]
  # res[i] = max( max(scores_drop | split > i), max(scores_keep | split <= i) )

  max_drop_by_split = np.full(n_target + 1, -np.inf)
  max_keep_by_split = np.full(n_target + 1, -np.inf)

  np.maximum.at(max_drop_by_split, global_fn_counts, scores_drop)
  np.maximum.at(max_keep_by_split, global_fn_counts, scores_keep)

  # Suffix Max for Drop: max_{s > i}
  # Calculate cumulative max from right to left.
  suffix_max_drop = np.maximum.accumulate(max_drop_by_split[::-1])[::-1]

  # Prefix Max for Stay: max_{s <= i}
  # Calculate cumulative max from left to right.
  prefix_max_keep = np.maximum.accumulate(max_keep_by_split)

  right_part = suffix_max_drop[1:]  # aligns with i+1 (covers split > i).
  left_part = prefix_max_keep[:-1]  # aligns with i=0..N-1.

  return np.maximum(right_part, left_part)


def _compute_mus(
    in_scores_sorted,
    out_scores_sorted,
    min_count,
    shrinkage,
):
  """Helper to compute mu for all possible thresholds.

  Args:
    in_scores_sorted: Sorted array-like of held-in scores.
    out_scores_sorted: Sorted array-like of held-out scores.
    min_count: Minimum number of samples (TP+FP) required to consider a
      threshold valid.
    shrinkage: Shrinkage factor to prevent infinite estimates.

  Returns:
    A tuple of (mus, thresholds) where mus is an array of mu values for each
    threshold, and thresholds is the array of thresholds used.
  """
  n_in = len(in_scores_sorted)
  n_out = len(out_scores_sorted)

  # This should still be linear time if the scores are already sorted.
  all_scores = np.union1d(in_scores_sorted, out_scores_sorted)

  fn_counts = np.searchsorted(in_scores_sorted, all_scores, side='left')
  tn_counts = np.searchsorted(out_scores_sorted, all_scores, side='left')

  neg_counts = fn_counts + tn_counts
  pos_counts = n_in + n_out - neg_counts
  mask = np.minimum(pos_counts, neg_counts) >= min_count

  fnr = _smoothed_rates(fn_counts[mask], n_in, shrinkage)
  tnr = _smoothed_rates(tn_counts[mask], n_out, shrinkage)

  # Right tail bound is:
  #   ppf(tpr) - ppf(fpr)
  #     = ppf(1 - fnr) - ppf(1 - tnr)
  #     = -ppf(fnr) - (-ppf(tnr))
  #     = ppf(tnr) - ppf(fnr)
  # which equals the left tail bound, so we actually only need to compute one.
  # It will be tight on both sides for a typical successful symmetrical attack.
  # Take absolute value for if the scores are reversed (out scores are higher
  # than in scores).

  return np.abs(_norm_ppf(tnr) - _norm_ppf(fnr)), all_scores[mask]


def _safe_quantile(values, q):
  """Computes a quantile, correctly interpolating np.inf."""
  # Interpolating np.inf and np.inf results in np.nan, so we need to check for
  # NaNs explicitly and change interpolation method to 'nearest' in that case.
  result = np.quantile(values, q, method='linear')
  if np.isnan(result):
    result = np.quantile(values, q, method='inverted_cdf')
  return result


_safe_quantile_vectorized = np.vectorize(_safe_quantile, excluded={0, 'values'})


def audit_gdp(
    in_scores,
    out_scores,
    min_count = 50,
    alpha = 0.95,
    *,
    shrinkage = 0.5,
    bootstrap_samples = 1000,
    seed = None,
    central_with_interval = False,
    return_all_mus = False,
):
  """Calculates estimate of mu-GDP.

  By default returns the central estimate. If central_with_interval is True,
  returns the tuple (lower, central, upper) estimate. If return_all_mus is
  True, returns the tuple (mus, thresholds).

  Args:
    in_scores: Array-like of held-in scores.
    out_scores: Array-like of held-out scores.
    min_count: Minimum number of samples (TP+FP) required to consider a
      threshold valid.
    alpha: Confidence level (e.g., 0.95 for 95% confidence upper bound).
    shrinkage: Shrinkage factor to prevent infinite estimates. Set to 0.0 to
      disable shrinkage.
    bootstrap_samples: Number of bootstrap resamples.
    seed: Random seed for reproducibility.
    central_with_interval: Whether to return the central estimate with interval.
    return_all_mus: Whether to return all mu values for each threshold.

  Returns:
    The upper bound on mu, or the tuple (lower, central, upper) estimate if
    central_with_interval is True, or the tuple (mus, thresholds) if
    return_all_mus is True.
  """
  if min_count <= 0:
    raise ValueError(f'min_count must be positive, got {min_count}.')
  if not 0 < alpha < 1:
    raise ValueError(f'alpha must be in (0, 1), got {alpha}.')
  if bootstrap_samples <= 0:
    raise ValueError(
        f'bootstrap_samples must be positive, got {bootstrap_samples}.'
    )
  if central_with_interval and return_all_mus:
    raise ValueError(
        'central_with_interval and return_all_mus cannot both be True.'
    )

  # Sort scores for efficient computation of quantiles and jackknife.
  in_scores = np.sort(in_scores)
  out_scores = np.sort(out_scores)
  n_in = len(in_scores)
  n_out = len(out_scores)

  if min_count * 2 > (n_in + n_out):
    raise ValueError(
        'min_count must be less than half the total number of samples, got'
        f' min_count={min_count}, n_in={n_in}, n_out={n_out}.'
    )

  mus, thresholds = _compute_mus(in_scores, out_scores, min_count, shrinkage)
  mu_hat = np.max(mus, initial=0.0)
  if not central_with_interval:
    if return_all_mus:
      return mus, thresholds
    else:
      return mu_hat

  def bootstrap_task(inner_seed):
    # Poisson bootstrap resampling to maintain order.
    rng = np.random.default_rng(seed=inner_seed)
    in_weights = rng.poisson(1, n_in)
    out_weights = rng.poisson(1, n_out)
    in_sample = np.repeat(in_scores, in_weights)
    out_sample = np.repeat(out_scores, out_weights)
    mus, _ = _compute_mus(in_sample, out_sample, min_count, shrinkage)
    return np.max(mus, initial=0.0)

  rng = np.random.default_rng(seed=seed)
  seeds = rng.integers(np.iinfo(np.int64).max, size=bootstrap_samples)
  with futures.ThreadPoolExecutor() as executor:
    boot_results = list(executor.map(bootstrap_task, seeds))

  boot_results = np.array(boot_results)
  n_less = np.sum(boot_results < mu_hat)
  n_equal = np.sum(boot_results == mu_hat)
  # Split ties in half to avoid bias.
  prop_less = (n_less + n_equal / 2.0 + 1) / (bootstrap_samples + 2)
  z0 = scipy.stats.norm.ppf(prop_less)

  jackknife_results = np.concatenate([
      _max_mu_jackknife_one_side(in_scores, out_scores, min_count, shrinkage),
      _max_mu_jackknife_one_side(out_scores, in_scores, min_count, shrinkage),
  ])
  jackknife_mean = np.mean(jackknife_results)

  if np.isfinite(jackknife_mean):
    numer = np.sum((jackknife_mean - jackknife_results) ** 3)
    denom = 6.0 * (np.sum((jackknife_mean - jackknife_results) ** 2)) ** 1.5
    accel = numer / denom if denom > 0 else 0.0
  else:
    accel = 0.0

  q_low = (1 - alpha) / 2
  target_quantiles = np.array([q_low, 1 - q_low])

  z_alpha = scipy.stats.norm.ppf(target_quantiles)
  num = z0 + z_alpha
  den = 1.0 - accel * num
  alpha_adjusted = scipy.stats.norm.cdf(z0 + (num / den))
  mu_low, mu_high = _safe_quantile_vectorized(boot_results, alpha_adjusted)
  return mu_low, mu_hat, mu_high

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

"""Elevation validation metric formulas and computation.

Computes global pixel-level metrics from per-tile validation records:
- Bias (Mean Error): Pixel-weighted mean difference across tiles.
- Direct RMSE: Global pixel-level Root Mean Square Error.
- Tile-Debiased RMSE (DRMSE): Global pixel-level RMSE after removing
  per-tile mean bias offsets (within-tile variance pooling).
- Global Pixel-Level NMAD: Normalized Median Absolute Deviation computed
  across all pooled pixels globally using 1 cm resolution error distribution
  histograms (1.4826 * global_median(|diff - tile_median|)).
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_global_nmad_from_hist_series(
    hist_series, bin_width = 0.01, max_bins = 10000
):
  """Computes global pixel-level NMAD from per-tile sparse histogram JSON strings.

  Each entry in hist_series is a JSON string of [[bin_idx, count], ...] where
  bin_idx corresponds to [bin_idx * bin_width, (bin_idx + 1) * bin_width) meters
  of absolute deviation from tile median (|e_p - tile_median|).

  Args:
    hist_series: Pandas Series containing per-tile sparse histogram JSON
      strings.
    bin_width: Histogram bin width in meters (default 0.01m = 1cm).
    max_bins: Maximum bins to allocate (default 10000 bins = 0 to 100m).

  Returns:
    Global pixel-level NMAD = 1.4826 * global_median(|e_p - tile_median|).
  """
  global_hist = np.zeros(max_bins, dtype=np.float64)
  for val in hist_series.dropna():
    if not val or not isinstance(val, str) or val == '[]':
      continue
    pairs = json.loads(val)
    for item in pairs:
      b_idx = int(item[0])
      cnt = float(item[1])
      if not 0 <= b_idx < max_bins:
        raise ValueError(
            f'Histogram bin index {b_idx} outside [0, {max_bins}).'
        )
      if cnt > 0:
        global_hist[b_idx] += cnt

  total_px = float(np.sum(global_hist))
  if total_px <= 0:
    return np.nan

  half_px = 0.5 * total_px
  cum = np.cumsum(global_hist)
  k_star = int(np.searchsorted(cum, half_px))
  if k_star >= max_bins:
    k_star = max_bins - 1

  prev_cum = float(cum[k_star - 1]) if k_star > 0 else 0.0
  bin_cnt = float(global_hist[k_star])
  frac = (half_px - prev_cum) / bin_cnt if bin_cnt > 0 else 0.5
  global_mad = bin_width * (k_star + np.clip(frac, 0.0, 1.0))
  return float(1.4826 * global_mad)


def pooled_bias(
    sub_df, bias_col, cnt_col = None
):
  """Computes the pixel-weighted mean bias across tiles.

  Args:
    sub_df: Per-tile validation DataFrame.
    bias_col: Column name for tile mean bias.
    cnt_col: Column name for tile valid pixel count.

  Returns:
    Pixel-weighted mean bias in meters, or NaN if unavailable.
  """
  if bias_col not in sub_df.columns or cnt_col is None:
    return np.nan
  if cnt_col not in sub_df.columns:
    return np.nan
  valid = (
      sub_df[bias_col].notna() & sub_df[cnt_col].notna() & (sub_df[cnt_col] > 0)
  )
  if not valid.any():
    return np.nan
  b = sub_df.loc[valid, bias_col].to_numpy(dtype=float)
  n = sub_df.loc[valid, cnt_col].to_numpy(dtype=float)
  return float(np.sum(n * b) / np.sum(n))


def pooled_direct_rmse(
    sub_df, dir_col, cnt_col = None
):
  """Computes the global pixel-level RMSE across tiles.

  Args:
    sub_df: Per-tile validation DataFrame.
    dir_col: Column name for tile RMSE.
    cnt_col: Column name for tile valid pixel count.

  Returns:
    Pooled pixel-level RMSE in meters, or NaN if unavailable.
  """
  if dir_col not in sub_df.columns or cnt_col is None:
    return np.nan
  if cnt_col not in sub_df.columns:
    return np.nan
  valid = (
      sub_df[dir_col].notna() & sub_df[cnt_col].notna() & (sub_df[cnt_col] > 0)
  )
  if not valid.any():
    return np.nan
  r = sub_df.loc[valid, dir_col].to_numpy(dtype=float)
  n = sub_df.loc[valid, cnt_col].to_numpy(dtype=float)
  return float(np.sqrt(np.sum(n * (r**2)) / np.sum(n)))


def pooled_debiased_rmse(
    sub_df, deb_col, cnt_col = None
):
  """Computes global pixel-level tile-debiased RMSE (DRMSE) across tiles."""
  return pooled_direct_rmse(sub_df, deb_col, cnt_col)


def pooled_nmad(
    sub_df,
    src = '',
    sfx = '',
    hist_col = None,
):
  """Computes global pixel-level NMAD from per-tile deviation histograms.

  Args:
    sub_df: Per-tile validation DataFrame.
    src: Model source prefix (e.g. 'ge', 'cop').
    sfx: Subset suffix (e.g. '', '_ag', '_g').
    hist_col: Optional explicit histogram column name.

  Returns:
    Global pixel-level NMAD in meters, or NaN if unavailable.
  """
  candidates_hist = [
      hist_col,
      f'{src}_hist{sfx}',
      f'{src}_diff{sfx}_hist',
  ]
  active_hist_col = next(
      (c for c in candidates_hist if c and c in sub_df.columns), None
  )
  if active_hist_col is None:
    return np.nan
  return compute_global_nmad_from_hist_series(sub_df[active_hist_col])


def find_count_column(
    df, src, sfx = ''
):
  """Finds the pixel-count column for a model source and subset.

  Args:
    df: Per-tile validation DataFrame.
    src: Model source prefix.
    sfx: Subset suffix.

  Returns:
    Matching count column name, or None if not present.
  """
  candidates = [
      f'{src}_count{sfx}',
      f'{src}_diff{sfx}_count',
  ]
  return next((c for c in candidates if c in df.columns), None)


def compute_benchmark_metrics_from_df(
    df,
    min_slope_deg = 0.0,
    slope_col = 'slope_cop_mean',
):
  """Computes summary benchmark statistics from a validation results DataFrame.

  Applies pixel-level pooling across all tiles for:
  - Models: GEM-15 DSM (ge), Copernicus (cop), ALOS (alos), NASA (nasa),
            GEM-15 DTM (ge_dtm), FABDEM (fab).
  - Subsets: Overall (''), Above Ground ('_ag'), Bare Earth ('_g'),
             Flat ('_flat'), Steep ('_steep').

  Args:
    df: Results DataFrame containing tile-level metrics.
    min_slope_deg: Minimum terrain slope threshold to filter tiles.
    slope_col: Column name containing tile mean terrain slope.

  Returns:
    Dictionary of metric names mapped to their pooled scalar values.
  """
  if min_slope_deg > 0.0 and slope_col in df.columns:
    df = df[df[slope_col] > min_slope_deg].copy()

  subset_suffixes = ['', '_ag', '_g', '_flat', '_steep']
  subset_dfs = {sfx: df for sfx in subset_suffixes}
  metrics: Dict[str, float] = {'num_samples': float(len(subset_dfs['']))}

  def _sum_pixels(sub_df, sfx):
    """Total evaluated pixels for a subset."""
    cnt_col = next(
        (c for c in [f'n_px{sfx}', f'ge_count{sfx}'] if c in sub_df.columns),
        None,
    )
    if cnt_col is None:
      return np.nan
    valid = sub_df[cnt_col].dropna()
    return float(valid[valid > 0].sum())

  metrics['total_pixels'] = _sum_pixels(subset_dfs[''], '')
  metrics['total_pixels_ag'] = _sum_pixels(subset_dfs['_ag'], '_ag')
  metrics['total_pixels_g'] = _sum_pixels(subset_dfs['_g'], '_g')
  if '_flat' in subset_dfs:
    metrics['total_pixels_flat'] = _sum_pixels(subset_dfs['_flat'], '_flat')
  if '_steep' in subset_dfs:
    metrics['total_pixels_steep'] = _sum_pixels(subset_dfs['_steep'], '_steep')

  models = ['ge', 'cop', 'alos', 'nasa', 'ge_dtm', 'fab']

  for src in models:
    for sfx in subset_suffixes:
      sub_df = subset_dfs.get(sfx, df)
      cnt_col = find_count_column(sub_df, src, sfx)

      bias_col = f'{src}_bias{sfx}'
      dir_col = f'{src}_direct_rmse{sfx}'
      deb_col = f'{src}_debiased_rmse{sfx}'

      metrics[f'{src}_bias{sfx}'] = pooled_bias(sub_df, bias_col, cnt_col)
      metrics[f'{src}_direct_rmse{sfx}'] = pooled_direct_rmse(
          sub_df, dir_col, cnt_col
      )
      metrics[f'{src}_debiased_rmse{sfx}'] = pooled_debiased_rmse(
          sub_df, deb_col, cnt_col
      )
      metrics[f'{src}_nmad{sfx}'] = pooled_nmad(sub_df, src, sfx)

  return metrics


def compute_percentiles_metrics_from_df(
    df,
    bands = None,
):
  """Computes global pixel-level summary metrics for LiDAR percentile bands.

  Args:
    df: Results DataFrame containing tile-level percentile metrics.
    bands: List of band identifiers (defaults to ['dsm', 'dtm', 'p95', 'p50',
      'p5', 'std']).

  Returns:
    Dict of band -> dict of metrics ('bias', 'debiased_rmse', 'direct_rmse',
    'nmad').
  """
  if bands is None:
    bands = ['dsm', 'dtm', 'p95', 'p50', 'p5', 'std']

  results = {}
  for b in bands:
    mean_col = f'{b}_diff_mean'
    var_col = f'{b}_diff_variance'
    cnt_col = next(
        (c for c in [f'{b}_diff_count', f'{b}_count'] if c in df.columns),
        None,
    )

    if mean_col not in df.columns or var_col not in df.columns:
      results[b] = {
          'bias': np.nan,
          'debiased_rmse': np.nan,
          'direct_rmse': np.nan,
          'nmad': np.nan,
      }
      continue

    sub = df[df[mean_col].notna() & df[var_col].notna()].copy()
    m = sub[mean_col].to_numpy(dtype=float)
    v = np.maximum(0.0, sub[var_col].to_numpy(dtype=float))
    sub[f'{b}_bias'] = m
    sub[f'{b}_direct_rmse'] = np.sqrt(v + m**2)
    sub[f'{b}_debiased_rmse'] = np.sqrt(v)

    results[b] = {
        'bias': pooled_bias(sub, f'{b}_bias', cnt_col),
        'debiased_rmse': pooled_debiased_rmse(
            sub, f'{b}_debiased_rmse', cnt_col
        ),
        'direct_rmse': pooled_direct_rmse(sub, f'{b}_direct_rmse', cnt_col),
        'nmad': pooled_nmad(sub, src=b),
    }

  return results


def compute_point_metrics_from_df(
    df,
    models = None,
    ref_col = 'ref_elevation',
):
  """Computes exact validation metrics for point-based benchmarks (e.g. CORS).

  Calculates Mean Bias, Direct RMSE, Debiased RMSE, and Median NMAD
  per model against reference elevation.

  Args:
    df: DataFrame containing model predictions and reference elevations.
    models: List of model column names (default: ['ge', 'cop', 'alos', 'nasa']).
    ref_col: Reference elevation column name (default: 'ref_elevation').

  Returns:
    Dictionary with 'count' and {src}_bias, {src}_direct_rmse,
    {src}_debiased_rmse, {src}_nmad for each model.
  """
  if models is None:
    models = ['ge', 'cop', 'alos', 'nasa']

  res: Dict[str, Any] = {'count': len(df)}
  for src in models:
    if src not in df.columns or ref_col not in df.columns or df.empty:
      res[f'{src}_bias'] = np.nan
      res[f'{src}_direct_rmse'] = np.nan
      res[f'{src}_debiased_rmse'] = np.nan
      res[f'{src}_nmad'] = np.nan
      continue

    diff = (df[src] - df[ref_col]).dropna().to_numpy(dtype=float)
    if len(diff) > 0:
      bias = float(np.mean(diff))
      med = float(np.median(diff))
      var = float(np.var(diff))
      direct_rmse = float(np.sqrt(var + bias**2))
      debiased_rmse = float(np.sqrt(var))
      mad = float(np.median(np.abs(diff - med)))
      nmad = float(1.4826 * mad)
    else:
      bias, direct_rmse, debiased_rmse, nmad = np.nan, np.nan, np.nan, np.nan

    res[f'{src}_bias'] = bias
    res[f'{src}_direct_rmse'] = direct_rmse
    res[f'{src}_debiased_rmse'] = debiased_rmse
    res[f'{src}_nmad'] = nmad

  return res

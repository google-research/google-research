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

"""Probabilistic forecast verification and Bulletin 120 benchmarking metrics.

This module implements the core mathematical evaluation functions used during
Flowracle's expanding-window model selection and operational benchmarking:
  1. Pinball / quantile loss (`quantile_loss`).
  2. Empirical Continuous Ranked Probability Score (`compute_crps`, used in
     Flowracle's pre-2021 expanding-window validation objective).
  3. Operational California DWR Bulletin 120 April–July seasonal runoff
     benchmark alignment (`evaluate_against_b120`) across the 23 shared
     California snow-fed watersheds (15 tri-quantile `{0.1, 0.5, 0.9}` basins +
     8 median-only basins across February 1, March 1, and April 1 issuances).
  4. Composite expanding-window cross-validation objective
     (`compute_composite_cv_objective`) with the Bulletin 120 exceedance
     guardrail.
  5. Water-year hydrograph timing diagnostics
  (`compute_hydrograph_timing_metrics`):
     Center of Timing (`CT`), Spread of Timing (`SDoT`), and late-summer
     July–September runoff fraction (`f_summer`).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

from flowracle.data_pipeline import get_issuance_times
from flowracle.data_pipeline import resolve_dataset_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Expanding-window cross-validation objective weights
FOLD_VARIANCE_PENALTY = 0.1
B120_EXCEEDANCE_PENALTY = 0.3
CUMULATIVE_WEIGHT = 0.5
ISSUE_WEIGHTS: Dict[int, float] = {2: 0.25, 3: 0.35, 4: 0.40}
FOLD_EXCEEDANCE_WEIGHTS: Tuple[float, Ellipsis] = (0.2, 0.2, 0.6)

# True watershed mean elevations (m above sea level) computed from 1-arc-second
# SRTM DEM polygons across the 23 Bulletin 120 California snow-fed watersheds.
WATERSHED_MEAN_ELEVATION_M: Dict[str, float] = {
    "AMF": 1319.7,
    "CSN": 933.8,
    "EFC": 2328.0,
    "EWR": 2513.3,
    "FTO": 1528.8,
    "KGF": 2327.3,
    "KRI": 2196.7,
    "KWT": 1715.1,
    "MKM": 1619.0,
    "MRC": 1636.2,
    "MSS": 1342.9,
    "PSH": 1472.6,
    "SBB": 1203.8,
    "SBF": 2149.7,
    "SCC": 1209.9,
    "SDT": 1259.0,
    "SIS": 1394.7,
    "SNS": 1619.0,
    "TLG": 1789.1,
    "TRF": 2074.4,
    "WFC": 2458.8,
    "WWR": 2697.2,
    "YRS": 1380.4,
}

# Mid-month water-year day offsets measured from October 1 (Oct .. Sep).
WATER_YEAR_MONTH_MID_DAYS = np.array(
    [
        15.5,
        45.0,
        75.5,
        106.5,
        136.0,
        166.5,
        197.0,
        227.5,
        258.0,
        288.5,
        319.5,
        350.0,
    ],
    dtype=np.float64,
)


def quantile_loss(
    y_pred, y_true, quantile = 0.5
):
  """Computes mean pinball loss at level `quantile`, ignoring NaNs."""
  y_true_arr = np.asarray(y_true, dtype=np.float64)
  y_pred_arr = np.asarray(y_pred, dtype=np.float64)
  errors = y_true_arr - y_pred_arr
  return float(
      np.nanmean(np.maximum(quantile * errors, (quantile - 1.0) * errors))
  )


def compute_crps(
    ensemble_samples,
    obs,
    ensemble_axis = 0,
):
  """Computes exact empirical CRPS in `O(M log M)` time along `ensemble_axis`.

  Matches `swirl_dynamics.lib.metrics.probabilistic_forecast.crps`:
    `CRPS(F_M, y) = (1/M) sum_i |x_i - y| - (1/(2 M^2)) sum_{i,j} |x_i - x_j|`.

  Args:
    ensemble_samples: Ensemble forecast array with `M` members along
      `ensemble_axis`.
    obs: Observed target array matching `ensemble_samples` without
      `ensemble_axis`.
    ensemble_axis: Axis indexing ensemble members.

  Returns:
    Pointwise CRPS array of the same shape as `obs`.
  """
  ens = np.moveaxis(
      np.asarray(ensemble_samples, dtype=np.float64), ensemble_axis, 0
  )
  obs_arr = np.asarray(obs, dtype=np.float64)
  m_members = ens.shape[0]
  mae_term = np.mean(np.abs(ens - obs_arr[None, Ellipsis]), axis=0)
  if m_members <= 1:
    return mae_term

  sorted_ens = np.sort(ens, axis=0)
  ranks = 2.0 * np.arange(1, m_members + 1, dtype=np.float64) - m_members - 1.0
  reshape_dims = (m_members,) + (1,) * (sorted_ens.ndim - 1)
  spread_term = np.sum(ranks.reshape(reshape_dims) * sorted_ens, axis=0) / (
      m_members * m_members
  )
  return np.maximum(0.0, mae_term - spread_term)


def evaluate_against_b120(
    eval_list,
    val_y_pred,
    val_y_true,
    aux_ds,
    *,
    b120_da = None,
    data_dir = None,
    plot = True,
):
  """Benchmarks Flowracle April–July cumulative runoff against Bulletin 120.

  Aligns 1st-of-month February, March, and April issuances to the April–July
  snowmelt window (`amjj_indices = np.arange(4 - month, 8 - month)`):
    - Feb 1 issuance (`month=2`): leads `l in {3, 4, 5, 6}` -> indices `[2, 3,
    4, 5]`
    - Mar 1 issuance (`month=3`): leads `l in {2, 3, 4, 5}` -> indices `[1, 2,
    3, 4]`
    - Apr 1 issuance (`month=4`): leads `l in {1, 2, 3, 4}` -> indices `[0, 1,
    2, 3]`

  Args:
    eval_list: Evaluation batch sequence used to extract issuance dates.
    val_y_pred: Ensemble predictions of shape `(M, T, 6, S)` in Acre-Feet.
    val_y_true: Ground-truth monthly FNF of shape `(T, 6, S)` in Acre-Feet.
    aux_ds: Static metadata dataset carrying `station_id`.
    b120_da: Optional pre-loaded `forecast_fnf_april_july` DataArray.
    data_dir: Optional root data directory if opening `b120_consolidated.zarr`.
    plot: If True, renders a two-panel Feb/Mar/Apr quantile-loss comparison.

  Returns:
    Dictionary containing per-station/per-month quantile losses and summary
    curves across all 23 Bulletin 120 basins and the 15 tri-quantile basins.
  """
  result: Dict[str, Any] = {
      "per_station_per_month": [],
      "b120_available": False,
      "ours_23basin_losses": [],
      "b120_23basin_losses": [],
      "ours_15basin_3q_losses": [],
      "b120_15basin_3q_losses": [],
  }
  all_eval_times = get_issuance_times(iter(eval_list))
  if b120_da is None:
    b120_path = resolve_dataset_path(
        "b120-forecasts", "b120_consolidated.zarr", data_dir
    )
    b120_da = xr.open_zarr(b120_path)["forecast_fnf_april_july"]

  quantiles = [0.1, 0.5, 0.9]
  month_names = {2: "February", 3: "March", 4: "April"}
  model_station_ids = aux_ds.station_id.values.tolist()
  b120_station_ids = b120_da.station_id.values.tolist()
  shared_pairs = [
      (b_idx, model_station_ids.index(stn))
      for b_idx, stn in enumerate(b120_station_ids)
      if stn in model_station_ids
  ]
  b120_keep_idx = [p[0] for p in shared_pairs]
  b120_to_model_idx = [p[1] for p in shared_pairs]
  b120_da_aligned = b120_da.isel(station_id=b120_keep_idx)

  eval_dt_values = pd.to_datetime(all_eval_times).values
  for month in (2, 3, 4):
    month_mask = (pd.DatetimeIndex(all_eval_times).month == month) & np.isin(
        eval_dt_values, b120_da_aligned.time.values
    )
    if not month_mask.any():
      result["ours_23basin_losses"].append(0.0)
      result["b120_23basin_losses"].append(0.0)
      result["ours_15basin_3q_losses"].append(0.0)
      result["b120_15basin_3q_losses"].append(0.0)
      continue
    b120_month = b120_da_aligned.sel(time=eval_dt_values[month_mask])

    amjj_indices = np.arange(4 - month, 8 - month)
    ours_month = val_y_pred[
        :, month_mask, amjj_indices[0] : amjj_indices[-1] + 1, :
    ].sum(axis=2)[:, :, b120_to_model_idx]
    targets_amjj = val_y_true[
        month_mask, amjj_indices[0] : amjj_indices[-1] + 1, :
    ].sum(axis=1)[:, b120_to_model_idx]

    has_3q = ~b120_month.sel(quantile=0.1).isnull().all(dim="time")
    has_median = ~b120_month.sel(quantile=0.5).isnull().all(dim="time")
    q3_ids = np.where(has_3q)[0]
    all_b120_ids = np.where(has_median)[0]

    if len(q3_ids) > 0:
      ours_3q = (
          sum(
              quantile_loss(
                  np.quantile(ours_month[:, :, q3_ids], q, axis=0),
                  targets_amjj[:, q3_ids],
                  quantile=q,
              )
              for q in quantiles
          )
          / 3.0
      )
      b120_3q = (
          sum(
              quantile_loss(
                  b120_month.sel(quantile=q).isel(station_id=q3_ids).values,
                  targets_amjj[:, q3_ids],
                  quantile=q,
              )
              for q in quantiles
          )
          / 3.0
      )
    else:
      ours_3q = 0.0
      b120_3q = 0.0
    result["ours_15basin_3q_losses"].append(ours_3q)
    result["b120_15basin_3q_losses"].append(b120_3q)

    if len(all_b120_ids) > 0:
      per_station_ours = np.zeros(len(all_b120_ids))
      per_station_b120 = np.zeros(len(all_b120_ids))
      q3_set = set(q3_ids.tolist())
      for i, sid in enumerate(all_b120_ids):
        if sid in q3_set:
          per_station_ours[i] = np.mean([
              quantile_loss(
                  np.quantile(ours_month[:, :, sid], q, axis=0),
                  targets_amjj[:, sid],
                  quantile=q,
              )
              for q in quantiles
          ])
          per_station_b120[i] = np.mean([
              quantile_loss(
                  b120_month.sel(quantile=q).isel(station_id=sid).values,
                  targets_amjj[:, sid],
                  quantile=q,
              )
              for q in quantiles
          ])
        else:
          per_station_ours[i] = quantile_loss(
              np.quantile(ours_month[:, :, sid], 0.5, axis=0),
              targets_amjj[:, sid],
              quantile=0.5,
          )
          per_station_b120[i] = quantile_loss(
              b120_month.sel(quantile=0.5).isel(station_id=sid).values,
              targets_amjj[:, sid],
              quantile=0.5,
          )

      ours_all = float(np.mean(per_station_ours))
      b120_all = float(np.mean(per_station_b120))
      result["per_station_per_month"].append({
          "month": month,
          "ours": per_station_ours.copy(),
          "b120": per_station_b120.copy(),
      })
      result["b120_available"] = True
    else:
      ours_all, b120_all = 0.0, 0.0

    result["ours_23basin_losses"].append(ours_all)
    result["b120_23basin_losses"].append(b120_all)
    print(
        f"  [{month_names[month]}] 23-basin QLoss: Flowracle={ours_all:,.1f} AF"
        f" vs B120={b120_all:,.1f} AF"
    )

  if plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5.5))
    span_str = (
        f"({str(all_eval_times[0])[:10]} to {str(all_eval_times[-1])[:10]})"
    )
    axes[0].set_title(f"23-Basin Avg Quantile Loss\n{span_str}")
    axes[0].plot(
        range(3),
        result["b120_23basin_losses"],
        label="Bulletin 120",
        marker="o",
        color="crimson",
    )
    axes[0].plot(
        range(3),
        result["ours_23basin_losses"],
        label="Flowracle",
        marker="s",
        color="steelblue",
    )
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["Feb", "Mar", "Apr"])
    axes[0].set_ylabel("Quantile Loss (Acre-feet)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_title(
        f"15-Basin Tri-Quantile Loss (q=0.1, 0.5, 0.9)\n{span_str}"
    )
    axes[1].plot(
        range(3),
        result["b120_15basin_3q_losses"],
        label="Bulletin 120",
        marker="o",
        color="crimson",
    )
    axes[1].plot(
        range(3),
        result["ours_15basin_3q_losses"],
        label="Flowracle",
        marker="s",
        color="steelblue",
    )
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["Feb", "Mar", "Apr"])
    axes[1].set_ylabel("Quantile Loss (Acre-feet)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    del fig

  return result


def compute_fold_validation_ncrps(
    val_y_pred,
    val_y_true,
    issue_months,
):
  """Computes flow-balanced per-lead CRPS, spring AMJJ CRPS, and normalized CRPS."""
  import warnings  # pylint: disable=g-import-not-at-top

  issue_months_arr = np.asarray(issue_months)
  crps_per_fcast = compute_crps(val_y_pred, val_y_true, ensemble_axis=0)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    sample_magnitude = np.nanmean(val_y_true, axis=(-1, -2))

  valid_sample = ~np.isnan(sample_magnitude)
  threshold = (
      float(np.nanpercentile(sample_magnitude[valid_sample], 67))
      if valid_sample.any()
      else 0.0
  )
  high_mask = valid_sample & (sample_magnitude >= threshold)
  low_mask = valid_sample & (sample_magnitude < threshold)
  fallback_crps = (
      float(np.nanmean(crps_per_fcast))
      if not np.all(np.isnan(crps_per_fcast))
      else 0.0
  )
  crps_high = (
      float(np.nanmean(crps_per_fcast[high_mask]))
      if high_mask.any() and not np.all(np.isnan(crps_per_fcast[high_mask]))
      else fallback_crps
  )
  crps_low = (
      float(np.nanmean(crps_per_fcast[low_mask]))
      if low_mask.any() and not np.all(np.isnan(crps_per_fcast[low_mask]))
      else fallback_crps
  )
  per_lead_score = 0.5 * crps_high + 0.5 * crps_low

  amjj_crps_weighted, total_issue_weight = 0.0, 0.0
  for month, weight in ISSUE_WEIGHTS.items():
    month_mask = issue_months_arr == month
    if not month_mask.any():
      continue
    amjj_indices = np.arange(4 - month, 8 - month)
    amjj_indices = amjj_indices[(amjj_indices >= 0) & (amjj_indices < 6)]
    if len(amjj_indices) == 0:
      continue
    cumul_pred = val_y_pred[:, month_mask][:, :, amjj_indices, :].sum(axis=2)
    cumul_true = val_y_true[month_mask][:, amjj_indices, :].sum(axis=1)
    crps_cumul = compute_crps(cumul_pred, cumul_true, ensemble_axis=0)
    if np.all(np.isnan(crps_cumul)):
      continue
    cumul_score_month = float(np.nanmean(crps_cumul))
    amjj_crps_weighted += weight * cumul_score_month
    total_issue_weight += weight

  amjj_score = (
      amjj_crps_weighted / total_issue_weight
      if total_issue_weight > 0
      else per_lead_score
  )
  combined_score = (
      1.0 - CUMULATIVE_WEIGHT
  ) * per_lead_score + CUMULATIVE_WEIGHT * amjj_score
  mean_abs_true = (
      float(np.nanmean(np.abs(val_y_true)))
      if not np.all(np.isnan(val_y_true))
      else 0.0
  )
  ncrps = combined_score / max(mean_abs_true, 1e-8)
  return {
      "per_lead_crps": float(per_lead_score),
      "amjj_cumul_crps": float(amjj_score),
      "combined_crps": float(combined_score),
      "ncrps": float(ncrps),
  }


def compute_composite_cv_objective(
    validation_ncrps,
    b120_fold_exceedances,
):
  """Combines fold `nCRPS` and Bulletin 120 exceedance guardrail into scalar CV loss."""
  mean_ncrps = float(np.mean(validation_ncrps))
  std_ncrps = float(np.std(validation_ncrps))
  if len(b120_fold_exceedances) == len(FOLD_EXCEEDANCE_WEIGHTS):
    mean_b120_exceedance = float(
        np.sum(
            np.asarray(b120_fold_exceedances)
            * np.asarray(FOLD_EXCEEDANCE_WEIGHTS)
        )
    )
  elif b120_fold_exceedances:
    mean_b120_exceedance = float(np.mean(b120_fold_exceedances))
  else:
    mean_b120_exceedance = 0.0

  composite = (
      mean_ncrps
      + FOLD_VARIANCE_PENALTY * std_ncrps
      + B120_EXCEEDANCE_PENALTY * mean_b120_exceedance
  )
  return {
      "mean_ncrps": mean_ncrps,
      "std_ncrps": std_ncrps,
      "mean_b120_exceedance": mean_b120_exceedance,
      "composite_cv_objective": float(composite),
  }


def compute_hydrograph_timing_metrics(
    monthly_water_year_q,
):
  """Computes `CT` (days), `SDoT` (days), and Jul–Sep runoff fraction `%`.

  Args:
    monthly_water_year_q: Array of shape `(12, ...)` ordered from October (index
      0) through September (index 11) of a water year.

  Returns:
    Tuple `(ct, sdot, frac_summer_pct)` where:
      - `ct`: Center of Timing in days from October 1
      - `SDoT`: Standard Deviation of Timing (hydrograph width) in days
      - `frac_summer_pct`: July–September share of annual volume (`0..100` %)
  """
  q = np.asarray(monthly_water_year_q, dtype=np.float64)
  if q.shape[0] != 12:
    raise ValueError(
        "Expected leading water-year month dimension of size 12, got"
        f" {q.shape}."
    )
  q_clean = np.maximum(0.0, np.where(np.isnan(q), 0.0, q))
  total_q = q_clean.sum(axis=0) + 1e-6
  mid_days = WATER_YEAR_MONTH_MID_DAYS.reshape(
      (12,) + (1,) * (q_clean.ndim - 1)
  )
  ct = np.sum(mid_days * q_clean, axis=0) / total_q
  variance = (
      np.sum(((mid_days - ct[None, Ellipsis]) ** 2) * q_clean, axis=0) / total_q
  )
  sdot = np.sqrt(np.maximum(0.0, variance))
  frac_summer_pct = (q_clean[9:12].sum(axis=0) / total_q) * 100.0
  return ct, sdot, frac_summer_pct

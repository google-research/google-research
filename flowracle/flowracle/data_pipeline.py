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

"""Causal multi-source ingestion and Terrax spatio-temporal DataLoader setup.

This module implements the data ingestion, spatial subsetting, causal lag-lead
alignment, and expanding-window cross-validation slicing used by Flowracle.

Operational Causality Design
----------------------------
Forecasts are issued on the 1st of each calendar month `m` (e.g., February 1,
March 1, April 1) for monthly Full Natural Flow (FNF) across lead horizons
`l in {1, ..., 6}` months (target months `m, m+1, ..., m+5`):
  1. **Real-time monthly and daily sources** (PRISM 4 km climate, University of
     Arizona 4 km partial-monthly SWE omitting the final day of each month,
     USDA NRCS SNOTEL daily telemetry, California CDEC daily & end-of-month
     snow sensors, gridMET PDSI through day 25, and BoM MJO RMM index omitting
     the final day of each month) are available through month `m-1`
     (`lag_time = 0` relative to the end-of-month predictor origin, with a
     6-month trailing window `[-5, -4, -3, -2, -1, 0]`).
  2. **Lagged monthly sources** (CDEC monthly FNF gauge observations, which are
     finalized after month-end accounting) are restricted strictly to lags
     `[-5, -4, -3, -2, -1]` relative to month `m-1` (i.e., at least a 1-month
     reporting lag prior to forecast issuance).
  3. **Target stencil**: `target_fnf` spans `[+1, +2, +3, +4, +5, +6]` steps
     ahead of the month `m-1` predictor origin, corresponding to calendar
     months `m` through `m+5`. Six NaN placeholder months are appended beyond
     the end of the observational record so the 6-month target stencil can
     issue forecasts up to the final observed month without truncation.
"""

import copy
import dataclasses
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

# Extended California snow-fed domain bounding box for SNOTEL filtering.
CA_DOMAIN_LAT_MIN = 32.5
CA_DOMAIN_LAT_MAX = 42.0
CA_DOMAIN_LON_MIN = 360.0 - 124.0
CA_DOMAIN_LON_MAX = 360.0 - 115.0

# Trailing daily window (days) captured at the end of each calendar month.
EOM_LAG_DAYS = 5

# Number of placeholder months appended at the end of the aligned record so
# +6 month target stencils remain valid through the last observed issuance.
N_PAD_MONTHS = 6

# Uniform synthetic month step required by Terrax's fixed-timedelta stencil.
SYNTHETIC_MONTH_STEP = np.timedelta64(30, "D")
DEFAULT_BATCH_SIZE = 4

EXCLUDED_CLIM_VARS = frozenset({"SOI", "ONI"})


@dataclasses.dataclass
class AlignedFlowracleData:
  """Container holding aligned xarray datasets, static metadata, and Terrax specs."""

  dataloader: Any
  inputs_spec: Dict[str, Any]
  train_inputs_spec: Dict[str, Any]
  aux_ds: xr.Dataset
  fnf_ds: xr.Dataset
  real_time: np.ndarray
  synthetic_time: np.ndarray
  real_slice_to_synth: Callable[[Any], Any]


def resolve_dataset_path(
    slug,
    subpath = "",
    data_dir = None,
):
  """Resolves a Zarr dataset path inside `data_dir` or `FLOWRACLE_DATA_DIR`.

  Args:
    slug: Dataset subdirectory name produced by the preprocessing notebooks (for
      example, `'ca-monthly-fnf-extended'` or `'snotel-zarr'`).
    subpath: Relative path to the `.zarr` store within `slug`.
    data_dir: Optional explicit root directory overriding `FLOWRACLE_DATA_DIR`.

  Returns:
    Existing filesystem path to the requested Zarr store.

  Raises:
    FileNotFoundError: If the dataset cannot be located under any candidate
    path.
  """
  root = (
      data_dir
      if data_dir is not None
      else os.environ.get("FLOWRACLE_DATA_DIR", "")
  )
  candidates: List[str] = []
  if root:
    candidates.append(os.path.join(root, slug, subpath))
    if subpath:
      candidates.append(os.path.join(root, subpath))
  for candidate in candidates:
    cleaned = candidate.rstrip("/")
    if cleaned and os.path.exists(cleaned):
      return cleaned
  raise FileNotFoundError(
      f"Dataset '{slug}' (subpath='{subpath}') was not found. Set the "
      "FLOWRACLE_DATA_DIR environment variable (or pass `data_dir=...`) to the "
      "directory populated by `notebooks/data_preprocessing/`."
  )


def get_batch_predictors_ds(batch):
  """Extracts predictor xarray Datasets keyed by source name from a batch.

  Supports both live Terrax `coordax.Field` batches and pre-materialized
  dictionaries of `xarray.Dataset` objects (used in lightweight unit tests).

  Args:
    batch: Terrax batch dictionary or mapping `{source_name: xr.Dataset}`.

  Returns:
    Dictionary mapping each predictor source name to an `xr.Dataset` indexed by
    `('batch', 'lag_time', ...)`.
  """
  # Fast path for pre-converted xarray Dataset dictionaries (e.g. unit tests).
  predictor_entries = {k: v for k, v in batch.items() if k != "target_fnf"}
  if predictor_entries and all(
      isinstance(v, xr.Dataset) for v in predictor_entries.values()
  ):
    return predictor_entries

  import coordax as cx  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  from terrax.training import data_loading  # pylint: disable=g-import-not-at-top

  predictors = data_loading.sel_init_fields(batch)
  predictors_da = jax.tree.map(
      lambda f: f.to_xarray(), predictors, is_leaf=cx.is_field
  )
  result: Dict[str, xr.Dataset] = {}
  for source_name, var_dict in predictors_da.items():
    ds = xr.Dataset(var_dict)
    if "lead_time" in ds.coords:
      lts = np.array([
          lt if isinstance(lt, np.timedelta64) else lt.to_timedelta64()
          for lt in ds["lead_time"].values
      ])
      ds = ds.assign_coords(lead_time=lts)
    if "timedelta" in ds.dims or "timedelta" in ds.coords:
      ds = ds.rename({"timedelta": "lag_time"})
    result[source_name] = ds
  return result


def get_batch_target_ds(
    batch, field = None
):
  """Extracts target runoff fields from a batch as an `xarray.Dataset`.

  Args:
    batch: Terrax batch dictionary or dictionary containing an `xr.Dataset`
      under `'target_fnf'`.
    field: Optional substring filter for target variable names.

  Returns:
    Dataset with `lead_time` dimension and target variables named
    `'{source}_{var}'` (e.g. `'target_fnf_fnf'`).
  """
  if "target_fnf" in batch and isinstance(batch["target_fnf"], xr.Dataset):
    ds = batch["target_fnf"]
    if "fnf" in ds and "target_fnf_fnf" not in ds:
      ds = ds.rename({"fnf": "target_fnf_fnf"})
    if field:
      keep = [k for k in ds.data_vars if field in k]
      ds = ds[keep]
    return ds

  import coordax as cx  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  from terrax.training import data_loading  # pylint: disable=g-import-not-at-top

  targets = data_loading.sel_target_fields(batch)
  targets_da = jax.tree.map(
      lambda f: f.to_xarray(), targets, is_leaf=cx.is_field
  )
  flat_targets: Dict[str, xr.DataArray] = {}
  for k1, v1 in targets_da.items():
    for k2, v2 in v1.items():
      if not field or field in f"{k1}_{k2}":
        flat_targets[f"{k1}_{k2}"] = v2
  return xr.Dataset(flat_targets).rename({"timedelta": "lead_time"})


def consolidate_targets(
    data_iter,
    *,
    source = "target_fnf",
    field = "fnf",
    axis = 0,
):
  """Concatenates target arrays across an evaluation batch iterator."""
  target_list = [
      get_batch_target_ds(batch, field)[f"{source}_{field}"].values
      for batch in data_iter
  ]
  return np.concatenate(target_list, axis=axis)


def get_issuance_times(
    data_iter,
    *,
    source = "snotel_eom",
    time_var = "real_time",
    offset = pd.DateOffset(months=1),
    axis = 0,
):
  """Computes 1st-of-month forecast issuance timestamps for each batch item.

  Because `real_time` at `lag_time = 0` indexes the last fully observed month
  `m - 1`, adding `pd.DateOffset(months=1)` yields the operational issuance date
  on the 1st of month `m` (e.g., `2023-03-01` predictor origin -> `2023-04-01`
  forecast issuance).

  Args:
    data_iter: Iterator of Terrax or xarray batches.
    source: Predictor source carrying the `real_time` coordinate.
    time_var: Variable name storing calendar timestamps.
    offset: Offset from end-of-month predictor timestamp to forecast issuance.
    axis: Axis along which batch timestamps are concatenated.

  Returns:
    1D array of `pandas.Timestamp` forecast issuance dates.
  """
  times = []
  for batch in data_iter:
    pred_times = (
        get_batch_predictors_ds(batch)[source][time_var]
        .sel(lag_time=np.timedelta64(0, "D"), drop=True)
        .values
    )
    times.append([pd.Timestamp(t) + offset for t in pred_times])
  return np.concatenate(times, axis=axis)


def get_expanding_window_cv_slices(
    real_time_origin,
):
  """Returns the 3-fold expanding-window train and validation date slices.

  Fold structure:
    - Fold 1: Train up to Dec 2010, validate on 2011–2014 (drought onset).
    - Fold 2: Train up to Dec 2014, validate on 2015–2017 (2015 snow drought to
      2017 atmospheric river deluge).
    - Fold 3: Train up to Dec 2018, validate on 2019–2020 (strict pre-2021
      cutoff; produces the frozen checkpoint evaluated on 2021–2025).

  Predictor History Prepending Note:
    Each validation slice begins on June 1 of the training cutoff year
    (`2010-06-01`, `2014-06-01`, `2018-06-01`), prepending 6 extra months ahead
    of the validation period so Terrax's causal stencil can populate the full
    6-month trailing predictor window (`lags [-5, -4, -3, -2, -1, 0]`). Those
    initial 6 prepended months are never used as validation targets: Terrax
    begins emitting evaluation forecast origins only once all 6 trailing
    predictor months are available inside the slice (starting with predictor
    origin November -> December 1 issuance).

  Args:
    real_time_origin: Earliest aligned timestamp in the historical record
      (typically `1981-10-01`, the start of the UA 4 km SWE record).

  Returns:
    Tuple `(train_slices, val_slices)` of `(start, end)` `np.datetime64` pairs.
  """
  train_slices = [
      (real_time_origin, np.datetime64("2010-12-31")),
      (real_time_origin, np.datetime64("2014-12-31")),
      (real_time_origin, np.datetime64("2018-12-31")),
  ]
  val_slices = [
      (np.datetime64("2010-06-01"), np.datetime64("2014-12-31")),
      (np.datetime64("2014-06-01"), np.datetime64("2017-12-31")),
      (np.datetime64("2018-06-01"), np.datetime64("2020-12-31")),
  ]
  return train_slices, val_slices


def load_and_align_datasets(
    data_dir = None,
):
  """Loads all Zarr stores, filters SNOTEL stations, and aligns monthly series.

  Synthetic Equicadence Time Mapping Note:
    Calendar months have irregular durations (28, 29, 30, or 31 days), whereas
    Terrax's fixed-`TimeDelta` stencil slicer requires equicadence (strictly
    uniform) time steps to index multi-month predictor lags (`[-5, ..., 0]`)
    and target leads (`[+1, ..., +6]`) across data sources. For data-loading and
    batch-serving purposes, this function maps the aligned monthly `time`
    coordinate 1-to-1 onto a synthetic 30-day grid (`synthetic_time`, step
    `SYNTHETIC_MONTH_STEP = 30D`) while storing the true calendar timestamps in
    `real_time` (and calendar month numbers in `month`). At training and
    evaluation time, true calendar issuance dates are recovered directly from
    `real_time` via `get_issuance_times()`, so this equicadence mapping is
    purely for batch serving and does not affect feature values, causal
    alignment, or forecast results.

  Args:
    data_dir: Optional directory containing the preprocessed Zarr datasets;
      falls back to `FLOWRACLE_DATA_DIR`.

  Returns:
    Dictionary containing aligned predictor/target `xarray.Dataset` objects,
    `aux_ds`, `fnf_ds`, `real_time`, and `synthetic_time`.
  """
  fnf_ds = xr.open_zarr(
      resolve_dataset_path(
          "ca-monthly-fnf-extended", "monthly_fnf_consolidated.zarr", data_dir
      )
  )
  fnf_ds = fnf_ds.assign_coords(
      longitude=("station_id", fnf_ds.longitude.values % 360)
  )

  # Optional legacy macroclimate index store (retained in alignment when present
  # for exact RNG/buffer reproducibility; not read by Flowracle's feature map).
  clim_ind_ds: Optional[xr.Dataset] = None
  try:
    clim_ind_path = resolve_dataset_path(
        "noaa-psl-climate-indices", "macroclimate_indices.zarr", data_dir
    )
    clim_ind_ds = xr.open_zarr(clim_ind_path)
  except FileNotFoundError:
    clim_ind_ds = None

  prism_ds = xr.open_zarr(
      resolve_dataset_path(
          "prism-monthly-ca-region", "prism_monthly.zarr", data_dir
      )
  )
  prism_ds = prism_ds.assign_coords(x=prism_ds.x.values % 360).rename(
      {"x": "prism_longitude", "y": "prism_latitude"}
  )

  ca_sensors_ds = xr.open_zarr(
      resolve_dataset_path(
          "end-of-month-ca-snow-sensors",
          "endofmonth_snow_sensors.zarr",
          data_dir,
      )
  )
  ca_sensors_ds["time"] = (
      ca_sensors_ds["time"]
      .values.astype("datetime64[M]")
      .astype("datetime64[ns]")
  )

  ua_swe_ds = xr.open_zarr(
      resolve_dataset_path(
          "ua-swe-ca-region-operational-monthly-averages",
          "ua_swe_CA_region.zarr",
          data_dir,
      )
  )
  ua_swe_ds = ua_swe_ds.assign_coords(lon=ua_swe_ds.lon.values % 360).rename(
      {"lon": "ua_longitude", "lat": "ua_latitude"}
  )

  try:
    mjo_path = resolve_dataset_path(
        "mjo-rmm-index", "mjo_rmm_index.zarr/mjo_rmm_index.zarr", data_dir
    )
  except FileNotFoundError:
    mjo_path = resolve_dataset_path(
        "mjo-rmm-index", "mjo_rmm_index.zarr", data_dir
    )
  mjo_ds = xr.open_zarr(mjo_path)

  try:
    pdsi_path = resolve_dataset_path(
        "pdsi-gridmet-california",
        "pdsi_gridmet/pdsi_gridmet_monthly_CA.zarr",
        data_dir,
    )
  except FileNotFoundError:
    pdsi_path = resolve_dataset_path(
        "pdsi-gridmet-california", "pdsi_gridmet_monthly_CA.zarr", data_dir
    )
  pdsi_ds = xr.open_zarr(pdsi_path)
  pdsi_ds = pdsi_ds.assign_coords(lon=pdsi_ds.lon.values % 360).rename(
      {"lon": "pdsi_longitude", "lat": "pdsi_latitude"}
  )

  try:
    snotel_path = resolve_dataset_path("snotel-zarr", "snotel.zarr", data_dir)
  except FileNotFoundError:
    snotel_path = resolve_dataset_path("snotel-zarr", "", data_dir)
  snotel_ds = xr.open_zarr(snotel_path)

  daily_ca_sensors_ds = xr.open_zarr(
      resolve_dataset_path(
          "daily-ca-snow-sensors", "daily_snow_sensors.zarr", data_dir
      )
  )

  # 1. Subset SNOTEL network to the extended California snow-fed domain box.
  station_lat = snotel_ds["latitude"].values
  station_lon = snotel_ds["longitude"].values % 360.0
  if "longitude" in snotel_ds.data_vars:
    snotel_ds["longitude"] = (
        "index",
        station_lon.astype(snotel_ds["longitude"].dtype),
    )
  elif "longitude" in snotel_ds.coords:
    snotel_ds = snotel_ds.assign_coords(
        longitude=("index", station_lon.astype(snotel_ds["longitude"].dtype))
    )

  ca_domain_mask = (
      (station_lat >= CA_DOMAIN_LAT_MIN)
      & (station_lat <= CA_DOMAIN_LAT_MAX)
      & (station_lon >= CA_DOMAIN_LON_MIN)
      & (station_lon <= CA_DOMAIN_LON_MAX)
  )
  snotel_ca_ds = snotel_ds.isel(index=ca_domain_mask)
  snotel_ca_ds = snotel_ca_ds.assign_coords(
      time=snotel_ca_ds.time.dt.floor("D")
  )

  temporal_vars = [
      v for v in snotel_ca_ds.data_vars if "time" in snotel_ca_ds[v].dims
  ]
  static_vars = [
      v for v in snotel_ca_ds.variables if "time" not in snotel_ca_ds[v].dims
  ]
  snotel_ca_temporal = snotel_ca_ds[temporal_vars].load()
  snotel_ca_static = snotel_ca_ds[static_vars].rename(
      {v: f"snotel_{v}" for v in static_vars}
  )

  snotel_mean = snotel_ca_temporal.resample(time="1MS").mean()
  snotel_eom = (
      snotel_ca_temporal.rolling(time=EOM_LAG_DAYS)
      .construct("eom_lag")
      .resample(time="1MS")
      .last()
  )
  daily_ca_sensors = (
      daily_ca_sensors_ds.rolling(time=EOM_LAG_DAYS)
      .construct("eom_lag")
      .resample(time="1MS")
      .last()
  )

  fnf_subset = fnf_ds.get(["fnf"]).load()
  ca_static_vars = [
      v for v in ca_sensors_ds.variables if "time" not in ca_sensors_ds[v].dims
  ]
  ca_sensors_static = ca_sensors_ds[ca_static_vars].rename(
      {v: f"ca_sensor_{v}" for v in ca_static_vars}
  )
  aux_ds = xr.merge([
      fnf_ds.get(["longitude", "latitude", "elevation"]),
      snotel_ca_static,
      ca_sensors_static,
  ]).reset_coords()

  # 2. Inner-join all monthly series onto their shared historical span.
  to_align: List[xr.Dataset] = [
      fnf_subset,
      snotel_mean,
      snotel_eom,
      prism_ds,
      ca_sensors_ds,
      daily_ca_sensors,
      ua_swe_ds,
      mjo_ds,
      pdsi_ds,
  ]
  if clim_ind_ds is not None:
    to_align.insert(3, clim_ind_ds)

  aligned = list(xr.align(*to_align, join="inner"))
  for ds_item in aligned:
    ds_item.load()

  if clim_ind_ds is not None:
    (
        target_ds,
        predictor_snotel_mean_ds,
        predictor_snotel_eom_ds,
        predictor_clim_ind_ds,
        predictor_prism_ds,
        predictor_ca_sensors_ds,
        predictor_daily_ca_sensors,
        predictor_ua_swe_ds,
        predictor_mjo_ds,
        predictor_pdsi_ds,
    ) = aligned
  else:
    predictor_clim_ind_ds = None
    (
        target_ds,
        predictor_snotel_mean_ds,
        predictor_snotel_eom_ds,
        predictor_prism_ds,
        predictor_ca_sensors_ds,
        predictor_daily_ca_sensors,
        predictor_ua_swe_ds,
        predictor_mjo_ds,
        predictor_pdsi_ds,
    ) = aligned

  # 3. Pad 6 placeholder months at record end for +6m target stencils.
  real_time_unpadded = target_ds.time.values.copy()
  last_time = pd.Timestamp(real_time_unpadded[-1])
  pad_times = pd.date_range(
      start=last_time + pd.DateOffset(months=1),
      periods=N_PAD_MONTHS,
      freq="MS",
  ).values.astype(real_time_unpadded.dtype)
  extended_times = np.concatenate([real_time_unpadded, pad_times])

  target_ds = target_ds.reindex(time=extended_times)
  predictor_snotel_mean_ds = predictor_snotel_mean_ds.reindex(
      time=extended_times
  )
  predictor_snotel_eom_ds = predictor_snotel_eom_ds.reindex(time=extended_times)
  predictor_prism_ds = predictor_prism_ds.reindex(time=extended_times)
  predictor_ca_sensors_ds = predictor_ca_sensors_ds.reindex(time=extended_times)
  predictor_daily_ca_sensors = predictor_daily_ca_sensors.reindex(
      time=extended_times
  )
  predictor_ua_swe_ds = predictor_ua_swe_ds.reindex(time=extended_times)
  predictor_mjo_ds = predictor_mjo_ds.reindex(time=extended_times)
  predictor_pdsi_ds = predictor_pdsi_ds.reindex(time=extended_times)
  if predictor_clim_ind_ds is not None:
    predictor_clim_ind_ds = predictor_clim_ind_ds.reindex(time=extended_times)

  predictor_fnf_ds = target_ds.copy()
  real_time = target_ds.time.values.copy()

  # 4. Map variable-length calendar months to a uniform 30-day coordinate.
  synthetic_time = (
      np.datetime64(str(real_time[0]))
      + np.arange(len(real_time)) * SYNTHETIC_MONTH_STEP
  )
  time_vars = dict(
      real_time=("time", real_time),
      month=("time", (real_time.astype("datetime64[M]").astype(int) % 12) + 1),
  )

  target_ds = target_ds.assign_coords(time=synthetic_time)
  predictor_snotel_mean_ds = predictor_snotel_mean_ds.assign_coords(
      time=synthetic_time, **time_vars
  )
  if "spatial_ref" in predictor_prism_ds:
    predictor_prism_ds = predictor_prism_ds.drop_vars("spatial_ref")
  predictor_prism_ds = predictor_prism_ds.assign_coords(time=synthetic_time)
  predictor_ca_sensors_ds = predictor_ca_sensors_ds.assign_coords(
      time=synthetic_time
  )
  predictor_ua_swe_ds = predictor_ua_swe_ds.assign_coords(time=synthetic_time)
  predictor_pdsi_ds = predictor_pdsi_ds.assign_coords(time=synthetic_time)
  predictor_mjo_ds = predictor_mjo_ds.assign_coords(time=synthetic_time)
  predictor_fnf_ds = predictor_fnf_ds.assign_coords(time=synthetic_time)
  predictor_snotel_eom_ds = predictor_snotel_eom_ds.assign_coords(
      time=synthetic_time, **time_vars
  ).reset_coords(names=("real_time", "month"))
  predictor_daily_ca_sensors = predictor_daily_ca_sensors.assign_coords(
      time=synthetic_time
  )
  if predictor_clim_ind_ds is not None:
    predictor_clim_ind_ds = predictor_clim_ind_ds.assign_coords(
        time=synthetic_time
    )

  datasets_dict: Dict[str, xr.Dataset] = {
      "target_fnf": target_ds,
      "snotel": predictor_snotel_mean_ds,
      "snotel_eom": predictor_snotel_eom_ds,
      "past_fnf": predictor_fnf_ds,
      "mjo": predictor_mjo_ds,
      "prism": predictor_prism_ds,
      "ca_sensors": predictor_ca_sensors_ds,
      "daily_ca_sensors": predictor_daily_ca_sensors,
      "ua_swe": predictor_ua_swe_ds,
      "pdsi": predictor_pdsi_ds,
  }
  if predictor_clim_ind_ds is not None:
    datasets_dict["clim_ind"] = predictor_clim_ind_ds

  return {
      "datasets_dict": datasets_dict,
      "aux_ds": aux_ds,
      "fnf_ds": fnf_ds,
      "real_time": real_time,
      "synthetic_time": synthetic_time,
  }


def build_terrax_dataloader(
    data_dir = None,
    past_history_months = 5,
):
  """Constructs the causal Terrax `DataLoader` and input/target specs.

  Args:
    data_dir: Optional path overriding `FLOWRACLE_DATA_DIR`.
    past_history_months: Number of trailing months before the `lag_time = 0`
      origin month (`5` yields a 6-month window `[-5, ..., 0]`).

  Returns:
    `AlignedFlowracleData` instance ready to build train and eval iterators.
  """
  import coordax as cx  # pylint: disable=g-import-not-at-top
  from terrax.core import coordinates  # pylint: disable=g-import-not-at-top
  from terrax.training import data_loading  # pylint: disable=g-import-not-at-top

  bundle = load_and_align_datasets(data_dir=data_dir)
  datasets_dict = bundle["datasets_dict"]
  real_time = bundle["real_time"]
  synthetic_time = bundle["synthetic_time"]

  dataloader = data_loading.DataLoader(datasets_dict, parallelism_mesh=None)

  target_ds = datasets_dict["target_fnf"]
  snotel_mean_ds = datasets_dict["snotel"]
  snotel_eom_ds = datasets_dict["snotel_eom"]
  ca_sensors_ds = datasets_dict["ca_sensors"]
  daily_ca_ds = datasets_dict["daily_ca_sensors"]
  prism_ds = datasets_dict["prism"]
  ua_swe_ds = datasets_dict["ua_swe"]
  pdsi_ds = datasets_dict["pdsi"]
  mjo_ds = datasets_dict["mjo"]

  fnf_station_axis = cx.LabeledAxis("station_id", target_ds.station_id.values)
  snotel_station_axis = cx.LabeledAxis("index", snotel_mean_ds.index.values)
  snotel_eom_axis = cx.LabeledAxis("eom_lag", snotel_eom_ds.eom_lag.values)
  snotel_daily_axes = cx.coords.compose(snotel_station_axis, snotel_eom_axis)
  ca_sensors_axis = cx.LabeledAxis("station", ca_sensors_ds.station.values)
  ca_sensors_daily_axes = cx.coords.compose(ca_sensors_axis, snotel_eom_axis)
  prism_axes = cx.coords.compose(
      cx.LabeledAxis("prism_latitude", prism_ds.prism_latitude.values),
      cx.LabeledAxis("prism_longitude", prism_ds.prism_longitude.values),
  )
  ua_swe_axes = cx.coords.compose(
      cx.LabeledAxis("ua_latitude", ua_swe_ds.ua_latitude.values),
      cx.LabeledAxis("ua_longitude", ua_swe_ds.ua_longitude.values),
  )
  pdsi_axes = cx.coords.compose(
      cx.LabeledAxis("pdsi_latitude", pdsi_ds.pdsi_latitude.values),
      cx.LabeledAxis("pdsi_longitude", pdsi_ds.pdsi_longitude.values),
  )

  realtime_months = np.arange(
      -past_history_months, 1
  )  # [-5, -4, -3, -2, -1, 0]
  lagged_months = np.arange(-past_history_months, 0)  # [-5, -4, -3, -2, -1]
  last_month = np.arange(0, 1)  # [0]
  target_months = np.arange(1, 7)  # [1, 2, 3, 4, 5, 6]

  predictor_td = coordinates.TimeDelta(realtime_months * SYNTHETIC_MONTH_STEP)
  lagged_td = coordinates.TimeDelta(lagged_months * SYNTHETIC_MONTH_STEP)
  last_month_td = coordinates.TimeDelta(last_month * SYNTHETIC_MONTH_STEP)
  target_td = coordinates.TimeDelta(target_months * SYNTHETIC_MONTH_STEP)

  with_predictor_td = lambda c: cx.coords.compose(predictor_td, c)
  with_lagged_td = lambda c: cx.coords.compose(lagged_td, c)
  with_last_month_td = lambda c: cx.coords.compose(last_month_td, c)
  with_target_td = lambda c: cx.coords.compose(target_td, c)

  inputs_spec: Dict[str, Any] = {
      "target_fnf": {"fnf": with_target_td(fnf_station_axis)},
      "snotel": {
          var: with_predictor_td(snotel_station_axis)
          for var in snotel_mean_ds.data_vars
          if "index" in snotel_mean_ds[var].dims
      },
      "prism": {
          var: with_predictor_td(prism_axes) for var in prism_ds.data_vars
      },
      "ua_swe": {
          var: with_predictor_td(ua_swe_axes) for var in ua_swe_ds.data_vars
      },
      "ca_sensors": {
          var: with_predictor_td(ca_sensors_axis)
          for var in ca_sensors_ds.data_vars
      },
      "pdsi": {var: with_predictor_td(pdsi_axes) for var in pdsi_ds.data_vars},
      "mjo": {var: predictor_td for var in mjo_ds.data_vars},
      "snotel_eom": (
          {
              var: with_last_month_td(snotel_daily_axes)
              for var in snotel_eom_ds.data_vars
              if "index" in snotel_eom_ds[var].dims
          }
          | {"real_time": last_month_td, "month": last_month_td}
      ),
      "daily_ca_sensors": {
          var: with_last_month_td(ca_sensors_daily_axes)
          for var in daily_ca_ds.data_vars
      },
      "past_fnf": {"fnf": with_lagged_td(fnf_station_axis)},
  }
  if "clim_ind" in datasets_dict:
    clim_ds = datasets_dict["clim_ind"]
    inputs_spec["clim_ind"] = {
        var: lagged_td
        for var in clim_ds.data_vars
        if var not in EXCLUDED_CLIM_VARS
    }

  train_inputs_spec = copy.deepcopy(inputs_spec)
  del train_inputs_spec["snotel_eom"]["real_time"]

  def real_slice_to_synth(
      real_slice,
  ):
    if isinstance(real_slice, list):
      return [real_slice_to_synth(subslice) for subslice in real_slice]
    return (
        str(synthetic_time[np.where(real_time >= real_slice[0])[0][0]]),
        str(synthetic_time[np.where(real_time >= real_slice[-1])[0][0]]),
    )

  return AlignedFlowracleData(
      dataloader=dataloader,
      inputs_spec=inputs_spec,
      train_inputs_spec=train_inputs_spec,
      aux_ds=bundle["aux_ds"],
      fnf_ds=bundle["fnf_ds"],
      real_time=real_time,
      synthetic_time=synthetic_time,
      real_slice_to_synth=real_slice_to_synth,
  )

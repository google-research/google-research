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

"""Physics-informed hydroclimatic feature engineering for Flowracle.

This module maps multi-source gridded and point observations onto a 55-feature
tabular representation per `(batch_date, lead_horizon, watershed)` tuple.

Summary of Physical Feature Groups (55 total)
---------------------------------------------
1. **Cyclical issuance encoding (2)**: `sin_m`, `cos_m` of the last observed
   month `m_iss` prior to forecast issuance.
2. **Latitudinal hydroclimatic bands (4 bands x 6 = 24)**: Across northern
   (`39.5–42.0°N`), mid-northern (`38.0–39.5°N`), mid-southern (`36.5–38.0°N`),
   and southern (`32.5–36.5°N`) California mountain bands:
   - Log-transformed 6-month PRISM precipitation accumulation (`b_*_ppt_acc`)
   - Cumulative positive degree-day index (`b_*_dd`)
   - Log-transformed UA 4 km SWE (`b_*_swe`)
   - Snowpack ripeness index (`b_*_ripeness`, ratio of cumulative thermal
     forcing to snowpack water mass, capped at 10)
   - Month-over-month SWE velocity (`b_*_swe_vel`)
   - Months elapsed since 6-month peak SWE (`b_*_time_since_peak`)
3. **Subseasonal MJO teleconnections (3)**: End-of-month RMM phase
(`mjo_sin_recent`,
   `mjo_cos_recent`) and monthly mean amplitude (`monthly_mjo_amp_recent`).
4. **Target lead & watershed static coordinates (6)**: Target calendar month
   encoding (`lead_sin`, `lead_cos`, `is_amjj`) and normalized gauge latitude,
   longitude, and elevation (`stn_lat`, `stn_lon`, `stn_elev`).
5. **Local watershed & 3D inverse-distance-weighted (IDW) predictors (20)**:
   - 3D-IDW interpolation over the `K = 32` nearest SNOTEL and CDEC snow courses
     in `(latitude, longitude, elevation / 720 m)` space, with CDEC snow sensors
     conditionally gated south of `37.0°N` (`LAT_THRESH`) where SNOTEL coverage
     thins out in the high southern Sierra (Kings, Kaweah, Tule, Kern).
   - **Antecedent soil priming efficiency (`l_priming_state`)**: Sigmoidal
     efficiency combining regional gridMET Palmer Drought Severity Index (PDSI)
     and local 6-month PRISM precipitation to capture soil-moisture deficit
     losses (e.g., dry-soil runoff suppression following multi-year drought).
   - **March kinematic storm-surge damping (`surge_damping`)**: Attenuates
     transient late-February snow surges (`SURGE_SENSITIVITY = 0.75`) on March 1
     issuances so single storms do not inflate full-season runoff projections.
   - **Cross-basin storm synchrony (`l_storm_coherence`,
   `l_storm_bulk_ratio`)**:
     Measures regional spatial coherence of monthly SWE accumulation across
     California's snow-fed mountain bands.
   - **3D-IDW lagged neighbor runoff (`l_nbr_fnf`, `l_nbr_fnf_avg`)** across the
     `K = 8` nearest neighboring watersheds.
"""

from typing import Any, Dict, List, Optional, Tuple

from flowracle.data_pipeline import get_batch_predictors_ds
from flowracle.data_pipeline import get_batch_target_ds
import numpy as np

# Volumetric and physical scaling constants
TARGET_SCALE = 1e4  # 10,000 Acre-Feet (TAF x 10)
SNOW_SCALE = 450.0  # Reference SWE depth (mm)
PRECIP_SCALE = 60.0  # Reference monthly precipitation (mm)
TEMP_SCALE = 15.0  # Reference temperature scale (deg C)
NUM_ENSEMBLE = 100  # Number of members interpolated from quantile predictions

# Ensemble combination & physical regularization hyperparameters
SHRINKAGE_FACTOR = 0.18  # Shrinkage toward uniform (1/3, 1/3, 1/3) weights
SURGE_SENSITIVITY = 0.75  # March kinematic storm-surge damping coefficient
SURGE_BASE_EPS = 25.0  # Basepack SWE denominator stabilizer (mm)
LOG_BIAS_CORRECTION_FACTOR = 0.65  # Log-space Jensen bias-correction factor
CLIM_PERCENTILE = 50.0  # Median climatology anchor for Anomaly sub-model
LEAD_REG_GROWTH = 0.60  # L2 regularization growth factor
LEAD_RELAX_SLOPE = 0.45  # Outer-quantile L2 relaxation slope
ENSEMBLE_POWER = 4.2  # Inverse-RMSE exponent for monthly OOF weights

# 3D inverse-distance-weighting (IDW) & spatial gating constants
LAT_THRESH = 37.0  # Latitude gate (deg N) activating CDEC snow courses
TOP_K_SENSORS = 32  # Nearest snow sensors in 3D (lat, lon, elev) space
TOP_K_FNF_NEIGHBORS = 8  # Nearest FNF gauges for lagged neighbor runoff
DAILY_VEL_DIVISOR = 5.0  # 5-day trailing window divisor for daily SWE velocity
ELEV_SCALE = 720.0  # Elevation-to-degree scaling (m / deg) in 3D IDW
FT_TO_M = 0.3048  # Feet-to-meters conversion for CDEC station metadata
OUTER_QUANTILE_RELAX_FACTOR = 0.45

# Substrings identifying features constrained to be monotonically non-decreasing
# with respect to log-streamflow in every XGBoost quantile tree.
POS_TERMS = [
    "ppt",
    "swe",
    "fnf",
    "pot",
    "acc",
    "elev_swe",
    "gate",
    "daily_ca",
    "frac",
    "available",
    "priming",
    "interact",
    "interaction",
]

# Training-time stochastic augmentation & temporal lag-phase jittering scales
CLIM_PRED_JITTER = 0.10
PDSI_PRED_JITTER = 0.08
PROVISIONAL_JITTER = 0.07
GROUND_TRUTH_JITTER = 0.015
TARGET_JITTER = 0.012
LAG_PHASE_JITTER = 0.15
AUGMENTATION_EPOCHS = 3
MJO_ANGLE_JITTER = 0.05

# Latitudinal California mountain sub-regions (degrees North)
BAND_EDGES = [42.0, 39.5, 38.0, 36.5, 32.5]
BAND_NAMES = ["north", "mid_north", "mid_south", "south"]
NUM_LOCAL_FEATURES = 20

# 13 target quantiles spanning [0.01, 0.99]
QUANTILES = np.unique(
    np.concatenate([
        [0.01, 0.05],
        np.linspace(0.1, 0.9, 9),
        [0.95, 0.99],
    ])
)


def prepare_shared_features(
    batch,
    aux_ds,
    jitter = False,
):
  """Computes global regional-band features and local 3D-IDW watershed features.

  Args:
    batch: Terrax batch or mapping `{source_name: xr.Dataset}`.
    aux_ds: Static metadata `xarray.Dataset` containing FNF, SNOTEL, and CDEC
      station latitudes, longitudes (`0..360`), and elevations.
    jitter: If True, applies training-time temporal lag-phase interpolation and
      observational noise augmentation. Must be False at evaluation/inference.

  Returns:
    Tuple `(global_feat, local_feat, global_names, local_names, m_iss)` where:
      - `global_feat` has shape `(B, 29)`
      - `local_feat` has shape `(B, S, 20)`
      - `global_names` lists the 29 global feature names
      - `local_names` lists the 20 local feature names
      - `m_iss` has shape `(B,)` containing the calendar month (`1..12`) of the
        `lag_time = 0` predictor origin (i.e., `issuance_month - 1 mod 12`).
  """
  predictors = get_batch_predictors_ds(batch)
  batch_size = next(iter(predictors.values())).sizes["batch"]
  parts: List[np.ndarray] = []
  names: List[str] = []

  def add_f(data, name_root, fill_nan = 0.0):
    v = np.nan_to_num(data.astype(np.float32), nan=fill_nan)
    if len(v.shape) == 1:
      v = v.reshape(batch_size, -1)
    parts.append(v)
    n_cols = v.shape[-1]
    if n_cols == 1:
      names.append(name_root)
    else:
      names.extend([f"{name_root}_{i}" for i in range(n_cols)])

  # 1. Cyclical month of predictor origin
  m_iss = predictors["snotel_eom"]["month"].values[:, -1]
  add_f(np.sin(2 * np.pi * m_iss / 12), "sin_m")
  add_f(np.cos(2 * np.pi * m_iss / 12), "cos_m")

  # 2. Latitudinal regional bands (PRISM precipitation/degree-days & UA SWE)
  p = predictors["prism"]
  ua = predictors["ua_swe"]
  melt_gate = ((m_iss >= 3) & (m_iss <= 7)).astype(np.float32)

  for i, (b_max, b_min) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
    b_name = BAND_NAMES[i]
    dd_acc = np.zeros(batch_size, dtype=np.float32)
    p_mask = (p.prism_latitude.values >= b_min) & (
        p.prism_latitude.values < b_max
    )
    if np.any(p_mask):
      ppt_v = np.nan_to_num(p["ppt"].values[:, :, p_mask, :], 0.0).mean(
          axis=(2, 3)
      )
      tmean_v = np.nan_to_num(p["tmean"].values[:, :, p_mask, :], 0.0).mean(
          axis=(2, 3)
      )
      add_f(np.log1p(ppt_v.sum(axis=1) / PRECIP_SCALE), f"b_{b_name}_ppt_acc")
      dd_acc = np.maximum(0, tmean_v).sum(axis=1) / 30.0
      add_f(dd_acc, f"b_{b_name}_dd")

    ua_mask = (ua.ua_latitude.values >= b_min) & (ua.ua_latitude.values < b_max)
    if np.any(ua_mask):
      swe_v = np.nan_to_num(ua["SWE"].values[:, :, ua_mask, :], 0.0).mean(
          axis=(2, 3)
      )
      curr_swe = swe_v[:, -1]
      if jitter and LAG_PHASE_JITTER > 0.0 and swe_v.shape[1] >= 3:
        alpha = np.random.uniform(
            0.0, LAG_PHASE_JITTER, size=(batch_size,)
        ).astype(np.float32)
        prev_swe = (1.0 - alpha) * swe_v[:, -2] + alpha * swe_v[:, -3]
      else:
        prev_swe = swe_v[:, -2]
      add_f(np.log1p(curr_swe / SNOW_SCALE), f"b_{b_name}_swe")
      add_f(
          np.minimum(dd_acc / (curr_swe / 100.0 + 1.0), 10.0),
          f"b_{b_name}_ripeness",
      )
      add_f((curr_swe - prev_swe) / 50.0, f"b_{b_name}_swe_vel")
      add_f(
          ((swe_v.shape[1] - 1) - np.argmax(swe_v, axis=1)).astype(np.float32),
          f"b_{b_name}_time_since_peak",
      )

  # 3. Subseasonal MJO phase & amplitude (operational 2-day lag applied
  # upstream).
  rmm1 = np.nan_to_num(
      predictors["mjo"]["eom_rmm1"].values.astype(np.float32)[:, -1], 0.0
  )
  rmm2 = np.nan_to_num(
      predictors["mjo"]["eom_rmm2"].values.astype(np.float32)[:, -1], 0.0
  )
  amp = np.nan_to_num(
      predictors["mjo"]["monthly_mjo_amp"].values.astype(np.float32)[:, -1], 0.0
  )
  mjo_angle = np.arctan2(rmm2, rmm1)
  if jitter and MJO_ANGLE_JITTER > 0.0:
    mjo_angle += np.random.normal(
        0.0, MJO_ANGLE_JITTER, size=mjo_angle.shape
    ).astype(np.float32)
    amp *= 1.0 + np.random.normal(0.0, CLIM_PRED_JITTER, size=amp.shape).astype(
        np.float32
    )
  add_f(amp, "monthly_mjo_amp_recent")
  add_f(np.sin(mjo_angle), "mjo_sin_recent")
  add_f(np.cos(mjo_angle), "mjo_cos_recent")

  # 4. Regional Palmer Drought Severity Index (PDSI) for antecedent soil priming
  pdsi_val = np.zeros(batch_size, dtype=np.float32)
  if "pdsi" in predictors:
    pdsi_val = np.nan_to_num(
        predictors["pdsi"]["daily_mean_palmer_drought_severity_index"]
        .values.astype(np.float32)
        .mean(axis=(2, 3))[:, 0],
        0.0,
    )
    if jitter and PDSI_PRED_JITTER > 0.0:
      pdsi_val += np.random.normal(
          0.0, PDSI_PRED_JITTER, size=pdsi_val.shape
      ).astype(np.float32)

  global_feat = np.concatenate(parts, -1)

  # 5. Local watershed predictors & 3D-IDW snow/precip/neighbor-runoff features
  if aux_ds is not None:
    n_stations = len(aux_ds.station_id)
  elif "past_fnf" in predictors and "fnf" in predictors["past_fnf"]:
    n_stations = predictors["past_fnf"]["fnf"].shape[-1]
  else:
    n_stations = 1
  l_names = [
      "l_ppt_acc",
      "l_swe",
      "l_fnf_m2",
      "l_fnf_avg",
      "l_sn_elev_swe",
      "l_sn_vel",
      "l_primed_melt_gate",
      "l_eom_swe_diff",
      "l_ca_swe",
      "l_ca_vel",
      "l_daily_ca_swe",
      "l_daily_ca_vel",
      "l_nbr_fnf",
      "l_nbr_fnf_avg",
      "l_snow_frac",
      "l_melt_available",
      "l_priming_state",
      "l_storm_coherence",
      "l_storm_bulk_ratio",
      "l_sn_snow_frac",
  ]
  if aux_ds is not None:
    l_ppt = np.nan_to_num(predictors["prism"]["ppt"].values, 0.0)
    l_swe = np.nan_to_num(predictors["ua_swe"]["SWE"].values, 0.0)
    past_fnf = np.nan_to_num(
        predictors["past_fnf"]["fnf"].values[:, :, :n_stations], 0.0
    )
    if jitter and GROUND_TRUTH_JITTER > 0.0:
      past_fnf *= 1.0 + np.random.normal(
          0.0, GROUND_TRUTH_JITTER, size=past_fnf.shape
      ).astype(np.float32)
    fnf_m2 = past_fnf[:, -1, :]

    la_idx = np.array([
        np.abs(p.prism_latitude.values - lat).argmin()
        for lat in aux_ds.latitude.values[:n_stations]
    ])
    lo_idx = np.array([
        np.abs(p.prism_longitude.values - lon).argmin()
        for lon in aux_ds.longitude.values[:n_stations]
    ])

    sn_lat = aux_ds.snotel_latitude.values
    sn_lon = aux_ds.snotel_longitude.values
    sn_elev = np.nan_to_num(aux_ds.snotel_elevation.values, 0.0)
    fnf_lat = aux_ds.latitude.values[:n_stations]
    fnf_lon = aux_ds.longitude.values[:n_stations]
    fnf_elev = (
        np.nan_to_num(aux_ds.elevation.values[:n_stations], 0.0) * FT_TO_M
    )
    ca_lat = aux_ds.ca_sensor_latitude.values
    ca_lon = aux_ds.ca_sensor_longitude.values
    ca_elev = np.nan_to_num(aux_ds.ca_sensor_elevation.values, 0.0) * FT_TO_M

    def get_idw_feat(
        lat_arr,
        lon_arr,
        elev_arr,
        val_raw,
        top_k,
        conditional_lat = None,
    ):
      weighted = []
      for i in range(n_stations):
        if conditional_lat is not None and fnf_lat[i] >= conditional_lat:
          weighted.append(np.zeros(val_raw.shape[:-1], dtype=np.float32))
          continue
        dist = (
            (lat_arr - fnf_lat[i]) ** 2
            + (lon_arr - fnf_lon[i]) ** 2
            + ((elev_arr - fnf_elev[i]) / ELEV_SCALE) ** 2
        )
        top_idx = np.argsort(dist)[:top_k]
        vals = val_raw[Ellipsis, top_idx]
        w = 1.0 / (dist[top_idx] + 0.1)
        norm_w = (
            (w[None, :] * (~np.isnan(vals)))
            if len(vals.shape) == 2
            else (w[None, None, :] * (~np.isnan(vals)))
        )
        norm_w /= norm_w.sum(axis=-1, keepdims=True) + 1e-12
        weighted.append((np.nan_to_num(vals, nan=0.0) * norm_w).sum(axis=-1))
      return np.stack(weighted, axis=1 if len(val_raw.shape) == 2 else 2)

    w_sn_swe = get_idw_feat(
        sn_lat,
        sn_lon,
        sn_elev,
        predictors["snotel"]["snow_water_equivalent_start_of_day_PST"].values
        * 1000.0,
        TOP_K_SENSORS,
    )
    w_sn_precip = get_idw_feat(
        sn_lat,
        sn_lon,
        sn_elev,
        predictors["snotel"][
            "precipitation_accumulation_start_of_day_PST"
        ].values
        * 1000.0,
        TOP_K_SENSORS,
    )
    w_eom_swe = get_idw_feat(
        sn_lat,
        sn_lon,
        sn_elev,
        predictors["snotel_eom"][
            "snow_water_equivalent_start_of_day_PST"
        ].values[:, -1, :, -1]
        * 1000.0,
        TOP_K_SENSORS,
    )
    l_eom_diff = (w_eom_swe - w_sn_swe[:, -1, :]) / 50.0

    # CDEC snow courses (in -> mm) activated south of 37N where SNOTEL is
    # sparse.
    w_ca_swe = get_idw_feat(
        ca_lat,
        ca_lon,
        ca_elev,
        predictors["ca_sensors"]["sensor_82"].values * 25.4,
        TOP_K_SENSORS,
        LAT_THRESH,
    )
    daily_ca_raw = (predictors["daily_ca_sensors"]["sensor_82"].values * 25.4)[
        :, -1, :, :
    ].transpose(0, 2, 1)
    w_daily_ca = get_idw_feat(
        ca_lat, ca_lon, ca_elev, daily_ca_raw, TOP_K_SENSORS, LAT_THRESH
    )
    if jitter and PROVISIONAL_JITTER > 0.0:
      w_daily_ca *= 1.0 + np.random.normal(
          0.0, PROVISIONAL_JITTER, size=w_daily_ca.shape
      ).astype(np.float32)
    daily_ca_vel = (
        w_daily_ca[:, -1, :] - w_daily_ca[:, 0, :]
    ) / DAILY_VEL_DIVISOR

    w_nbr_fnf, w_nbr_fnf_avg = [], []
    for i in range(n_stations):
      dist = (
          (fnf_lat - fnf_lat[i]) ** 2
          + (fnf_lon - fnf_lon[i]) ** 2
          + ((fnf_elev - fnf_elev[i]) / ELEV_SCALE) ** 2
      )
      dist[i] = np.inf
      top_idx = np.argsort(dist)[:TOP_K_FNF_NEIGHBORS]
      f_vals = past_fnf[:, -1, top_idx]
      f_avg_vals = past_fnf.mean(axis=1)[:, top_idx]
      w = 1.0 / (dist[top_idx] + 0.1)
      norm_w = w / (w.sum() + 1e-12)
      w_nbr_fnf.append((f_vals * norm_w[None, :]).sum(axis=-1))
      w_nbr_fnf_avg.append((f_avg_vals * norm_w[None, :]).sum(axis=-1))

    if jitter and LAG_PHASE_JITTER > 0.0 and w_sn_swe.shape[1] >= 3:
      alpha_sn = np.random.uniform(
          0.0, LAG_PHASE_JITTER, size=(batch_size, 1)
      ).astype(np.float32)
      sn_prev = (1.0 - alpha_sn) * w_sn_swe[:, -2, :] + alpha_sn * w_sn_swe[
          :, -3, :
      ]
    else:
      sn_prev = w_sn_swe[:, -2, :]

    if jitter and LAG_PHASE_JITTER > 0.0 and w_ca_swe.shape[1] >= 3:
      alpha_ca = np.random.uniform(
          0.0, LAG_PHASE_JITTER, size=(batch_size, 1)
      ).astype(np.float32)
      ca_prev = (1.0 - alpha_ca) * w_ca_swe[:, -2, :] + alpha_ca * w_ca_swe[
          :, -3, :
      ]
    else:
      ca_prev = w_ca_swe[:, -2, :]

    # Antecedent soil-moisture priming efficiency (0..1)
    local_ppt_acc = l_ppt.sum(axis=1)[:, la_idx, lo_idx]
    priming_factor = 0.6 * pdsi_val[:, None] + 0.4 * (
        local_ppt_acc / PRECIP_SCALE - 5.0
    )
    primed_efficiency = 1.0 / (1.0 + np.exp(-0.5 * priming_factor))

    # March kinematic storm-surge damping (`m_iss == 3` corresponds to end of
    # March, or early-spring transient surges depending on predictor month).
    sn_prev2 = w_sn_swe[:, -3, :] if w_sn_swe.shape[1] >= 3 else sn_prev
    sn_swe_3mo = (w_sn_swe[:, -1, :] + sn_prev + sn_prev2) / 3.0
    sn_surge_delta = np.maximum(0.0, w_sn_swe[:, -1, :] - sn_prev)
    sn_surge_ratio = np.clip(
        sn_surge_delta / (sn_swe_3mo + SURGE_BASE_EPS), 0.0, 3.0
    )
    is_march = np.repeat(
        (m_iss[:, None] == 3).astype(np.float32), n_stations, axis=1
    )
    surge_damping = 1.0 / (1.0 + SURGE_SENSITIVITY * sn_surge_ratio * is_march)

    damped_sn_swe = w_sn_swe[:, -1, :] * surge_damping
    damped_ca_swe = w_ca_swe[:, -1, :] * surge_damping
    damped_ua_swe = l_swe[:, -1, la_idx, lo_idx] * surge_damping

    # Cross-basin storm synchrony & bulk surge ratio
    sn_vel_raw = w_sn_swe[:, -1, :] - sn_prev
    vel_mean = np.mean(sn_vel_raw, axis=1, keepdims=True)
    vel_std = np.std(sn_vel_raw, axis=1, keepdims=True)
    storm_coherence = np.repeat(
        1.0 / (1.0 + vel_std / (np.abs(vel_mean) + 1.0)), n_stations, axis=1
    )
    mean_pos_vel = np.repeat(
        np.mean(np.maximum(0.0, sn_vel_raw), axis=1, keepdims=True),
        n_stations,
        axis=1,
    )
    storm_bulk_ratio = mean_pos_vel / (w_sn_swe[:, -1, :] + 1.0)

    # Preserve IEEE-754 NaN routing for occasional negative sensor drift values
    # so XGBoost's learned missing-value branch handles sensor floor noise.
    with np.errstate(invalid="ignore"):
      local_feat = np.stack(
          [
              np.log1p(local_ppt_acc / PRECIP_SCALE),
              np.log1p(damped_ua_swe / SNOW_SCALE),
              np.log1p(fnf_m2 / TARGET_SCALE),
              np.log1p(past_fnf.mean(axis=1) / TARGET_SCALE),
              np.log1p(damped_sn_swe / SNOW_SCALE),
              (w_sn_swe[:, -1, :] - sn_prev) / 50.0,
              l_eom_diff * melt_gate[:, None] * primed_efficiency,
              l_eom_diff,
              np.log1p(damped_ca_swe / SNOW_SCALE),
              (w_ca_swe[:, -1, :] - ca_prev) / 50.0,
              np.log1p(w_daily_ca[:, -1, :] / SNOW_SCALE),
              daily_ca_vel * surge_damping,
              np.log1p(np.stack(w_nbr_fnf, 1) / TARGET_SCALE),
              np.log1p(np.stack(w_nbr_fnf_avg, 1) / TARGET_SCALE),
              np.log1p(damped_ua_swe / (local_ppt_acc + 1.0)),
              np.log1p((damped_sn_swe / SNOW_SCALE) * primed_efficiency),
              primed_efficiency,
              storm_coherence,
              storm_bulk_ratio,
              np.log1p(damped_sn_swe / (w_sn_precip[:, -1, :] + 1.0)),
          ],
          axis=-1,
      )
  else:
    local_feat = np.zeros(
        (batch_size, n_stations, NUM_LOCAL_FEATURES), dtype=np.float32
    )

  return (
      global_feat.astype(np.float32),
      local_feat.astype(np.float32),
      names,
      l_names,
      m_iss,
  )


def batch_to_table(
    batch,
    aux_ds,
    coords_np,
    jitter = False,
):
  """Flattens a `(batch, lead=6, station=S)` batch into a 2D tabular matrix.

  Rows are ordered by `(lead_idx in 0..5, station_idx in 0..S-1, batch_idx in
  0..B-1)`.
  Targets `y` are log-transformed via `log1p(max(FNF, 0) / TARGET_SCALE)`.

  Args:
    batch: Terrax batch or mapping `{source_name: xr.Dataset}`.
    aux_ds: Static watershed and sensor metadata dataset.
    coords_np: Normalized watershed coordinate array of shape `(S, 3)`.
    jitter: Whether to apply training-time stochastic augmentation.

  Returns:
    Tuple `(X, y, feature_names, m_iss_flat)` where `X` has shape
    `(6 * S * B, 55)`.
  """
  f_glob, f_loc, n_glob, n_loc, m_iss = prepare_shared_features(
      batch, aux_ds, jitter=jitter
  )
  batch_size = f_glob.shape[0]
  n_stations = coords_np.shape[0]
  leads = np.arange(1, 7)
  t_months = (m_iss[:, None] + leads[None, :] - 1) % 12 + 1
  l_enc = np.stack(
      [
          np.sin(2 * np.pi * t_months / 12),
          np.cos(2 * np.pi * t_months / 12),
          (t_months >= 4) & (t_months <= 7),
      ],
      -1,
  ).astype(np.float32)

  y_raw = (
      get_batch_target_ds(batch)["target_fnf_fnf"].values
      if "target_fnf" in batch
      else None
  )
  if y_raw is not None and jitter and TARGET_JITTER > 0.0:
    y_raw = y_raw * (
        1.0
        + np.random.normal(0.0, TARGET_JITTER, size=y_raw.shape).astype(
            np.float32
        )
    )

  x_list, y_list, m_iss_list = [], [], []
  for l_idx in range(6):
    for s_idx in range(n_stations):
      x_list.append(
          np.concatenate(
              [
                  f_glob,
                  l_enc[:, l_idx, :],
                  coords_np[s_idx : s_idx + 1].repeat(batch_size, 0),
                  f_loc[:, 0 if f_loc.shape[1] == 1 else s_idx, :],
              ],
              axis=-1,
          )
      )
      if y_raw is not None:
        y_list.append(
            np.log1p(
                np.maximum(np.nan_to_num(y_raw[:, l_idx, s_idx], 0.0), 0.0)
                / TARGET_SCALE
            )
        )
      m_iss_list.extend(m_iss.tolist())

  feature_names = (
      n_glob
      + ["lead_sin", "lead_cos", "is_amjj", "stn_lat", "stn_lon", "stn_elev"]
      + n_loc
  )
  return (
      np.concatenate(x_list, 0),
      (np.concatenate(y_list, 0) if y_list else None),
      feature_names,
      np.array(m_iss_list),
  )


# Alias retained for backward compatibility with notebook checkpoints.
_batch_to_table = batch_to_table

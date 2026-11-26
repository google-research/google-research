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

"""Orbit-guided ALOS PRISM destriping for GEM-15 elevation layers.

Estimates and removes ~50 km periodic across-track ALOS PRISM waves (adapted
from Step 1 of Elkhrachy 2026, EarthArXiv 16205) without blending coarse
reference terrain into GEM-15:
  1. Masks (GEM-15 DSM - reference DEM) to bare, gentle ground (reference slope
     < 3 deg and DSM - DTM < 1 m) in a common vertical datum, and removes >150
     km regional bias.
  2. Projects coordinates onto curved sun-synchronous ALOS tracks, computes
     robust medians per 4 km x 35 km bin (>=10% bare-gentle coverage), smooths
     along-track (20 km Gaussian) and slightly across 4 km across-track bin
     boundaries (3.6 km Gaussian after 148 km high-pass detrending).
  3. Optionally blends the 3x3 grid of overlapping 3x3 deg local fits across a
     5x5 deg overview window with smoothstep weights for seamless 1x1 deg tile
     edges, and subtracts S(lon, lat) across all GEM-15 elevation bands.
"""

from collections.abc import Callable, Mapping
import dataclasses
import math

import numpy as np
from scipy import interpolate
from scipy import ndimage

ELEVATION_BANDS: tuple[str, Ellipsis] = (
    'dsm',
    'dsm_percentile_5',
    'dsm_percentile_50',
    'dsm_percentile_95',
    'dtm',
)

_ALOS_INCLINATION_DEG = 98.16
_ALOS_REPEAT_REV_PER_DAY = 671.0 / 46.0
_WGS84_SEMI_MAJOR_KM = 6378.137
_WGS84_ECCENTRICITY_SQ = 0.00669437999014
_METERS_PER_DEG_LAT = 111320.0

# Stripe evaluator `eval_stripe_m(lon_deg, lat_deg) -> stripe_m` (float32, m).
StripeField = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class DestripingConfig:
  """Hyperparameters for GEM-15 orbit-guided destriping.

  Attributes:
    du_km: Across-track bin width in kilometers (perpendicular to ALOS track).
    dv_km: Along-track bin length in kilometers (35 km matches 1 PRISM scene).
    sigma_v_km: Along-track normalized Gaussian smoothing scale in kilometers.
    sigma_u_km: Across-track Gaussian smoothing scale across bin boundaries.
    min_cell_coverage_pct: Minimum percentage (0, 100] of each du_km x dv_km bin
      that must contain valid bare-gentle ground.
    slope_max_deg: Maximum reference DEM slope in degrees for bare-gentle mask.
    ndsm_max_m: Maximum (DSM - DTM) height in meters for bare-ground mask.
    outlier_gate_m: Gross outlier threshold |diff - median| in meters.
    sigma_bias_km: 2D Gaussian scale in kilometers for >150 km regional detrend.
    win_hp_km: Across-track running-mean high-pass window in kilometers.
    min_v_weight: Minimum along-track Gaussian weight before tapering S to 0.
    min_u_weight: Minimum across-track Gaussian weight before tapering S to 0.
  """

  du_km: float = 4.0
  dv_km: float = 35.0
  sigma_v_km: float = 20.0
  sigma_u_km: float = 3.6
  min_cell_coverage_pct: float = 10.0
  slope_max_deg: float = 3.0
  ndsm_max_m: float = 1.0
  outlier_gate_m: float = 15.0
  sigma_bias_km: float = 55.0
  win_hp_km: float = 148.0
  min_v_weight: float = 0.35
  min_u_weight: float = 0.25


def _compute_alos_orbit_azimuth_deg(lat_deg):
  """Computes the ALOS PRISM ground-track azimuth (degrees east of north).

  Args:
    lat_deg: Scalar or array of geodetic latitudes in degrees (clipped to the
      [-81, 81] degree ALOS orbital latitude range).

  Returns:
    Array of ground-track heading angles in degrees east of north.
  """
  phi_rad = np.radians(np.clip(lat_deg, -81.0, 81.0))
  inc_rad = math.radians(_ALOS_INCLINATION_DEG)
  cos_ratio = np.clip(math.cos(inc_rad) / np.cos(phi_rad), -0.999, 0.999)
  numerator = np.cos(phi_rad) - _ALOS_REPEAT_REV_PER_DAY * cos_ratio
  denominator = _ALOS_REPEAT_REV_PER_DAY * np.sqrt(1.0 - cos_ratio * cos_ratio)
  return np.degrees(np.arctan2(numerator, denominator))


def _build_alos_track_lookup_table():
  """Builds a global [-81, 81] deg lookup table for a curved ALOS ground track.

  Integrates the sun-synchronous ALOS orbit azimuth on WGS84 from south to north
  and anchors longitude offset and along-track arc length to 0 at the equator.

  Returns:
    Tuple `(lat_lut_deg, lon_offset_deg, arc_length_km)` of 1D arrays sampled
    every 0.02 degrees across [-81, 81] degrees latitude.
  """
  lat_lut_deg = np.linspace(-81.0, 81.0, 8101)
  phi_g = np.radians(lat_lut_deg)
  denom_g = np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQ * np.sin(phi_g) ** 2)
  merid_km = (
      _WGS84_SEMI_MAJOR_KM * (1.0 - _WGS84_ECCENTRICITY_SQ) / (denom_g**3)
  )
  prime_cos_km = (_WGS84_SEMI_MAJOR_KM / denom_g) * np.cos(phi_g)
  beta_g = np.radians(_compute_alos_orbit_azimuth_deg(lat_lut_deg))

  dlat_deg = lat_lut_deg[1] - lat_lut_deg[0]
  dlon_dlat = np.tan(beta_g) * merid_km / np.maximum(prime_cos_km, 1.0)
  lon_cum = np.concatenate(
      [[0.0], np.cumsum(0.5 * (dlon_dlat[:-1] + dlon_dlat[1:]) * dlat_deg)]
  )
  lon_cum -= float(np.interp(0.0, lat_lut_deg, lon_cum))

  ds_dlat = math.radians(1.0) * merid_km / np.cos(beta_g)
  arc_cum = np.concatenate(
      [[0.0], np.cumsum(0.5 * (ds_dlat[:-1] + ds_dlat[1:]) * dlat_deg)]
  )
  arc_cum -= float(np.interp(0.0, lat_lut_deg, arc_cum))
  return lat_lut_deg, lon_cum, arc_cum


_LAT_LUT_DEG, _LON_TRACK_LUT_DEG, _ARC_TRACK_LUT_KM = (
    _build_alos_track_lookup_table()
)


def _project_lonlat_to_alos_track_km(
    lon_deg,
    lat_deg,
    ref_equator_lon_deg,
):
  """Projects (lon, lat) coordinates into curved ALOS orbit (u, v) kilometers.

  Args:
    lon_deg: Longitude array in WGS84 degrees.
    lat_deg: Latitude array in WGS84 degrees (matching `lon_deg` shape).
    ref_equator_lon_deg: Equator-crossing longitude in degrees of the reference
      ALOS ground track passing through the local fit center.

  Returns:
    Tuple `(across_track_km, along_track_km)` of float64 arrays giving signed
    distance perpendicular (`u`) and parallel (`v`) to the reference ALOS track.
  """
  lon_arr = np.asarray(lon_deg, dtype=np.float64)
  lat_arr = np.clip(np.asarray(lat_deg, dtype=np.float64), -81.0, 81.0)
  lon_track = ref_equator_lon_deg + np.interp(
      lat_arr, _LAT_LUT_DEG, _LON_TRACK_LUT_DEG
  )
  arc_track_km = np.interp(lat_arr, _LAT_LUT_DEG, _ARC_TRACK_LUT_KM)

  phi_pts = np.radians(lat_arr)
  denom_pts = np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQ * np.sin(phi_pts) ** 2)
  ncos_pts_km = (_WGS84_SEMI_MAJOR_KM / denom_pts) * np.cos(phi_pts)
  beta_pts = np.radians(_compute_alos_orbit_azimuth_deg(lat_arr))

  east_off_km = np.radians(lon_arr - lon_track) * ncos_pts_km
  across_track_km = east_off_km * np.cos(beta_pts)
  along_track_km = arc_track_km + east_off_km * np.sin(beta_pts)
  return across_track_km, along_track_km


def _estimate_regional_bias_m(
    raw_diff_m,
    bare_gentle_mask,
    sigma_bias_km,
    dy_km,
    dx_km,
):
  """Estimates the smooth >150 km 2D regional bias surface via Gaussian filter.

  Computes a normalized Gaussian average of `raw_diff_m` over `bare_gentle_mask`
  on a block-downsampled grid and bilinearly upsamples back to the full grid so
  coarse datum or regional offsets are separated from ~50 km stripes.

  Args:
    raw_diff_m: 2D array of raw elevation differences `(gem_dsm - ref_dem)` in
      meters.
    bare_gentle_mask: 2D boolean mask selecting valid bare, gentle, non-outlier
      pixels.
    sigma_bias_km: 2D Gaussian smoothing scale in kilometers.
    dy_km: Pixel spacing along rows (latitude) in kilometers.
    dx_km: Pixel spacing along columns (longitude) in kilometers.

  Returns:
    2D float32 array of estimated regional bias in meters (zeros if
    `bare_gentle_mask` has no valid pixels).
  """
  if not np.any(bare_gentle_mask):
    return np.zeros_like(raw_diff_m, dtype=np.float32)
  height, width = raw_diff_m.shape
  block = max(1, min(16, height // 16, width // 16))
  h_pad = (height // block) * block
  w_pad = (width // block) * block
  mask_sub = (
      bare_gentle_mask[:h_pad, :w_pad]
      .reshape(h_pad // block, block, w_pad // block, block)
      .mean(axis=(1, 3))
      .astype(np.float32)
  )
  diff_sub = (
      np.where(
          bare_gentle_mask[:h_pad, :w_pad], raw_diff_m[:h_pad, :w_pad], 0.0
      )
      .astype(np.float32)
      .reshape(h_pad // block, block, w_pad // block, block)
      .mean(axis=(1, 3))
  )
  sigma_yx = (
      sigma_bias_km / max(dy_km * block, 0.2),
      sigma_bias_km / max(dx_km * block, 0.2),
  )
  weight = ndimage.gaussian_filter(mask_sub, sigma=sigma_yx)
  numer = ndimage.gaussian_filter(diff_sub, sigma=sigma_yx)
  median_m = float(np.nanmedian(raw_diff_m[bare_gentle_mask]))
  coarse_m = np.where(weight > 1e-4, numer / np.maximum(weight, 1e-6), median_m)
  upsampled = ndimage.zoom(
      coarse_m, (height / coarse_m.shape[0], width / coarse_m.shape[1]), order=1
  )
  return upsampled[:height, :width].astype(np.float32)


def _compute_diff_and_bare_gentle_mask(
    gem_dsm_m,
    gem_dtm_m,
    ref_dem_m,
    dy_m,
    dx_m,
    config,
):
  """Computes elevation differences and masks pixels to bare-gentle ground.

  A pixel is included in the returned mask only if all four criteria hold:
    1. Both `gem_dsm_m` and `ref_dem_m` are finite.
    2. Terrain slope of `ref_dem_m` is below `config.slope_max_deg` (avoids
       steep-terrain co-registration artifacts).
    3. Above-ground height `gem_dsm_m - gem_dtm_m` (nDSM) is finite and below
       `config.ndsm_max_m` (excludes forest canopy and buildings).
    4. Elevation difference is within `config.outlier_gate_m` of the
       bare-gentle median (rejects gross outliers such as clouds or water).

  Args:
    gem_dsm_m: 2D GEM-15 DSM elevation array in meters.
    gem_dtm_m: 2D GEM-15 DTM elevation array in meters (same shape as
      `gem_dsm_m`).
    ref_dem_m: 2D reference DEM elevation array in meters in the same vertical
      datum as `gem_dsm_m`.
    dy_m: Pixel spacing along rows (latitude) in meters.
    dx_m: Pixel spacing along columns (longitude) in meters.
    config: Destriping hyperparameters supplying mask thresholds.

  Returns:
    Tuple `(raw_diff_m, bare_gentle_mask)` where `raw_diff_m` is a 2D float32
    array of `gem_dsm_m - ref_dem_m` in meters and `bare_gentle_mask` is a 2D
    boolean array of pixels satisfying all four criteria.
  """
  # Criterion 1: both GEM-15 DSM and the reference DEM must be finite.
  valid_both = np.isfinite(gem_dsm_m) & np.isfinite(ref_dem_m)
  if not np.any(valid_both):
    return np.zeros_like(gem_dsm_m, dtype=np.float32), valid_both

  # Criterion 2: gentle terrain slope on the reference DEM (fill NaNs with the
  # median so finite-difference gradients do not leak NaNs across voids).
  ref_fill_m = np.where(
      np.isfinite(ref_dem_m), ref_dem_m, float(np.nanmedian(ref_dem_m))
  )
  grad_y, grad_x = np.gradient(ref_fill_m, dy_m, dx_m)
  slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))

  # Criterion 3: bare ground (low above-ground height, nDSM = DSM - DTM).
  ndsm_m = gem_dsm_m - gem_dtm_m
  bare_gentle = (
      valid_both
      & (slope_deg < config.slope_max_deg)
      & np.isfinite(ndsm_m)
      & (ndsm_m < config.ndsm_max_m)
  )
  raw_diff_m = (gem_dsm_m - ref_dem_m).astype(np.float32)
  if not np.any(bare_gentle):
    return raw_diff_m, bare_gentle

  # Criterion 4: reject gross outliers relative to the bare-gentle median.
  median_diff_m = float(np.nanmedian(raw_diff_m[bare_gentle]))
  clean_mask = bare_gentle & (
      np.abs(raw_diff_m - median_diff_m) < config.outlier_gate_m
  )
  return raw_diff_m, clean_mask


def _build_zero_stripe_interpolator():
  """Returns a constant 0.0 m stripe interpolator for empty bare-ground fits."""
  axis = np.array([-1.0, 1.0], dtype=np.float64)
  return interpolate.RegularGridInterpolator(
      (axis, axis),
      np.zeros((2, 2), dtype=np.float64),
      method='linear',
      bounds_error=False,
      fill_value=0.0,
  )


def _fit_orbit_stripe_interpolator(
    detrended_diff_m,
    bare_gentle_mask,
    across_track_km,
    along_track_km,
    min_bin_pixels,
    config,
):
  """Fits a 2D stripe-error interpolator in curved ALOS orbit coordinates.

  Aggregates `detrended_diff_m` over `bare_gentle_mask` by taking the median
  within each `(du_km, dv_km)` orbit bin (requiring at least `min_bin_pixels`),
  smooths along-track (`v`), removes remaining long-wavelength across-track
  drift via a high-pass filter, and lightly smooths across bin edges (`u`).

  Args:
    detrended_diff_m: 2D array of elevation differences `(gem_dsm - ref_dem)` in
      meters after subtracting the >150 km regional bias surface.
    bare_gentle_mask: 2D boolean mask selecting valid bare, gentle, non-outlier
      pixels (same shape as `detrended_diff_m`).
    across_track_km: 2D array of pixel across-track coordinates (`u`,
      perpendicular to the ALOS track) in kilometers.
    along_track_km: 2D array of pixel along-track coordinates (`v`, parallel to
      the ALOS track) in kilometers.
    min_bin_pixels: Minimum number of valid bare-gentle pixels required in a
      single `(du_km, dv_km)` bin to keep its median; sparser bins are ignored.
    config: Destriping hyperparameters controlling bin sizes, smoothing scales,
      high-pass window, and minimum Gaussian support weights.

  Returns:
    `RegularGridInterpolator` queried at `(along_track_km, across_track_km)`
    returning the estimated stripe error in meters (0.0 m outside support).
  """
  if int(np.sum(bare_gentle_mask)) < min_bin_pixels:
    return _build_zero_stripe_interpolator()

  du_km, dv_km = config.du_km, config.dv_km
  u_min = math.floor(float(across_track_km.min()) / du_km) * du_km
  u_max = math.ceil(float(across_track_km.max()) / du_km) * du_km
  v_min = math.floor(float(along_track_km.min()) / dv_km) * dv_km
  v_max = math.ceil(float(along_track_km.max()) / dv_km) * dv_km
  num_u = max(4, int(round((u_max - u_min) / du_km)))
  num_v = max(2, int(round((v_max - v_min) / dv_km)))

  idx_u = np.clip(
      ((across_track_km[bare_gentle_mask] - u_min) / du_km).astype(int),
      0,
      num_u - 1,
  )
  idx_v = np.clip(
      ((along_track_km[bare_gentle_mask] - v_min) / dv_km).astype(int),
      0,
      num_v - 1,
  )
  flat_idx = idx_v * num_u + idx_u
  order = np.argsort(flat_idx)
  unique_cells, split_pos = np.unique(flat_idx[order], return_index=True)

  cell_med_m = np.full((num_v, num_u), np.nan, dtype=np.float64)
  for cell_id, group_vals in zip(
      unique_cells,
      np.split(detrended_diff_m[bare_gentle_mask][order], split_pos[1:]),
      strict=True,
  ):
    if group_vals.size >= min_bin_pixels:
      cell_med_m[cell_id // num_u, cell_id % num_u] = np.median(group_vals)

  if not np.any(np.isfinite(cell_med_m)):
    return _build_zero_stripe_interpolator()

  # 1. Smooth along-track (v) across 35 km bins.
  raw_valid = np.isfinite(cell_med_m).astype(np.float64)
  raw_vals_m = np.where(np.isfinite(cell_med_m), cell_med_m, 0.0)
  sig_v_bins = config.sigma_v_km / dv_km
  num_v_sm = ndimage.gaussian_filter1d(
      raw_vals_m, sigma=sig_v_bins, axis=0, mode='constant', cval=0.0
  )
  den_v_sm = ndimage.gaussian_filter1d(
      raw_valid, sigma=sig_v_bins, axis=0, mode='constant', cval=0.0
  )
  valid_uv = (den_v_sm > config.min_v_weight).astype(np.float64)
  smoothed_v_m = np.where(
      valid_uv > 0, num_v_sm / np.maximum(den_v_sm, 1e-5), 0.0
  )

  # 2. High-pass detrend (>148 km) and lightly smooth across 4 km bin edges.
  win_hp_bins = max(3, int(round(config.win_hp_km / du_km)))
  if win_hp_bins % 2 == 0:
    win_hp_bins += 1
  sig_u_bins = config.sigma_u_km / du_km
  # `smoothed_v_m` is already 0 wherever `valid_uv` is 0.
  lowpass_m = ndimage.uniform_filter1d(
      smoothed_v_m, size=win_hp_bins, axis=1, mode='nearest'
  ) / np.maximum(
      ndimage.uniform_filter1d(
          valid_uv, size=win_hp_bins, axis=1, mode='nearest'
      ),
      1e-5,
  )
  highpass_m = (smoothed_v_m - lowpass_m) * valid_uv
  num_bp = ndimage.gaussian_filter1d(
      highpass_m, sigma=sig_u_bins, axis=1, mode='constant', cval=0.0
  )
  den_bp = ndimage.gaussian_filter1d(
      valid_uv, sigma=sig_u_bins, axis=1, mode='constant', cval=0.0
  )
  stripe_grid_m = (
      np.where(
          den_bp > config.min_u_weight, num_bp / np.maximum(den_bp, 1e-5), 0.0
      )
      * valid_uv
  )

  u_centers_km = u_min + (np.arange(num_u) + 0.5) * du_km
  v_centers_km = v_min + (np.arange(num_v) + 0.5) * dv_km
  return interpolate.RegularGridInterpolator(
      (v_centers_km, u_centers_km),
      stripe_grid_m,
      method='linear',
      bounds_error=False,
      fill_value=0.0,
  )


def _fit_local_stripe_field(
    gem_dsm_m,
    gem_dtm_m,
    ref_dem_m,
    bbox_deg,
    config,
):
  """Fits a local ALOS stripe evaluator `S(lon, lat)` over a single window.

  Args:
    gem_dsm_m: 2D GEM-15 DSM elevation array in meters (north-up).
    gem_dtm_m: 2D GEM-15 DTM elevation array in meters (same shape as
      `gem_dsm_m`).
    ref_dem_m: 2D coarse reference DEM array in meters in the same vertical
      datum as `gem_dsm_m`.
    bbox_deg: Geographic bounds `(west, south, east, north)` of the window in
      WGS84 degrees.
    config: Destriping hyperparameters.

  Returns:
    Callable `eval_stripe_m(lon_deg, lat_deg) -> stripe_m` evaluating the fitted
    stripe surface in meters (float32).
  """
  west_deg, south_deg, east_deg, north_deg = bbox_deg
  height, width = ref_dem_m.shape
  lon0_deg = 0.5 * (west_deg + east_deg)
  lat0_deg = 0.5 * (south_deg + north_deg)
  ref_equator_lon_deg = lon0_deg - float(
      np.interp(lat0_deg, _LAT_LUT_DEG, _LON_TRACK_LUT_DEG)
  )

  dy_m = ((north_deg - south_deg) / height) * _METERS_PER_DEG_LAT
  dx_m = (
      ((east_deg - west_deg) / width)
      * _METERS_PER_DEG_LAT
      * max(0.15, math.cos(math.radians(lat0_deg)))
  )
  px_area_km2 = (dy_m * dx_m) * 1e-6
  min_area_km2 = (
      (config.min_cell_coverage_pct / 100.0) * config.du_km * config.dv_km
  )
  min_bin_pixels = max(5, int(round(min_area_km2 / max(px_area_km2, 1e-4))))

  raw_diff_m, bare_gentle_mask = _compute_diff_and_bare_gentle_mask(
      gem_dsm_m, gem_dtm_m, ref_dem_m, dy_m, dx_m, config
  )
  regional_bias_m = _estimate_regional_bias_m(
      raw_diff_m,
      bare_gentle_mask,
      config.sigma_bias_km,
      dy_m * 1e-3,
      dx_m * 1e-3,
  )
  detrended_diff_m = raw_diff_m - regional_bias_m

  lats_deg = np.linspace(
      north_deg - (north_deg - south_deg) / (2.0 * height),
      south_deg + (north_deg - south_deg) / (2.0 * height),
      height,
  )
  lons_deg = np.linspace(
      west_deg + (east_deg - west_deg) / (2.0 * width),
      east_deg - (east_deg - west_deg) / (2.0 * width),
      width,
  )
  lon2d_deg, lat2d_deg = np.meshgrid(lons_deg, lats_deg)
  across_track_km, along_track_km = _project_lonlat_to_alos_track_km(
      lon2d_deg, lat2d_deg, ref_equator_lon_deg
  )
  stripe_interpolator = _fit_orbit_stripe_interpolator(
      detrended_diff_m,
      bare_gentle_mask,
      across_track_km,
      along_track_km,
      min_bin_pixels,
      config,
  )

  def _eval_stripe_m(
      target_lon_deg, target_lat_deg
  ):
    target_across_km, target_along_km = _project_lonlat_to_alos_track_km(
        target_lon_deg, target_lat_deg, ref_equator_lon_deg
    )
    return stripe_interpolator((target_along_km, target_across_km)).astype(
        np.float32
    )

  return _eval_stripe_m


def _smoothstep_tent_weight(normalized_offset):
  """Evaluates C1 smoothstep tent blending weights in [0, 1].

  Args:
    normalized_offset: Signed distance from a local window center divided by
      tile size; weights are positive for `|normalized_offset| < 1` and 0
      outside.

  Returns:
    Array of blending weights `t^2 * (3 - 2*t)` where
    `t = clip(1 - |normalized_offset|, 0, 1)`.
  """
  t = np.clip(1.0 - np.abs(normalized_offset), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def compute_stripe_field(
    gem_dsm_m,
    gem_dtm_m,
    ref_dem_m,
    bbox_deg,
    tile_bbox_deg = None,
    halo_deg = 1.0,
    config = DestripingConfig(),
):
  """Fits the ALOS stripe field S(lon, lat) on a north-up EPSG:4326 grid.

  All three input elevation surfaces must share a common vertical datum (geoid
  or ellipsoid); the caller is responsible for any vertical datum conversion
  before calling this function.

  When `tile_bbox_deg` is provided (e.g., a 1x1 deg target tile inside a 5x5
  deg `bbox_deg` with `halo_deg=1.0`), fits the 3x3 grid of overlapping 3x3 deg
  windows around `tile_bbox_deg` and blends them with a C1 smoothstep weighted
  mean so adjacent tiles share identical values along their borders.

  Args:
    gem_dsm_m: 2D GEM-15 DSM elevation array in meters (north-up).
    gem_dtm_m: 2D GEM-15 DTM elevation array in meters (matching gem_dsm_m).
    ref_dem_m: 2D coarse reference DEM in meters in the same vertical datum.
    bbox_deg: Outer mosaic `(west, south, east, north)` in WGS84 degrees.
    tile_bbox_deg: Optional target tile `(west, south, east, north)` in degrees
      for cross-tile weighted blending across the 3x3 local neighborhood. When
      `None`, fits a single local stripe field over `bbox_deg`.
    halo_deg: Context halo in degrees around each local fit when `tile_bbox_deg`
      is specified.
    config: Destriping hyperparameters.

  Returns:
    Callable `eval_stripe_m(lon_deg, lat_deg) -> stripe_m` in meters (float32).

  Raises:
    ValueError: If `config.min_cell_coverage_pct` is outside `(0, 100]`, the
      input arrays have mismatched shapes, `tile_bbox_deg` or `halo_deg` is
      invalid, or `bbox_deg` does not cover the required 3x3 local windows.
  """
  if not 0.0 < config.min_cell_coverage_pct <= 100.0:
    raise ValueError(
        'min_cell_coverage_pct must be in (0, 100]; got'
        f' {config.min_cell_coverage_pct}.'
    )
  if gem_dsm_m.shape != gem_dtm_m.shape or gem_dsm_m.shape != ref_dem_m.shape:
    raise ValueError(
        'gem_dsm_m, gem_dtm_m, and ref_dem_m must share identical 2D shapes;'
        f' got {gem_dsm_m.shape}, {gem_dtm_m.shape}, {ref_dem_m.shape}.'
    )

  if tile_bbox_deg is None:
    return _fit_local_stripe_field(
        gem_dsm_m, gem_dtm_m, ref_dem_m, bbox_deg, config
    )

  west, south, east, north = bbox_deg
  tw, ts, te, tn = tile_bbox_deg
  tile_w, tile_h = te - tw, tn - ts
  if tile_w <= 0.0 or tile_h <= 0.0 or halo_deg < 0.0:
    raise ValueError(
        f'Invalid tile_bbox_deg={tile_bbox_deg} or halo_deg={halo_deg}.'
    )
  req_w, req_s = tw - tile_w - halo_deg, ts - tile_h - halo_deg
  req_e, req_n = te + tile_w + halo_deg, tn + tile_h + halo_deg
  if (
      west > req_w + 1e-6
      or south > req_s + 1e-6
      or east < req_e - 1e-6
      or north < req_n - 1e-6
  ):
    raise ValueError(
        f'bbox_deg={bbox_deg} must cover ({req_w}, {req_s}, {req_e}, {req_n}).'
    )

  height, width = ref_dem_m.shape
  px_per_deg_x = width / (east - west)
  px_per_deg_y = height / (north - south)

  # 3. Fit 3x3 overlapping local windows and blend via smoothstep weighted mean.
  evaluators: list[tuple[float, float, StripeField]] = []
  for dj in (-1, 0, 1):
    for di in (-1, 0, 1):
      sub_w = tw + di * tile_w - halo_deg
      sub_e = te + di * tile_w + halo_deg
      sub_s = ts + dj * tile_h - halo_deg
      sub_n = tn + dj * tile_h + halo_deg

      c0 = int(round((sub_w - west) * px_per_deg_x))
      c1 = int(round((sub_e - west) * px_per_deg_x))
      r0 = int(round((north - sub_n) * px_per_deg_y))
      r1 = int(round((north - sub_s) * px_per_deg_y))
      sub_bbox = (
          west + c0 / px_per_deg_x,
          north - r1 / px_per_deg_y,
          west + c1 / px_per_deg_x,
          north - r0 / px_per_deg_y,
      )
      fn = _fit_local_stripe_field(
          gem_dsm_m[r0:r1, c0:c1],
          gem_dtm_m[r0:r1, c0:c1],
          ref_dem_m[r0:r1, c0:c1],
          sub_bbox,
          config,
      )
      evaluators.append(
          (0.5 * (tw + te) + di * tile_w, 0.5 * (ts + tn) + dj * tile_h, fn)
      )

  def _eval_blended_m(
      target_lon_deg, target_lat_deg
  ):
    lon_arr = np.asarray(target_lon_deg, dtype=np.float64)
    lat_arr = np.asarray(target_lat_deg, dtype=np.float64)
    numer = np.zeros_like(lon_arr, dtype=np.float64)
    denom = np.zeros_like(lon_arr, dtype=np.float64)
    for cx, cy, fn in evaluators:
      w = _smoothstep_tent_weight((lon_arr - cx) / tile_w) * (
          _smoothstep_tent_weight((lat_arr - cy) / tile_h)
      )
      if not np.any(w > 0.0):
        continue
      numer += w * fn(lon_arr, lat_arr)
      denom += w
    return np.where(denom > 1e-8, numer / np.maximum(denom, 1e-8), 0.0).astype(
        np.float32
    )

  return _eval_blended_m


def destripe_elevation_bands(
    bands_m,
    lon_deg,
    lat_deg,
    eval_stripe_m,
):
  """Subtracts the fitted stripe surface S(lon, lat) across GEM-15 bands.

  Input bands may use any vertical datum (consistent with the surfaces passed
  to `compute_stripe_field`); output bands remain in that same vertical datum.

  Args:
    bands_m: Mapping from GEM-15 band name (e.g., 'dsm', 'dsm_percentile_5',
      'dsm_percentile_50', 'dsm_percentile_95', 'dtm', 'dsm_std') to 2D float
      arrays in meters.
    lon_deg: 2D longitude array in degrees matching the target tile pixels.
    lat_deg: 2D latitude array in degrees matching the target tile pixels.
    eval_stripe_m: Stripe evaluator returned by `compute_stripe_field`.

  Returns:
    Dictionary of destriped float32 arrays with `dtm <= dsm` enforced on finite
    pixels and non-elevation bands (such as `dsm_std`) preserved unchanged.
  """
  stripe_m = eval_stripe_m(lon_deg, lat_deg).astype(np.float32)

  corrected: dict[str, np.ndarray] = {}
  for name, arr in bands_m.items():
    out = np.asarray(arr, dtype=np.float32).copy()
    if name in ELEVATION_BANDS:
      out = out - stripe_m
    corrected[name] = out

  if 'dsm' in corrected and 'dtm' in corrected:
    valid = np.isfinite(corrected['dsm']) & np.isfinite(corrected['dtm'])
    corrected['dtm'] = np.where(
        valid,
        np.minimum(corrected['dtm'], corrected['dsm']),
        corrected['dtm'],
    )
  return corrected

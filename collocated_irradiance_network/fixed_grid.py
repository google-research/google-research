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

"""GOES ABI fixed-grid projection, geolocation, and Earth-central angle cut.

Constructs the GOES East 2 km geostationary pyresample AreaDefinition on the
GRS80 ellipsoid, computes geodetic (lats, lngs) and the spherical haversine
Earth-central angle mask (< 1.14 rad), and caches GridGeometry instances in
process via build_goes_east_grid.
"""

import dataclasses
import functools
import threading

import numpy as np
import pyresample

from collocated_irradiance_network import coin_constants

ArrayOrFloat = float | np.ndarray
_GRID_CACHE_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class GridGeometry:
  """Precomputed geolocation and in-range mask for a geostationary grid."""

  lats: np.ndarray
  lngs: np.ndarray
  in_range_mask: np.ndarray
  n_in_range: int
  area_def: pyresample.geometry.AreaDefinition


def normalize_lat_lng(
    latitude_degrees, longitude_degrees
):
  """Normalizes latitudes to [-90, 90] and longitudes to [-180, 180].

  Args:
    latitude_degrees: Latitude(s) in degrees.
    longitude_degrees: Longitude(s) in degrees.

  Returns:
    Tuple of (normalized_lat, normalized_lng) as float64 ndarrays.
  """
  lat_arr = np.asarray(latitude_degrees, dtype=np.float64)
  lng_arr = np.asarray(longitude_degrees, dtype=np.float64)
  normalized_lat = np.maximum(-90.0, np.minimum(90.0, lat_arr))
  normalized_lng = np.where(
      np.logical_and(-180.0 <= lng_arr, lng_arr <= 180.0),
      lng_arr,
      np.mod(lng_arr + 180.0, 360.0) - 180.0,
  )
  return normalized_lat, normalized_lng


def earth_central_angle_radians(
    lat1_degrees,
    lng1_degrees,
    lat2_degrees,
    lng2_degrees,
):
  """Returns element-wise Earth-central angle in radians via haversine formula.

  Args:
    lat1_degrees: First latitude(s) in degrees.
    lng1_degrees: First longitude(s) in degrees.
    lat2_degrees: Second latitude(s) in degrees.
    lng2_degrees: Second longitude(s) in degrees.
  """
  lat1_norm, lng1_norm = normalize_lat_lng(lat1_degrees, lng1_degrees)
  lat2_norm, lng2_norm = normalize_lat_lng(lat2_degrees, lng2_degrees)
  non_scalar_shapes = {
      arr.shape
      for arr in (lat1_norm, lng1_norm, lat2_norm, lng2_norm)
      if arr.ndim > 0
  }
  if len(non_scalar_shapes) > 1:
    raise ValueError(
        'All non-scalar coordinate arrays must share one shape for '
        'element-wise earth_central_angle_radians, got '
        f'{sorted(non_scalar_shapes)}'
    )

  lats1 = np.deg2rad(lat1_norm)
  lngs1 = np.deg2rad(lng1_norm)
  lats2 = np.deg2rad(lat2_norm)
  lngs2 = np.deg2rad(lng2_norm)
  dlats = np.sin(0.5 * (lats2 - lats1))
  dlngs = np.sin(0.5 * (lngs2 - lngs1))
  x = dlats * dlats + dlngs * dlngs * np.cos(lats1) * np.cos(lats2)
  return 2.0 * np.arcsin(np.sqrt(np.minimum(1.0, x)))


def area_extent_from_image_bounds(
    x_image_bounds,
    y_image_bounds,
    perspective_point_height,
):
  """Computes pyresample area_extent in meters from ABI image_bounds in radians.

  Note that ABI y_image_bounds is ordered [north_max, south_min].

  Args:
    x_image_bounds: 2-element array [west_min_rad, east_max_rad].
    y_image_bounds: 2-element array [north_max_rad, south_min_rad].
    perspective_point_height: Geostationary altitude `h` in meters.

  Returns:
    Tuple `(x_min, y_min, x_max, y_max)` in projection meters.
  """
  x_bounds = np.asarray(x_image_bounds, dtype=np.float64).reshape(-1)
  y_bounds = np.asarray(y_image_bounds, dtype=np.float64).reshape(-1)
  if x_bounds.shape != (2,) or y_bounds.shape != (2,):
    raise ValueError(
        'x_image_bounds and y_image_bounds must have shape (2,), got '
        f'{x_bounds.shape} and {y_bounds.shape}'
    )
  if not (np.all(np.isfinite(x_bounds)) and np.all(np.isfinite(y_bounds))):
    raise ValueError(
        'x_image_bounds and y_image_bounds must be finite, got '
        f'{x_bounds} and {y_bounds}'
    )
  if not (x_bounds[0] < x_bounds[1]):
    raise ValueError(
        f'Expected x_image_bounds[0] < x_image_bounds[1], got {x_bounds}'
    )
  if not (y_bounds[0] > y_bounds[1]):
    raise ValueError(
        'Expected y_image_bounds[0] > y_image_bounds[1] (north first), '
        f'got {y_bounds}'
    )
  h = float(perspective_point_height)
  if not np.isfinite(h) or h <= 0.0:
    raise ValueError(
        f'perspective_point_height must be finite and positive, got {h}'
    )
  return (
      float(x_bounds[0] * h),
      float(y_bounds[1] * h),
      float(x_bounds[1] * h),
      float(y_bounds[0] * h),
  )


def build_area_def(
    satellite_name,
    shape,
    area_extent_meters,
):
  """Constructs a pyresample AreaDefinition for a geostationary fixed grid.

  Args:
    satellite_name: Satellite identifier (currently only
      `coin_constants.SATELLITE_NAME`, i.e. `'goes'`).
    shape: (rows, cols), e.g. (5424, 5424).
    area_extent_meters: (x_min, y_min, x_max, y_max) in geostationary meters.

  Returns:
    Configured `pyresample.geometry.AreaDefinition`.

  Raises:
    NotImplementedError: If `satellite_name` is not supported.
  """
  if satellite_name != coin_constants.SATELLITE_NAME:
    raise NotImplementedError(
        f'Unsupported satellite_name {satellite_name!r}; only '
        f'{coin_constants.SATELLITE_NAME!r} is currently supported.'
    )
  rows, cols = shape
  return pyresample.geometry.AreaDefinition(
      area_id='goes_east_abi_fixed_grid',
      description='GOES East ABI fixed grid',
      proj_id='geos_goes_east',
      projection=dict(coin_constants.GOES_PROJ_DICT),
      width=cols,
      height=rows,
      area_extent=area_extent_meters,
  )


def compute_grid_geometry(
    area_def,
    subsat_lat_degrees,
    subsat_lng_degrees,
    max_earth_central_angle_radians,
):
  """Computes read-only `(lats, lngs, in_range_mask)` for `area_def`.

  Args:
    area_def: Target geostationary AreaDefinition.
    subsat_lat_degrees: Subsatellite latitude in degrees (0.0 for GOES East).
    subsat_lng_degrees: Subsatellite longitude in degrees (-75.0 for GOES East).
    max_earth_central_angle_radians: Maximum Earth-central angle in radians
      (1.14 for COIN).

  Returns:
    `GridGeometry` containing geodetic coordinates and the in-range mask.
  """
  if (
      not np.isfinite(max_earth_central_angle_radians)
      or max_earth_central_angle_radians <= 0.0
  ):
    raise ValueError(
        'max_earth_central_angle_radians must be finite and strictly positive, '
        f'got {max_earth_central_angle_radians}'
    )

  raw_lngs, raw_lats = area_def.get_lonlats()
  raw_lats = np.asarray(raw_lats, dtype=np.float64)
  raw_lngs = np.asarray(raw_lngs, dtype=np.float64)

  on_earth = (
      np.isfinite(raw_lats)
      & np.isfinite(raw_lngs)
      & (np.abs(raw_lats) <= 90.0)
      & (np.abs(raw_lngs) <= 180.0)
  )

  lats = np.where(on_earth, raw_lats, np.nan)
  lngs = np.where(on_earth, raw_lngs, np.nan)

  in_range_mask = np.zeros(raw_lats.shape, dtype=bool)
  if np.any(on_earth):
    valid_lats = raw_lats[on_earth]
    valid_lngs = raw_lngs[on_earth]
    distances = earth_central_angle_radians(
        subsat_lat_degrees, subsat_lng_degrees, valid_lats, valid_lngs
    )
    in_range_mask[on_earth] = distances < max_earth_central_angle_radians

  n_in_range = int(np.sum(in_range_mask))

  lats.setflags(write=False)
  lngs.setflags(write=False)
  in_range_mask.setflags(write=False)

  return GridGeometry(
      lats=lats,
      lngs=lngs,
      in_range_mask=in_range_mask,
      n_in_range=n_in_range,
      area_def=area_def,
  )


@functools.lru_cache(maxsize=1)
def _cached_goes_east_grid(
    shape,
    area_extent_meters,
):
  """Caches the active GOES East `GridGeometry` (~500 MB at full disk)."""
  area_def = build_area_def(
      coin_constants.SATELLITE_NAME, shape, area_extent_meters
  )
  return compute_grid_geometry(
      area_def,
      coin_constants.SUBSATELLITE_LAT_DEGREES,
      coin_constants.SUBSATELLITE_LNG_DEGREES,
      coin_constants.MAX_EARTH_CENTRAL_ANGLE_RADIANS,
  )


def clear_grid_geometry_cache():
  """Clears the module-level `GridGeometry` cache."""
  with _GRID_CACHE_LOCK:
    _cached_goes_east_grid.cache_clear()


def build_goes_east_grid(
    shape,
    area_extent_meters,
):
  """Returns a cached or computed GOES East `GridGeometry` for `shape`."""
  rounded_extent = (
      round(float(area_extent_meters[0]), 6),
      round(float(area_extent_meters[1]), 6),
      round(float(area_extent_meters[2]), 6),
      round(float(area_extent_meters[3]), 6),
  )
  with _GRID_CACHE_LOCK:
    return _cached_goes_east_grid(
        (int(shape[0]), int(shape[1])),
        rounded_extent,
    )

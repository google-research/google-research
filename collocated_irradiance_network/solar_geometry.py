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

"""Solar zenith, azimuth, and day-of-year calculations for COIN inference.

Implements the analytical solar declination, Equation of Time, hour angle,
elevation, zenith, North-clockwise [0, 2*pi] azimuth, and cos(day_of_year)
equations matching the COIN training pipeline, with an explicit denominator
guard at the subsolar point and geographic poles and clean NaN propagation.
"""

import datetime

import numpy as np

from collocated_irradiance_network import coin_constants

ArrayOrFloat = float | np.ndarray


def _check_elementwise_shapes(*arrays):
  """Rejects mismatched non-scalar shapes to prevent silent outer products."""
  shapes = {arr.shape for arr in arrays if arr.ndim > 0}
  if len(shapes) > 1:
    raise ValueError(
        'Non-scalar inputs must share one shape for element-wise solar '
        f'geometry, got {sorted(shapes)}'
    )


def validate_utc_datetime(dt):
  """Validates that `dt` is timezone-aware UTC and returns float64 timestamp."""
  if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
    raise ValueError(
        f'Naive datetime is not permitted; must be timezone-aware UTC: {dt!r}'
    )
  offset = dt.tzinfo.utcoffset(dt)
  if offset != datetime.timedelta(0):
    raise ValueError(
        f'Datetime must be in UTC (offset 0), got offset {offset}: {dt!r}'
    )
  return float(dt.timestamp())


def declination_angle_radians(day):
  """Returns solar declination angle in radians for eccentric Earth orbit.

  Args:
    day: Days since Unix epoch (utc_timestamp / 86400.0).
  """
  day_arr = np.asarray(day, dtype=np.float64)
  return np.arcsin(
      np.sin(np.deg2rad(-23.44))
      * np.cos(
          coin_constants.RADIANS_PER_YEAR * (day_arr + 10.0)
          + 2.0
          * 0.0167
          * np.sin(coin_constants.RADIANS_PER_YEAR * (day_arr - 2.0))
      )
  )


def hour_angle_radians(
    lng_degrees, utc_timestamp
):
  """Returns solar hour angle in radians [-pi, pi], including Equation of Time.

  Args:
    lng_degrees: Longitude(s) in degrees East [-180, 180].
    utc_timestamp: UTC timestamp(s) in seconds since Unix epoch (float64).
  """
  lng_arr = np.asarray(lng_degrees, dtype=np.float64)
  ts = np.asarray(utc_timestamp, dtype=np.float64)
  _check_elementwise_shapes(lng_arr, ts)
  minute = ts / 60.0
  day = ts / coin_constants.SECONDS_PER_DAY
  eot_offset = coin_constants.RADIANS_PER_YEAR * (day - 81.0)
  eot = (
      9.87 * np.sin(2.0 * eot_offset)
      - 7.53 * np.cos(eot_offset)
      - 1.5 * np.sin(eot_offset)
  )
  return (
      np.mod(
          minute + (lng_arr * coin_constants.MINUTES_PER_DAY / 360.0) + eot,
          coin_constants.MINUTES_PER_DAY,
      )
      * coin_constants.RADIANS_PER_MINUTE
      - np.pi
  )


def elevation_angle_radians(
    lat_degrees,
    lng_degrees,
    utc_timestamp,
):
  """Returns solar elevation (altitude) angle in radians [-pi/2, pi/2].

  Args:
    lat_degrees: Latitude(s) in degrees North [-90, 90].
    lng_degrees: Longitude(s) in degrees East [-180, 180].
    utc_timestamp: UTC timestamp(s) in seconds since Unix epoch (float64).
  """
  lat_arr = np.asarray(lat_degrees, dtype=np.float64)
  lng_arr = np.asarray(lng_degrees, dtype=np.float64)
  ts = np.asarray(utc_timestamp, dtype=np.float64)
  _check_elementwise_shapes(lat_arr, lng_arr, ts)
  day = ts / coin_constants.SECONDS_PER_DAY
  declination_angle = declination_angle_radians(day)
  lat_rad = np.deg2rad(lat_arr)
  sin_el = np.sin(lat_rad) * np.sin(declination_angle) + (
      np.cos(lat_rad)
      * np.cos(declination_angle)
      * np.cos(hour_angle_radians(lng_arr, ts))
  )
  return np.arcsin(np.clip(sin_el, -1.0, 1.0))


def zenith_angle_degrees(
    lat_degrees,
    lng_degrees,
    utc_timestamp,
):
  """Returns solar zenith angle in degrees [0, 180].

  Args:
    lat_degrees: Latitude(s) in degrees North [-90, 90].
    lng_degrees: Longitude(s) in degrees East [-180, 180].
    utc_timestamp: UTC timestamp(s) in seconds since Unix epoch (float64).
  """
  return 90.0 - np.rad2deg(
      elevation_angle_radians(lat_degrees, lng_degrees, utc_timestamp)
  )


def azimuth_angle_radians(
    lat_degrees,
    lng_degrees,
    utc_timestamp,
):
  """Returns solar azimuth angle in radians in [0, 2*pi] clockwise from North.

  Includes an explicit denominator guard at the subsolar point (zenith == 0)
  and geographic poles while propagating non-finite inputs as NaN.

  Args:
    lat_degrees: Latitude(s) in degrees North [-90, 90].
    lng_degrees: Longitude(s) in degrees East [-180, 180].
    utc_timestamp: UTC timestamp(s) in seconds since Unix epoch (float64).
  """
  lat_arr = np.asarray(lat_degrees, dtype=np.float64)
  lng_arr = np.asarray(lng_degrees, dtype=np.float64)
  ts = np.asarray(utc_timestamp, dtype=np.float64)
  _check_elementwise_shapes(lat_arr, lng_arr, ts)
  day = ts / coin_constants.SECONDS_PER_DAY
  declination = declination_angle_radians(day)
  zenith_radians = np.deg2rad(zenith_angle_degrees(lat_arr, lng_arr, ts))
  lat_rad = np.deg2rad(lat_arr)
  numerator = np.sin(declination) - np.cos(zenith_radians) * np.sin(lat_rad)
  denominator = np.sin(zenith_radians) * np.cos(lat_rad)
  hour_angle = hour_angle_radians(lng_arr, ts)
  inputs_finite = (
      np.isfinite(numerator)
      & np.isfinite(denominator)
      & np.isfinite(hour_angle)
  )
  denom_valid = inputs_finite & (
      np.abs(denominator) > coin_constants.AZIMUTH_DENOMINATOR_EPSILON
  )
  safe_denominator = np.where(denom_valid, denominator, 1.0)
  ratio = np.where(denom_valid, numerator / safe_denominator, 1.0)
  azimuth_angle = np.arccos(np.clip(ratio, -1.0, 1.0))
  oriented_azimuth = np.where(
      hour_angle > 0.0, 2.0 * np.pi - azimuth_angle, azimuth_angle
  )
  return np.where(inputs_finite, oriented_azimuth, np.nan)


def cos_day_of_year(utc_timestamp):
  """Returns cosine of the fractional day of the year as float32.

  Evaluates `cos(2 * pi * ((utc_timestamp / 86400) mod 365.2425) / 365.2425)`
  in float64 precision and casts the final cosine to float32.

  Args:
    utc_timestamp: UTC timestamp(s) in seconds since Unix epoch.
  """
  ts = np.asarray(utc_timestamp, dtype=np.float64)
  day_mod = np.mod(
      ts / coin_constants.SECONDS_PER_DAY,
      coin_constants.DAYS_PER_YEAR_FOR_COS_DOY,
  )
  cos_val = np.cos(
      2.0 * np.pi * day_mod / coin_constants.DAYS_PER_YEAR_FOR_COS_DOY
  )
  return np.asarray(cos_val, dtype=np.float32)

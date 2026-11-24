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

"""Unit tests for solar_geometry."""

import datetime
import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from collocated_irradiance_network import coin_constants
from collocated_irradiance_network import solar_geometry


def _utc_timestamp(
    year,
    month,
    day,
    hour,
    minute,
    second,
):
  return datetime.datetime(
      year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc
  ).timestamp()


class SolarGeometryTest(parameterized.TestCase):

  def test_validate_utc_datetime_accepts_utc(self):
    dt = datetime.datetime(
        2019, 6, 1, 17, 0, 44, 700000, tzinfo=datetime.timezone.utc
    )
    self.assertEqual(solar_geometry.validate_utc_datetime(dt), dt.timestamp())

  @parameterized.named_parameters(
      dict(
          testcase_name='naive',
          dt=datetime.datetime(2019, 6, 1, 17, 0, 0),
          expected_regex='Naive datetime',
      ),
      dict(
          testcase_name='non_utc_offset',
          dt=datetime.datetime(
              2019,
              6,
              1,
              12,
              0,
              0,
              tzinfo=datetime.timezone(datetime.timedelta(hours=-5)),
          ),
          expected_regex='offset 0',
      ),
  )
  def test_validate_utc_datetime_rejects_invalid(
      self, dt, expected_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      solar_geometry.validate_utc_datetime(dt)

  @parameterized.named_parameters(
      dict(
          testcase_name='march_equinox',
          ts=_utc_timestamp(2021, 3, 20, 12, 0, 0),
          expected_deg=0.0,
          atol_deg=1.5,
      ),
      dict(
          testcase_name='june_solstice',
          ts=_utc_timestamp(2021, 6, 21, 12, 0, 0),
          expected_deg=23.44,
          atol_deg=0.5,
      ),
      dict(
          testcase_name='september_equinox',
          ts=_utc_timestamp(2021, 9, 22, 12, 0, 0),
          expected_deg=0.0,
          atol_deg=1.5,
      ),
      dict(
          testcase_name='december_solstice',
          ts=_utc_timestamp(2021, 12, 21, 12, 0, 0),
          expected_deg=-23.44,
          atol_deg=0.5,
      ),
  )
  def test_declination_angle_seasonal_anchors(
      self, ts, expected_deg, atol_deg
  ):
    decl_deg = np.rad2deg(
        solar_geometry.declination_angle_radians(
            ts / coin_constants.SECONDS_PER_DAY
        )
    )
    np.testing.assert_allclose(decl_deg, expected_deg, atol=atol_deg)

  def test_hour_angle_range_and_morning_afternoon_sign(self):
    # Prime meridian (lng=0): 06:00 UTC is morning (<0), 18:00 UTC is afternoon.
    morning_ts = _utc_timestamp(2021, 6, 15, 6, 0, 0)
    afternoon_ts = _utc_timestamp(2021, 6, 15, 18, 0, 0)
    ha_morning = float(solar_geometry.hour_angle_radians(0.0, morning_ts))
    ha_afternoon = float(solar_geometry.hour_angle_radians(0.0, afternoon_ts))
    self.assertGreaterEqual(ha_morning, -np.pi)
    self.assertLess(ha_morning, 0.0)
    self.assertGreater(ha_afternoon, 0.0)
    self.assertLessEqual(ha_afternoon, np.pi)

  @parameterized.named_parameters(
      dict(
          testcase_name='san_mateo_sunrise',
          # 2021-08-16 06:27 PDT (UTC-7) = 13:27 UTC.
          ts=_utc_timestamp(2021, 8, 16, 13, 27, 0),
          lat=37.5630,
          lng=-122.3255,
      ),
      dict(
          testcase_name='sydney_sunset',
          # 2010-06-19 16:53 AEST (UTC+10) = 06:53 UTC.
          ts=_utc_timestamp(2010, 6, 19, 6, 53, 0),
          lat=-33.8688,
          lng=151.2093,
      ),
  )
  def test_horizon_zenith_and_elevation_angles(
      self, ts, lat, lng
  ):
    sza = solar_geometry.zenith_angle_degrees(lat, lng, ts)
    elev = solar_geometry.elevation_angle_radians(lat, lng, ts)
    np.testing.assert_allclose(sza, 90.0, atol=1.0)
    np.testing.assert_allclose(elev, 0.0, atol=np.deg2rad(1.0))
    np.testing.assert_allclose(sza, 90.0 - np.rad2deg(elev), atol=1e-12)

  def test_zenith_angle_invariant_to_longitude_wrap(self):
    ts = 1648848614.0
    sza_unwrapped = solar_geometry.zenith_angle_degrees(18.5, 180.5, ts)
    sza_wrapped = solar_geometry.zenith_angle_degrees(18.5, -179.5, ts)
    np.testing.assert_allclose(sza_unwrapped, sza_wrapped, atol=1e-12)

  @parameterized.named_parameters(
      dict(
          testcase_name='equinox_sunrise_east',
          # San Mateo 2022-09-22 06:57 PDT (UTC-7) = 13:57 UTC.
          ts=_utc_timestamp(2022, 9, 22, 13, 57, 0),
          lat=37.5630,
          lng=-122.3255,
          expected_az_deg=90.0,
      ),
      dict(
          testcase_name='equinox_sunset_west',
          # San Mateo 2022-09-22 19:06 PDT (UTC-7) = 2022-09-23 02:06 UTC.
          ts=_utc_timestamp(2022, 9, 23, 2, 6, 0),
          lat=37.5630,
          lng=-122.3255,
          expected_az_deg=270.0,
      ),
  )
  def test_azimuth_angle_equinox_sunrise_and_sunset(
      self, ts, lat, lng, expected_az_deg
  ):
    az_deg = np.rad2deg(solar_geometry.azimuth_angle_radians(lat, lng, ts))
    np.testing.assert_allclose(az_deg, expected_az_deg, atol=1.0)

  def test_azimuth_angle_solar_noon_and_midnight(self):
    start_ts = _utc_timestamp(2018, 9, 22, 0, 0, 0)
    timestamps = start_ts + np.arange(25 * 60, dtype=np.float64) * 60.0
    lats = np.full_like(timestamps, 37.5630)
    lngs = np.full_like(timestamps, -122.3255)

    zenith_deg = solar_geometry.zenith_angle_degrees(lats, lngs, timestamps)
    azimuth_rad = solar_geometry.azimuth_angle_radians(lats, lngs, timestamps)
    # At Northern Hemisphere mid-latitude, solar noon is due South (pi rad) and
    # solar midnight is due North (0 / 2*pi rad).
    noon_idx = int(np.argmin(zenith_deg))
    midnight_idx = int(np.argmax(zenith_deg))
    np.testing.assert_allclose(
        np.asarray(azimuth_rad)[noon_idx], np.pi, atol=0.05
    )
    midnight_az = float(np.asarray(azimuth_rad)[midnight_idx])
    dist_to_north = min(midnight_az, 2.0 * np.pi - midnight_az)
    self.assertLess(dist_to_north, 0.05)

  def test_subsolar_point_poles_and_nan_propagation(self):
    ts = 1559408444.7
    sub_lat = math.degrees(
        float(
            solar_geometry.declination_angle_radians(
                ts / coin_constants.SECONDS_PER_DAY
            )
        )
    )
    sub_lng = -math.degrees(float(solar_geometry.hour_angle_radians(0.0, ts)))
    sza_sub = float(solar_geometry.zenith_angle_degrees(sub_lat, sub_lng, ts))
    self.assertAlmostEqual(sza_sub, 0.0, places=5)
    self.assertTrue(
        np.isfinite(solar_geometry.azimuth_angle_radians(sub_lat, sub_lng, ts))
    )
    for pole_lat in (90.0, -90.0):
      self.assertTrue(
          np.isfinite(solar_geometry.azimuth_angle_radians(pole_lat, -75.0, ts))
      )

    self.assertTrue(
        np.isnan(solar_geometry.zenith_angle_degrees(np.nan, -75.0, ts))
    )
    self.assertTrue(
        np.isnan(solar_geometry.azimuth_angle_radians(np.nan, -75.0, ts))
    )
    self.assertTrue(
        np.isnan(solar_geometry.azimuth_angle_radians(0.0, np.nan, ts))
    )
    self.assertTrue(
        np.isnan(solar_geometry.azimuth_angle_radians(0.0, -75.0, np.nan))
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='hour_angle',
          fn=lambda ts: solar_geometry.hour_angle_radians(
              np.zeros((5,)), np.full((5, 1), ts)
          ),
      ),
      dict(
          testcase_name='elevation_angle',
          fn=lambda ts: solar_geometry.elevation_angle_radians(
              np.zeros((5,)), np.zeros((5, 1)), ts
          ),
      ),
      dict(
          testcase_name='zenith_angle',
          fn=lambda ts: solar_geometry.zenith_angle_degrees(
              np.zeros((5,)), np.zeros((5, 1)), ts
          ),
      ),
      dict(
          testcase_name='azimuth_angle',
          fn=lambda ts: solar_geometry.azimuth_angle_radians(
              np.zeros((5,)), np.zeros((5, 1)), ts
          ),
      ),
  )
  def test_mismatched_non_scalar_shapes_rejected(self, fn):
    with self.assertRaisesRegex(ValueError, 'share one shape'):
      fn(1559408444.7)

  def test_cos_day_of_year_annual_cycle_and_dtype(self):
    start_ts = _utc_timestamp(2022, 1, 1, 0, 0, 0)
    timestamps = (
        start_ts
        + np.arange(365, dtype=np.float64) * coin_constants.SECONDS_PER_DAY
    )
    cos_doy = solar_geometry.cos_day_of_year(timestamps)
    self.assertEqual(cos_doy.dtype, np.float32)
    self.assertEqual(cos_doy.shape, (365,))
    np.testing.assert_allclose(
        cos_doy, np.cos(np.linspace(0.0, 2.0 * np.pi, 365)), atol=0.1
    )

    scalar_out = solar_geometry.cos_day_of_year(start_ts)
    self.assertEqual(scalar_out.dtype, np.float32)
    np.testing.assert_allclose(scalar_out, cos_doy[0], atol=0.0)


if __name__ == '__main__':
  absltest.main()

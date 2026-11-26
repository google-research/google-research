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

"""Unit tests for the GEM-15 orbit-guided destriping module."""

from absl.testing import absltest
import numpy as np
from gem_15 import destriping


def _synthetic_alos_u_km(
    lon_deg, lat_deg, lon0_deg, lat0_deg
):
  """Computes approximate across-track distance (km) for synthetic waves."""
  az_rad = np.radians(12.5)
  dx_km = (lon_deg - lon0_deg) * 111.32 * np.cos(np.radians(lat0_deg))
  dy_km = (lat_deg - lat0_deg) * 111.32
  return dx_km * np.cos(az_rad) - dy_km * np.sin(az_rad)


class DestripingTest(absltest.TestCase):

  def test_compute_stripe_field_removes_50km_wave_and_preserves_canopy_steep(
      self,
  ):
    bbox_deg = (129.0, -26.0, 133.0, -22.0)
    height, width = 320, 320
    lats = np.linspace(
        -22.0 - 4.0 / (2 * height), -26.0 + 4.0 / (2 * height), height
    )
    lons = np.linspace(
        129.0 + 4.0 / (2 * width), 133.0 - 4.0 / (2 * width), width
    )
    lon2d, lat2d = np.meshgrid(lons, lats)
    u_km = _synthetic_alos_u_km(lon2d, lat2d, 131.0, -24.0)

    ref_dem_m = 400.0 + 2.0 * np.sin(lon2d * 0.4) + 1.5 * np.cos(lat2d * 0.4)
    mesa = (np.abs(lon2d - 131.2) < 0.18) & (np.abs(lat2d + 24.1) < 0.18)
    ref_dem_m = np.where(mesa, ref_dem_m + 180.0, ref_dem_m)

    canopy_m = np.zeros_like(ref_dem_m)
    canopy_patch = (
        (lon2d > 130.2) & (lon2d < 130.7) & (lat2d > -24.8) & (lat2d < -24.3)
    )
    canopy_m[canopy_patch] = 14.0

    true_stripe_m = 0.85 * np.sin(2.0 * np.pi * u_km / 50.0)
    gem_dtm_m = (ref_dem_m + 3.2 + true_stripe_m).astype(np.float32)
    gem_dsm_m = (gem_dtm_m + canopy_m).astype(np.float32)

    eval_s = destriping.compute_stripe_field(
        gem_dsm_m=gem_dsm_m,
        gem_dtm_m=gem_dtm_m,
        ref_dem_m=ref_dem_m.astype(np.float32),
        bbox_deg=bbox_deg,
    )

    in_focal = (
        (lon2d >= 130.5)
        & (lon2d <= 131.5)
        & (lat2d >= -24.5)
        & (lat2d <= -23.5)
    )
    est_stripe_m = eval_s(lon2d[in_focal], lat2d[in_focal])
    resid_m = true_stripe_m[in_focal] - est_stripe_m
    rms_before = float(np.sqrt(np.mean(true_stripe_m[in_focal] ** 2)))
    rms_after = float(np.sqrt(np.mean(resid_m**2)))

    self.assertLess(rms_after, 0.25 * rms_before)

  def test_compute_stripe_field_with_tile_bbox_is_seamless_across_border(self):
    px_per_deg = 60
    west, south, east, north = 128.0, -27.0, 134.0, -22.0
    height = int((north - south) * px_per_deg)
    width = int((east - west) * px_per_deg)
    lats = np.linspace(
        north - 0.5 / px_per_deg, south + 0.5 / px_per_deg, height
    )
    lons = np.linspace(west + 0.5 / px_per_deg, east - 0.5 / px_per_deg, width)
    lon2d, lat2d = np.meshgrid(lons, lats)
    u_km = _synthetic_alos_u_km(lon2d, lat2d, 131.0, -24.5)

    ref_m = np.zeros_like(lon2d, dtype=np.float32)
    gem_m = (
        0.4 * (lon2d - 131.0) ** 2
        + (0.9 + 0.15 * (lon2d - 131.0)) * np.sin(2.0 * np.pi * u_km / 50.0)
    ).astype(np.float32)

    eval_a = destriping.compute_stripe_field(
        gem_dsm_m=gem_m[:, : 5 * px_per_deg],
        gem_dtm_m=gem_m[:, : 5 * px_per_deg],
        ref_dem_m=ref_m[:, : 5 * px_per_deg],
        bbox_deg=(128.0, -27.0, 133.0, -22.0),
        tile_bbox_deg=(130.0, -25.0, 131.0, -24.0),
        halo_deg=1.0,
    )
    eval_b = destriping.compute_stripe_field(
        gem_dsm_m=gem_m[:, px_per_deg:],
        gem_dtm_m=gem_m[:, px_per_deg:],
        ref_dem_m=ref_m[:, px_per_deg:],
        bbox_deg=(129.0, -27.0, 134.0, -22.0),
        tile_bbox_deg=(131.0, -25.0, 132.0, -24.0),
        halo_deg=1.0,
    )

    seam_lats = np.linspace(-24.99, -24.01, 200)
    seam_lons = np.full_like(seam_lats, 131.0)
    np.testing.assert_allclose(
        eval_a(seam_lons, seam_lats),
        eval_b(seam_lons, seam_lats),
        atol=1e-6,
    )

  def test_destripe_elevation_bands_preserves_percentile_order_and_dtm_bound(
      self,
  ):
    lon_deg, lat_deg = np.meshgrid(
        np.linspace(42.2, 42.8, 32), np.linspace(26.8, 26.2, 32)
    )
    base_m = np.full((32, 32), 610.0, dtype=np.float32)
    bands_m = {
        'dsm': base_m + 0.4,
        'dsm_percentile_5': base_m - 0.5,
        'dsm_percentile_50': base_m + 0.3,
        'dsm_percentile_95': base_m + 1.2,
        'dtm': base_m + 0.45,
        'dsm_std': np.full((32, 32), 0.35, dtype=np.float32),
    }

    def fake_stripe_m(lon, lat):
      return (0.6 * np.sin((lon - 42.5) * 8.0 + (lat - 26.5) * 2.0)).astype(
          np.float32
      )

    out = destriping.destripe_elevation_bands(
        bands_m=bands_m,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        eval_stripe_m=fake_stripe_m,
    )

    self.assertTrue(np.all(out['dsm_percentile_5'] <= out['dsm_percentile_50']))
    self.assertTrue(
        np.all(out['dsm_percentile_50'] <= out['dsm_percentile_95'])
    )
    self.assertTrue(np.all(out['dtm'] <= out['dsm']))
    np.testing.assert_allclose(out['dsm_std'], bands_m['dsm_std'])

  def test_compute_stripe_field_returns_zero_when_no_bare_ground(self):
    shape = (64, 64)
    ref_dem_m = np.zeros(shape, dtype=np.float32)
    gem_dtm_m = np.zeros(shape, dtype=np.float32)
    gem_dsm_m = np.full(shape, 25.0, dtype=np.float32)
    eval_s = destriping.compute_stripe_field(
        gem_dsm_m=gem_dsm_m,
        gem_dtm_m=gem_dtm_m,
        ref_dem_m=ref_dem_m,
        bbox_deg=(15.0, 2.0, 19.0, 6.0),
    )
    s_vals = eval_s(np.array([17.0, 17.5]), np.array([4.0, 4.5]))
    np.testing.assert_allclose(s_vals, np.zeros(2, dtype=np.float32))

  def test_compute_stripe_field_rejects_invalid_inputs(self):
    with self.assertRaisesRegex(ValueError, 'identical 2D shapes'):
      destriping.compute_stripe_field(
          gem_dsm_m=np.zeros((16, 16), dtype=np.float32),
          gem_dtm_m=np.zeros((16, 8), dtype=np.float32),
          ref_dem_m=np.zeros((16, 16), dtype=np.float32),
          bbox_deg=(0.0, 0.0, 4.0, 4.0),
      )
    with self.assertRaisesRegex(ValueError, 'min_cell_coverage_pct'):
      destriping.compute_stripe_field(
          gem_dsm_m=np.zeros((16, 16), dtype=np.float32),
          gem_dtm_m=np.zeros((16, 16), dtype=np.float32),
          ref_dem_m=np.zeros((16, 16), dtype=np.float32),
          bbox_deg=(0.0, 0.0, 4.0, 4.0),
          config=destriping.DestripingConfig(min_cell_coverage_pct=0.0),
      )


if __name__ == '__main__':
  absltest.main()

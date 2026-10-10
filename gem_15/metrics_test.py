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

"""Unit tests for the elevation validation metrics module."""

import json
from absl.testing import absltest
import numpy as np
import pandas as pd
from gem_15 import metrics


class MetricsTest(absltest.TestCase):

  def test_compute_global_nmad_from_hist_series(self):
    # Two tiles:
    # Tile 1: 100 px in bin 10 (0.10-0.11m), 100 px in bin 20 (0.20-0.21m)
    # Tile 2 has 200 pixels in bin 20 (0.20m - 0.21m)
    # Total pixels = 400. Half = 200.
    # Cumulative: bin 10 has 100 px, bin 20 has 300 px (cum = 400).
    # Median falls in bin 20 at fraction (200 - 100) / 300 = 1/3 = 0.333333
    # Interpolated MAD = 0.01 * (20 + 1/3) = 0.2033333 m.
    # NMAD = 1.4826 * 0.2033333 = 0.30146199 m.
    hist1 = json.dumps([[10, 100], [20, 100]])
    hist2 = json.dumps([[20, 200]])
    series = pd.Series([hist1, hist2])
    nmad = metrics.compute_global_nmad_from_hist_series(series)
    expected_mad = 0.01 * (20.0 + 100.0 / 300.0)
    expected_nmad = 1.4826 * expected_mad
    self.assertAlmostEqual(nmad, expected_nmad, places=5)

  def test_compute_global_nmad_empty(self):
    series = pd.Series(['[]', '', None])
    nmad = metrics.compute_global_nmad_from_hist_series(series)
    self.assertTrue(np.isnan(nmad))

  def test_pooled_bias_and_rmse(self):
    # Tile 1: 100 pixels, bias = +2.0m, variance = 1.0m^2 (RMSE = sqrt(5))
    # Tile 2: 300 pixels, bias = -1.0m, variance = 4.0m^2 (RMSE = sqrt(5))
    # Total pixels = 400.
    # Pooled bias = (100 * 2.0 + 300 * -1.0) / 400 = -100 / 400 = -0.25m.
    # Pooled variance (DRMSE^2) = (100 * 1.0 + 300 * 4.0) / 400 = 3.25 m^2
    # Pooled DRMSE = sqrt(3.25) = 1.8027756m.
    # Direct RMSE^2 = 3.25 + (-0.25)^2 = 3.25 + 0.0625 = 3.3125 m^2
    # sum(n * rmse^2) / N = (100 * 5 + 300 * 5) / 400 = 5.0 -> sqrt(5) = 2.236m
    df = pd.DataFrame({
        'ge_bias': [2.0, -1.0],
        'ge_debiased_rmse': [1.0, 2.0],  # sqrt(1)=1, sqrt(4)=2
        'ge_direct_rmse': [np.sqrt(5.0), np.sqrt(5.0)],
        'ge_count': [100, 300],
    })
    b = metrics.pooled_bias(df, 'ge_bias', 'ge_count')
    deb = metrics.pooled_debiased_rmse(df, 'ge_debiased_rmse', 'ge_count')
    dir_r = metrics.pooled_direct_rmse(df, 'ge_direct_rmse', 'ge_count')

    self.assertAlmostEqual(b, -0.25, places=5)
    self.assertAlmostEqual(deb, np.sqrt(3.25), places=5)
    self.assertAlmostEqual(dir_r, np.sqrt(5.0), places=5)

  def test_pooling_has_no_unweighted_fallback(self):
    df = pd.DataFrame({
        'ge_bias': [2.0, -1.0],
        'ge_direct_rmse': [1.0, 3.0],
    })
    self.assertTrue(np.isnan(metrics.pooled_bias(df, 'ge_bias', None)))
    self.assertTrue(np.isnan(metrics.pooled_bias(df, 'ge_bias', 'ge_count')))
    self.assertTrue(
        np.isnan(metrics.pooled_direct_rmse(df, 'ge_direct_rmse', None))
    )

  def test_find_count_column_rejects_generic_columns(self):
    df = pd.DataFrame({'n_px': [1], 'n_px_g': [1], 'cop_count_g': [1]})
    self.assertIsNone(metrics.find_count_column(df, 'cop', ''))
    self.assertEqual(metrics.find_count_column(df, 'cop', '_g'), 'cop_count_g')

  def test_compute_global_nmad_rejects_out_of_range_bin(self):
    series = pd.Series([json.dumps([[10001, 5]])])
    with self.assertRaises(ValueError):
      metrics.compute_global_nmad_from_hist_series(series)

  def test_compute_percentiles_metrics_from_df(self):
    df = pd.DataFrame({
        'dsm_diff_mean': [0.5, 1.0],
        'dsm_diff_variance': [1.0, 4.0],
        'dsm_diff_count': [100, 100],
        'dsm_diff_hist': [json.dumps([[50, 100]]), json.dumps([[50, 100]])],
    })
    res = metrics.compute_percentiles_metrics_from_df(df, bands=['dsm'])
    self.assertIn('dsm', res)
    self.assertAlmostEqual(res['dsm']['bias'], 0.75)
    self.assertAlmostEqual(res['dsm']['debiased_rmse'], np.sqrt(2.5))
    self.assertAlmostEqual(res['dsm']['nmad'], 1.4826 * 0.505, places=3)

  def test_percentiles_nmad_is_nan_without_histogram(self):
    df = pd.DataFrame({
        'dsm_diff_mean': [0.5, 1.0],
        'dsm_diff_variance': [1.0, 4.0],
        'dsm_diff_count': [100, 100],
    })
    res = metrics.compute_percentiles_metrics_from_df(df, bands=['dsm'])
    self.assertTrue(np.isnan(res['dsm']['nmad']))
    self.assertFalse(np.isnan(res['dsm']['debiased_rmse']))

  def test_pooled_nmad_with_histogram(self):
    hist1 = json.dumps([[50, 100]])  # 0.50m - 0.51m
    hist2 = json.dumps([[50, 100]])
    df_hist = pd.DataFrame({
        'ge_hist': [hist1, hist2],
        'ge_count': [100, 100],
    })
    nmad_from_hist = metrics.pooled_nmad(df_hist, 'ge')
    self.assertAlmostEqual(nmad_from_hist, 1.4826 * 0.505, places=3)

    df_no_hist = pd.DataFrame({
        'ge_nmad': [1.0, 2.0],
        'ge_count': [100, 300],
    })
    nmad_no_hist = metrics.pooled_nmad(df_no_hist, 'ge')
    self.assertTrue(np.isnan(nmad_no_hist))

  def test_compute_benchmark_metrics_from_df(self):
    hist1 = json.dumps([[50, 100]])
    hist2 = json.dumps([[50, 200]])
    df = pd.DataFrame({
        'ge_bias': [1.0, 2.0],
        'ge_debiased_rmse': [1.5, 2.5],
        'ge_direct_rmse': [2.0, 3.0],
        'ge_hist': [hist1, hist2],
        'ge_count': [100, 200],
        'slope_cop_mean': [4.0, 12.0],
    })
    res = metrics.compute_benchmark_metrics_from_df(df, min_slope_deg=0.0)
    self.assertIn('ge_bias', res)
    self.assertIn('ge_direct_rmse', res)
    self.assertIn('ge_debiased_rmse', res)
    self.assertIn('ge_nmad', res)

    self.assertAlmostEqual(res['ge_bias'], (100 * 1.0 + 200 * 2.0) / 300.0)
    self.assertAlmostEqual(res['ge_nmad'], 1.4826 * 0.505, places=3)

    res_steep = metrics.compute_benchmark_metrics_from_df(
        df, min_slope_deg=10.0
    )
    self.assertAlmostEqual(res_steep['ge_bias'], 2.0)

  def test_compute_point_metrics_from_df(self):
    df = pd.DataFrame({
        'ref_elevation': [100.0, 200.0, 300.0],
        'ge': [101.0, 199.0, 303.0],
    })
    res = metrics.compute_point_metrics_from_df(df, models=['ge'])
    self.assertEqual(res['count'], 3)
    self.assertAlmostEqual(res['ge_bias'], 1.0, places=5)
    self.assertAlmostEqual(
        res['ge_debiased_rmse'], np.sqrt(8.0 / 3.0), places=5
    )
    self.assertAlmostEqual(res['ge_direct_rmse'], np.sqrt(11.0 / 3.0), places=5)
    self.assertAlmostEqual(res['ge_nmad'], 1.4826 * 2.0, places=5)


if __name__ == '__main__':
  absltest.main()

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

"""CPU unit tests for Flowracle feature engineering, GBDTs, and metrics."""

import tempfile
from typing import Tuple
import unittest

from flowracle.data_pipeline import get_issuance_times
from flowracle.features import batch_to_table
from flowracle.features import NUM_ENSEMBLE
from flowracle.features import prepare_shared_features
from flowracle.features import QUANTILES
from flowracle.metrics import compute_composite_cv_objective
from flowracle.metrics import compute_crps
from flowracle.metrics import compute_fold_validation_ncrps
from flowracle.metrics import compute_hydrograph_timing_metrics
from flowracle.metrics import quantile_loss
from flowracle.model import build_monotone_constraints
from flowracle.model import compute_submodel_feature_masks
from flowracle.model import fit_model
from flowracle.model import load_model
from flowracle.model import predict
import numpy as np
import xarray as xr


def _make_synthetic_aux_and_batch(
    batch_size = 4,
    n_stations = 5,
    n_snotel = 12,
    n_ca_sensors = 10,
    origin_months = (1, 2, 3, 4),
    seed = 123,
):
  """Builds deterministic synthetic multi-source xarray batches for CPU tests."""
  rng = np.random.RandomState(seed)

  station_ids = np.array([f"STN_{i}" for i in range(n_stations)], dtype=object)
  # Span across the 37.0N southern-Sierra CDEC sensor gating boundary.
  fnf_lats = np.linspace(35.8, 40.5, n_stations).astype(np.float64)
  fnf_lons = np.linspace(239.0, 241.5, n_stations).astype(np.float64)
  fnf_elev_ft = np.linspace(1500.0, 7500.0, n_stations).astype(np.float64)

  sn_lats = np.linspace(35.5, 41.5, n_snotel).astype(np.float32)
  sn_lons = np.linspace(238.5, 242.0, n_snotel).astype(np.float32)
  sn_elev_m = np.linspace(1200.0, 2900.0, n_snotel).astype(np.float32)

  ca_lats = np.linspace(35.5, 40.5, n_ca_sensors).astype(np.float64)
  ca_lons = np.linspace(238.8, 241.8, n_ca_sensors).astype(np.float64)
  ca_elev_ft = np.linspace(4000.0, 9500.0, n_ca_sensors).astype(np.float64)

  aux_ds = xr.Dataset(
      data_vars={
          "latitude": ("station_id", fnf_lats),
          "longitude": ("station_id", fnf_lons),
          "elevation": ("station_id", fnf_elev_ft),
          "snotel_latitude": ("index", sn_lats),
          "snotel_longitude": ("index", sn_lons),
          "snotel_elevation": ("index", sn_elev_m),
          "ca_sensor_latitude": ("station", ca_lats),
          "ca_sensor_longitude": ("station", ca_lons),
          "ca_sensor_elevation": ("station", ca_elev_ft),
      },
      coords={
          "station_id": station_ids,
          "index": np.arange(n_snotel),
          "station": np.array(
              [f"CA_{i}" for i in range(n_ca_sensors)], dtype=object
          ),
      },
  )

  lag6 = np.arange(-5, 1) * np.timedelta64(30, "D")
  lag5 = np.arange(-5, 0) * np.timedelta64(30, "D")
  lag1 = np.array([0]) * np.timedelta64(30, "D")
  lead6 = np.arange(1, 7) * np.timedelta64(30, "D")
  eom_lag = np.arange(5)

  grid_lat = np.linspace(33.0, 41.8, 8).astype(np.float32)
  grid_lon = np.linspace(237.0, 243.0, 6).astype(np.float32)

  m_arr = np.array(origin_months[:batch_size], dtype=np.int64).reshape(
      batch_size, 1
  )
  real_times = np.array(
      [np.datetime64(f"2022-{int(m):02d}-01") for m in m_arr[:, 0]],
      dtype="datetime64[ns]",
  ).reshape(batch_size, 1)

  prism_ds = xr.Dataset(
      data_vars={
          "ppt": (
              ("batch", "lag_time", "prism_latitude", "prism_longitude"),
              rng.uniform(5.0, 180.0, size=(batch_size, 6, 8, 6)).astype(
                  np.float32
              ),
          ),
          "tmean": (
              ("batch", "lag_time", "prism_latitude", "prism_longitude"),
              rng.uniform(-4.0, 14.0, size=(batch_size, 6, 8, 6)).astype(
                  np.float32
              ),
          ),
      },
      coords={
          "lag_time": lag6,
          "prism_latitude": grid_lat,
          "prism_longitude": grid_lon,
      },
  )

  ua_swe_ds = xr.Dataset(
      data_vars={
          "SWE": (
              ("batch", "lag_time", "ua_latitude", "ua_longitude"),
              rng.uniform(0.0, 650.0, size=(batch_size, 6, 8, 6)).astype(
                  np.float32
              ),
          ),
      },
      coords={
          "lag_time": lag6,
          "ua_latitude": grid_lat,
          "ua_longitude": grid_lon,
      },
  )

  snotel_ds = xr.Dataset(
      data_vars={
          "snow_water_equivalent_start_of_day_PST": (
              ("batch", "lag_time", "index"),
              rng.uniform(0.02, 0.9, size=(batch_size, 6, n_snotel)).astype(
                  np.float32
              ),
          ),
          "precipitation_accumulation_start_of_day_PST": (
              ("batch", "lag_time", "index"),
              rng.uniform(0.1, 1.4, size=(batch_size, 6, n_snotel)).astype(
                  np.float32
              ),
          ),
      },
      coords={"lag_time": lag6, "index": np.arange(n_snotel)},
  )

  snotel_eom_ds = xr.Dataset(
      data_vars={
          "snow_water_equivalent_start_of_day_PST": (
              ("batch", "lag_time", "index", "eom_lag"),
              rng.uniform(0.02, 0.95, size=(batch_size, 1, n_snotel, 5)).astype(
                  np.float32
              ),
          ),
          "month": (("batch", "lag_time"), m_arr),
          "real_time": (("batch", "lag_time"), real_times),
      },
      coords={
          "lag_time": lag1,
          "index": np.arange(n_snotel),
          "eom_lag": eom_lag,
      },
  )

  ca_sensors_ds = xr.Dataset(
      data_vars={
          "sensor_82": (
              ("batch", "lag_time", "station"),
              rng.uniform(0.5, 35.0, size=(batch_size, 6, n_ca_sensors)).astype(
                  np.float32
              ),
          ),
      },
      coords={"lag_time": lag6},
  )

  daily_ca_ds = xr.Dataset(
      data_vars={
          "sensor_82": (
              ("batch", "lag_time", "station", "eom_lag"),
              rng.uniform(
                  0.5, 36.0, size=(batch_size, 1, n_ca_sensors, 5)
              ).astype(np.float32),
          ),
      },
      coords={"lag_time": lag1, "eom_lag": eom_lag},
  )

  mjo_ds = xr.Dataset(
      data_vars={
          "eom_rmm1": (
              ("batch", "lag_time"),
              rng.normal(0, 1, size=(batch_size, 6)).astype(np.float32),
          ),
          "eom_rmm2": (
              ("batch", "lag_time"),
              rng.normal(0, 1, size=(batch_size, 6)).astype(np.float32),
          ),
          "monthly_mjo_amp": (
              ("batch", "lag_time"),
              rng.uniform(0.3, 2.2, size=(batch_size, 6)).astype(np.float32),
          ),
      },
      coords={"lag_time": lag6},
  )

  pdsi_ds = xr.Dataset(
      data_vars={
          "daily_mean_palmer_drought_severity_index": (
              ("batch", "lag_time", "pdsi_latitude", "pdsi_longitude"),
              rng.uniform(-3.5, 3.5, size=(batch_size, 6, 8, 6)).astype(
                  np.float32
              ),
          ),
      },
      coords={
          "lag_time": lag6,
          "pdsi_latitude": grid_lat,
          "pdsi_longitude": grid_lon,
      },
  )

  past_fnf_ds = xr.Dataset(
      data_vars={
          "fnf": (
              ("batch", "lag_time", "station_id"),
              rng.uniform(2e3, 1.8e5, size=(batch_size, 5, n_stations)).astype(
                  np.float32
              ),
          ),
      },
      coords={"lag_time": lag5, "station_id": station_ids},
  )

  target_fnf_ds = xr.Dataset(
      data_vars={
          "target_fnf_fnf": (
              ("batch", "lead_time", "station_id"),
              rng.uniform(5e3, 3.5e5, size=(batch_size, 6, n_stations)).astype(
                  np.float32
              ),
          ),
      },
      coords={"lead_time": lead6, "station_id": station_ids},
  )

  batch = {
      "prism": prism_ds,
      "ua_swe": ua_swe_ds,
      "snotel": snotel_ds,
      "snotel_eom": snotel_eom_ds,
      "ca_sensors": ca_sensors_ds,
      "daily_ca_sensors": daily_ca_ds,
      "mjo": mjo_ds,
      "pdsi": pdsi_ds,
      "past_fnf": past_fnf_ds,
      "target_fnf": target_fnf_ds,
  }
  return aux_ds, batch


class FlowraclePipelineTest(unittest.TestCase):

  def test_feature_dimensions_submodel_masks_and_monotonicity(self):
    aux_ds, batch = _make_synthetic_aux_and_batch(batch_size=4, n_stations=5)
    f_glob, f_loc, n_glob, n_loc, m_iss = prepare_shared_features(
        batch, aux_ds, jitter=False
    )
    self.assertEqual(f_glob.shape, (4, 29))
    self.assertEqual(f_loc.shape, (4, 5, 20))
    self.assertEqual(len(n_glob), 29)
    self.assertEqual(len(n_loc), 20)
    self.assertEqual(m_iss.shape, (4,))

    coords_np = np.zeros((5, 3), dtype=np.float32)
    x_mat, y_vec, f_names, m_iss_flat = batch_to_table(
        batch, aux_ds, coords_np, jitter=False
    )
    self.assertEqual(x_mat.shape, (4 * 6 * 5, 55))
    self.assertEqual(y_vec.shape, (4 * 6 * 5,))
    self.assertEqual(len(f_names), 55)
    self.assertEqual(len(m_iss_flat), 4 * 6 * 5)

    _, elev_mask, melt_mask, resid_mask = compute_submodel_feature_masks(
        f_names
    )
    self.assertEqual(int(elev_mask.sum()), 44)
    self.assertEqual(int(melt_mask.sum()), 51)
    self.assertEqual(int(resid_mask.sum()), 47)

    mono_melt = build_monotone_constraints(
        np.array(f_names)[melt_mask].tolist()
    )
    self.assertEqual(len(mono_melt), 51)
    self.assertGreater(sum(mono_melt), 15)

  def test_march_kinematic_surge_damping_attenuates_swe(self):
    aux_ds, batch_feb = _make_synthetic_aux_and_batch(
        batch_size=1, n_stations=3, origin_months=(2,), seed=7
    )
    _, batch_mar = _make_synthetic_aux_and_batch(
        batch_size=1, n_stations=3, origin_months=(3,), seed=7
    )
    # Impose a sharp late-month SNOTEL SWE jump at the final lag step (-1).
    for b in (batch_feb, batch_mar):
      b["snotel"]["snow_water_equivalent_start_of_day_PST"].values[
          :, :-1, :
      ] = 0.10
      b["snotel"]["snow_water_equivalent_start_of_day_PST"].values[
          :, -1, :
      ] = 0.80

    _, loc_feb, _, l_names, _ = prepare_shared_features(batch_feb, aux_ds)
    _, loc_mar, _, _, _ = prepare_shared_features(batch_mar, aux_ds)
    sn_swe_idx = l_names.index("l_sn_elev_swe")
    self.assertTrue(
        np.all(loc_mar[:, :, sn_swe_idx] < loc_feb[:, :, sn_swe_idx])
    )

  def test_fit_load_and_predict_end_to_end_cpu(self):
    aux_ds, batch1 = _make_synthetic_aux_and_batch(
        batch_size=4, n_stations=4, origin_months=(1, 2, 3, 4), seed=10
    )
    _, batch2 = _make_synthetic_aux_and_batch(
        batch_size=4, n_stations=4, origin_months=(5, 6, 11, 12), seed=11
    )
    eval_dates = get_issuance_times([batch1])
    self.assertEqual(str(eval_dates[0])[:10], "2022-02-01")

    with tempfile.TemporaryDirectory() as tmpdir:
      trained = fit_model(
          iter([batch1, batch2]),
          [batch1],
          aux_ds,
          model_dir=tmpdir,
          num_collect_batches=2,
          augmentation_epochs=1,
          n_estimators_override=6,
          n_oof_splits=2,
          device="cpu",
      )
      self.assertEqual(len(trained["models_elev"]), len(QUANTILES))
      self.assertEqual(len(trained["models_melt"]), len(QUANTILES))
      self.assertEqual(len(trained["models_resid"]), len(QUANTILES))

      reloaded = load_model(model_dir=tmpdir)
      preds_mem = predict([batch1], aux_ds, model=trained)
      preds_disk = predict([batch1], aux_ds, model=reloaded)

      self.assertEqual(preds_mem.shape, (NUM_ENSEMBLE, 4, 6, 4))
      self.assertTrue(np.all(preds_mem >= 0.0))
      # Ensemble members are interpolated from sorted quantiles and must be
      # monotonically non-decreasing along axis 0.
      self.assertTrue(np.all(np.diff(preds_mem, axis=0) >= -1e-5))
      np.testing.assert_allclose(preds_mem, preds_disk, rtol=1e-6, atol=1e-6)

  def test_probabilistic_and_timing_metrics(self):
    rng = np.random.RandomState(99)
    ens = rng.lognormal(mean=10.0, sigma=0.4, size=(20, 6, 3))
    obs = rng.lognormal(mean=10.0, sigma=0.4, size=(6, 3))

    # Verify O(M log M) CRPS matches exact O(M^2) pairwise definition.
    std_crps = compute_crps(ens, obs, ensemble_axis=0)
    m = ens.shape[0]
    mae = np.mean(np.abs(ens - obs[None, :, :]), axis=0)
    pairwise = np.sum(
        np.abs(ens[:, None, :, :] - ens[None, :, :, :]), axis=(0, 1)
    )
    crps_exact = mae - pairwise / (2.0 * m * m)
    np.testing.assert_allclose(std_crps, crps_exact, rtol=1e-10, atol=1e-8)
    self.assertAlmostEqual(
        quantile_loss(np.array([100.0, 200.0]), np.array([150.0, 150.0]), 0.5),
        25.0,
    )

    # Fold nCRPS and CV objective check
    ens_4d = rng.uniform(1e4, 2e5, size=(25, 4, 6, 3))
    obs_3d = rng.uniform(1e4, 2e5, size=(4, 6, 3))
    fold_stats = compute_fold_validation_ncrps(ens_4d, obs_3d, [2, 3, 4, 5])
    self.assertGreater(fold_stats["ncrps"], 0.0)
    cv_summary = compute_composite_cv_objective(
        [0.18, 0.20, 0.19], [0.0, 0.02, 0.01]
    )
    self.assertGreater(
        cv_summary["composite_cv_objective"], cv_summary["mean_ncrps"]
    )

    # Hydrograph timing check: impulse in April (water-year index 6 -> 197.0 d).
    wy_q = np.zeros((12, 2))
    wy_q[6, 0] = 1000.0  # April only -> CT = 197.0, SDoT = 0, summer = 0%
    wy_q[9, 1] = 1000.0  # July only  -> CT = 288.5, SDoT = 0, summer = 100%
    ct, sdot, f_sum = compute_hydrograph_timing_metrics(wy_q)
    np.testing.assert_allclose(ct, [197.0, 288.5], atol=1e-4)
    np.testing.assert_allclose(sdot, [0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(f_sum, [0.0, 100.0], atol=1e-4)

  def test_aux_ds_none_multi_station_features_and_predict(self):
    aux_ds, batch = _make_synthetic_aux_and_batch(
        batch_size=3, n_stations=4, origin_months=(2, 3, 4), seed=21
    )
    _, f_loc, _, n_loc, _ = prepare_shared_features(batch, aux_ds=None)
    self.assertEqual(f_loc.shape, (3, 4, 20))
    self.assertIn("l_sn_elev_swe", n_loc)

    # Also verify batch_to_table broadcast guard when past_fnf is absent
    # (`f_loc` shape `(B, 1, 20)` while `coords_np` has `S = 4`).
    batch_no_past_fnf = {k: v for k, v in batch.items() if k != "past_fnf"}
    coords_np = np.zeros((4, 3), dtype=np.float32)
    x_mat, _, _, _ = batch_to_table(batch_no_past_fnf, None, coords_np)
    self.assertEqual(x_mat.shape, (3 * 6 * 4, 55))

    with tempfile.TemporaryDirectory() as tmpdir:
      trained_with_aux = fit_model(
          iter([batch]),
          None,
          aux_ds,
          model_dir=tmpdir,
          num_collect_batches=1,
          augmentation_epochs=1,
          n_estimators_override=4,
          n_oof_splits=2,
          device="cpu",
      )
      preds_no_aux = predict([batch], aux_ds=None, model=trained_with_aux)
      self.assertEqual(preds_no_aux.shape, (NUM_ENSEMBLE, 3, 6, 4))
      self.assertTrue(np.all(np.isfinite(preds_no_aux)))

      preds_no_past_fnf = predict(
          [batch_no_past_fnf], aux_ds=None, model=trained_with_aux
      )
      self.assertEqual(preds_no_past_fnf.shape, (NUM_ENSEMBLE, 3, 6, 4))
      self.assertTrue(np.all(np.isfinite(preds_no_past_fnf)))

      # Verify fit_model with aux_ds=None also succeeds on multi-station data.
      trained_no_aux = fit_model(
          iter([batch]),
          None,
          None,
          model_dir=tmpdir,
          num_collect_batches=1,
          augmentation_epochs=1,
          n_estimators_override=4,
          n_oof_splits=2,
          device="cpu",
      )
      preds_fit_no_aux = predict([batch], aux_ds=None, model=trained_no_aux)
      self.assertEqual(preds_fit_no_aux.shape, (NUM_ENSEMBLE, 3, 6, 4))

  def test_fold_validation_ncrps_nan_handling(self):
    import warnings  # pylint: disable=g-import-not-at-top

    rng = np.random.RandomState(42)
    ens_4d = rng.uniform(1e4, 2e5, size=(20, 5, 6, 3))
    obs_3d = rng.uniform(1e4, 2e5, size=(5, 6, 3))
    issue_months = [1, 2, 3, 4, 5]

    obs_with_nans = obs_3d.copy()
    # Inject sporadic missing gauge observations and a trailing N_PAD_MONTHS
    # all-NaN issuance step.
    obs_with_nans[1, 2, 0] = np.nan
    obs_with_nans[2, 4, 1] = np.nan
    obs_with_nans[4, :, :] = np.nan

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always", category=RuntimeWarning)
      stats = compute_fold_validation_ncrps(ens_4d, obs_with_nans, issue_months)
    self.assertEqual(len(caught), 0)
    for key in ("per_lead_crps", "amjj_cumul_crps", "combined_crps", "ncrps"):
      self.assertTrue(np.isfinite(stats[key]), msg=f"{key} was not finite")
      self.assertGreater(stats[key], 0.0, msg=f"{key} was not positive")


if __name__ == "__main__":
  unittest.main()

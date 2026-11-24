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

"""Hermetic unit and invariant tests for the COIN single-scantime pipeline."""

from collections.abc import Mapping
import datetime
import hashlib
import json
import math
import os
import tempfile
from typing import cast
from unittest import mock

from absl import app
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import netCDF4  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
import xarray

from collocated_irradiance_network import coin_constants
from collocated_irradiance_network import coin_features
from collocated_irradiance_network import coin_model
from collocated_irradiance_network import fixed_grid
from collocated_irradiance_network import goes_abi_l1b
from collocated_irradiance_network import run_single_scantime
from collocated_irradiance_network import solar_geometry
from collocated_irradiance_network import write_output


ARBITRARY_INT = 5
ARBITRARY_FLOAT = 50.0


def _make_valid_train_config(label_mode: str) -> dict[str, object]:
  """Creates a valid COIN train_config matching the production schema."""
  return {
      **coin_constants.REQUIRED_ARCHITECTURE_KEYS,
      'label_mode': label_mode,
      **{
          key: ARBITRARY_FLOAT
          for key in coin_constants.REQUIRED_STANDARDIZATION_KEYS
      },
  }


def _make_synthetic_weights(
    seed: int, label_mode: str
) -> coin_model.CoinWeights:
  """Creates deterministic synthetic CoinWeights with valid shapes for mode."""
  rng = np.random.default_rng(seed)
  kernels: list[np.ndarray] = []
  biases: list[np.ndarray] = []
  k_shapes = coin_constants.EXPECTED_KERNEL_SHAPES_BY_MODE[label_mode]
  b_shapes = coin_constants.EXPECTED_BIAS_SHAPES_BY_MODE[label_mode]
  for k_shape, b_shape in zip(k_shapes, b_shapes, strict=True):
    kernels.append(rng.normal(0.0, 0.08, size=k_shape).astype(np.float32))
    biases.append(rng.normal(0.0, 0.02, size=b_shape).astype(np.float32))
  val_k, val_b = coin_model.validate_weights(kernels, biases, label_mode)
  return coin_model.CoinWeights(
      kernels=val_k,
      biases=val_b,
      weights_sha256=f'synthetic_sha256_{label_mode}_{seed}',
      label_mode=label_mode,
  )


def _write_synthetic_npz_subbundle(
    sub_dir: str,
    weights: coin_model.CoinWeights,
    train_config: dict[str, object],
    model_version: str,
) -> dict[str, object]:
  """Writes `coin_weights.npz`, `train_config.json`, and `MODEL_MANIFEST.json`."""
  os.makedirs(sub_dir, exist_ok=True)
  npz_path = os.path.join(sub_dir, 'coin_weights.npz')
  arrays_dict = {}
  for idx, (k, b) in enumerate(
      zip(weights.kernels, weights.biases, strict=True)
  ):
    arrays_dict[f'kernel_{idx}'] = k
    arrays_dict[f'bias_{idx}'] = b
  np.savez(npz_path, **arrays_dict)

  config_path = os.path.join(sub_dir, 'train_config.json')
  with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(train_config, f, indent=2, sort_keys=True)

  with open(npz_path, 'rb') as f:
    weights_sha256 = hashlib.sha256(f.read()).hexdigest()
  with open(config_path, 'rb') as f:
    config_sha256 = hashlib.sha256(f.read()).hexdigest()
  tensor_hasher = hashlib.sha256()
  for k, b in zip(weights.kernels, weights.biases, strict=True):
    tensor_hasher.update(np.asarray(k, dtype='<f4').tobytes())
    tensor_hasher.update(np.asarray(b, dtype='<f4').tobytes())
  manifest: dict[str, object] = {
      'bundle_mode': 'SINGLE',
      'label_mode': weights.label_mode,
      'model_version': model_version,
      'weights_sha256': weights_sha256,
      'tensors_sha256': tensor_hasher.hexdigest(),
      'train_config_sha256': config_sha256,
  }
  with open(
      os.path.join(sub_dir, 'MODEL_MANIFEST.json'), 'w', encoding='utf-8'
  ) as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest


def _write_synthetic_abi_nc(
    filepath: str,
    channel: int,
    rad_values: np.ndarray,
    timeline_id: str,
    lon_0: float,
    rad_units: str,
    use_packed_int16: bool,
    platform_id: str,
) -> None:
  """Writes a synthetic GOES ABI L1b RadF NetCDF4 file for testing."""
  rows, cols = rad_values.shape
  h = coin_constants.GOES_PERSPECTIVE_POINT_HEIGHT
  ext = coin_constants.GOES_AREA_EXTENT_METERS
  x_bounds = np.array([ext[0] / h, ext[2] / h], dtype=np.float64)
  y_bounds = np.array([ext[3] / h, ext[1] / h], dtype=np.float64)
  dx = (x_bounds[1] - x_bounds[0]) / float(cols)
  dy = (y_bounds[1] - y_bounds[0]) / float(rows)
  x_coords = np.linspace(
      x_bounds[0] + (dx / 2.0), x_bounds[1] - (dx / 2.0), cols, dtype=np.float32
  )
  y_coords = np.linspace(
      y_bounds[0] + (dy / 2.0), y_bounds[1] - (dy / 2.0), rows, dtype=np.float32
  )

  proj_da = xarray.DataArray(
      data=np.int32(-2147483647),
      attrs={
          'grid_mapping_name': 'geostationary',
          'perspective_point_height': h,
          'semi_major_axis': coin_constants.GOES_SEMI_MAJOR_AXIS,
          'semi_minor_axis': coin_constants.GOES_SEMI_MINOR_AXIS,
          'inverse_flattening': coin_constants.GOES_INVERSE_FLATTENING,
          'latitude_of_projection_origin': 0.0,
          'longitude_of_projection_origin': lon_0,
          'sweep_angle_axis': coin_constants.GOES_SWEEP_ANGLE_AXIS,
      },
  )

  if use_packed_int16:
    fill_val = np.int16(1023)
    scale_factor = np.float32(0.05)
    add_offset = np.float32(-1.0)
    is_nan = ~np.isfinite(rad_values)
    safe_rad = np.where(is_nan, add_offset, rad_values)
    packed = np.round((safe_rad - add_offset) / scale_factor).astype(np.int16)
    packed[is_nan] = fill_val
    rad_da = xarray.DataArray(
        data=packed,
        dims=['y', 'x'],
        attrs={
            '_FillValue': fill_val,
            'scale_factor': scale_factor,
            'add_offset': add_offset,
            'units': rad_units,
        },
    )
  else:
    rad_da = xarray.DataArray(
        data=rad_values.astype(np.float32),
        dims=['y', 'x'],
        attrs={'units': rad_units},
    )

  ds = xarray.Dataset(
      data_vars={
          'Rad': rad_da,
          coin_constants.GOES_PROJECTION_KEY: proj_da,
          'x_image_bounds': (['number_of_image_bounds'], x_bounds),
          'y_image_bounds': (['number_of_image_bounds'], y_bounds),
          'band_id': (['band'], np.array([channel], dtype=np.int8)),
      },
      coords={
          'y': y_coords,
          'x': x_coords,
      },
      attrs={
          'platform_ID': platform_id,
          'scene_id': coin_constants.REQUIRED_SCENE_ID,
          'timeline_id': timeline_id,
      },
  )
  with coin_constants.NETCDF4_LOCK:
    ds.to_netcdf(filepath, engine='netcdf4')


class CoinInferenceTest(parameterized.TestCase):
  """Unit and invariant tests for the COIN single-scantime inference pipeline."""

  def test_mode_and_mask_constants(self) -> None:
    self.assertEqual(
        coin_constants.ALLOWED_LABEL_MODES, frozenset({'OLR', 'RSR'})
    )
    self.assertEqual(
        coin_constants.GOES_LONGWAVE_CHANNELS,
        (8, 9, 10, 11, 12, 13, 14, 15, 16),
    )
    self.assertEqual(
        coin_constants.GOES_SHORTWAVE_CHANNELS,
        (1, 2, 3, 4, 5, 6, 7),
    )
    self.assertEqual(coin_constants.MASK_OLR_BIT, 1)
    self.assertEqual(coin_constants.MASK_RSR_BIT, 2)
    self.assertEqual(coin_constants.MASK_BOTH_BITS, 3)

  @parameterized.named_parameters(
      ('olr', 'OLR', 15, 14, coin_constants.GOES_SHORTWAVE_CHANNELS),
      ('rsr', 'RSR', 13, 12, coin_constants.GOES_LONGWAVE_CHANNELS),
  )
  def test_feature_order_constants_by_mode(
      self,
      label_mode: str,
      n_inputs: int,
      sza_index: int,
      excluded_channels: tuple[int, ...],
  ) -> None:
    """Verifies model input column order and solar zenith column per mode."""
    all_input_names = (
        'cos_day_of_year',
        'goes_channel_1',
        'goes_channel_10',
        'goes_channel_11',
        'goes_channel_12',
        'goes_channel_13',
        'goes_channel_14',
        'goes_channel_15',
        'goes_channel_16',
        'goes_channel_2',
        'goes_channel_3',
        'goes_channel_4',
        'goes_channel_5',
        'goes_channel_6',
        'goes_channel_7',
        'goes_channel_8',
        'goes_channel_9',
        'goes_cos_solar_azimuths',
        'goes_lats',
        'goes_lngs',
        'goes_sin_solar_azimuths',
        'goes_solar_zeniths',
    )
    sat_name = coin_constants.SATELLITE_NAME
    excluded_names = {f'{sat_name}_channel_{ch}' for ch in excluded_channels}
    names = coin_constants.SORTED_INPUT_NAMES_BY_MODE[label_mode]
    self.assertEqual(
        names,
        tuple(n for n in all_input_names if n not in excluded_names),
    )
    self.assertLen(names, n_inputs)
    self.assertEqual(names.index(f'{sat_name}_solar_zeniths'), sza_index)
    self.assertEqual(coin_constants.SZA_INDEX_BY_MODE[label_mode], sza_index)

  @parameterized.named_parameters(
      ('olr', 'OLR'),
      ('rsr', 'RSR'),
  )
  def test_weight_loading_and_sha256_validation_by_mode(
      self, label_mode: str
  ) -> None:
    """Verifies kernel/bias shapes, npz round-trip, and SHA-256 guard."""
    weights = _make_synthetic_weights(42, label_mode)
    self.assertEqual(weights.label_mode, label_mode)
    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, 'coin_weights.npz')
      arrays_dict: dict[str, np.ndarray] = {}
      for idx in range(5):
        arrays_dict[f'kernel_{idx}'] = weights.kernels[idx]
        arrays_dict[f'bias_{idx}'] = weights.biases[idx]
      np.savez(npz_path, **arrays_dict)

      loaded = coin_model.load_weights_from_npz(npz_path, None, label_mode)
      self.assertEqual(loaded.label_mode, label_mode)
      for idx in range(5):
        np.testing.assert_array_equal(loaded.kernels[idx], weights.kernels[idx])
        np.testing.assert_array_equal(loaded.biases[idx], weights.biases[idx])

      loaded_verified = coin_model.load_weights_from_npz(
          npz_path, loaded.weights_sha256, label_mode
      )
      self.assertEqual(loaded_verified.weights_sha256, loaded.weights_sha256)

      with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
        coin_model.load_weights_from_npz(npz_path, 'bad_hash', label_mode)

      bad_kernels = list(weights.kernels)
      bad_kernels[0] = np.zeros((99, 525), dtype=np.float32)
      with self.assertRaisesRegex(ValueError, 'kernel shape mismatch'):
        coin_model.validate_weights(bad_kernels, weights.biases, label_mode)

  @parameterized.named_parameters(
      ('olr', 'OLR', 15, 1),
      ('rsr', 'RSR', 13, 1),
  )
  def test_leaky_relu_slope_and_batch_invariance_by_mode(
      self, label_mode: str, n_in: int, n_out: int
  ) -> None:
    """Verifies LeakyReLU negative slope 0.2 across all 4 hidden layers."""
    k0 = np.zeros((n_in, 525), dtype=np.float32)
    b0 = np.zeros((525,), dtype=np.float32)
    k0[0, 0] = 1.0

    k1 = np.zeros((525, 178), dtype=np.float32)
    b1 = np.zeros((178,), dtype=np.float32)
    k1[0, 0] = 1.0

    k2 = np.zeros((178, 60), dtype=np.float32)
    b2 = np.zeros((60,), dtype=np.float32)
    k2[0, 0] = 1.0

    k3 = np.zeros((60, 20), dtype=np.float32)
    b3 = np.zeros((20,), dtype=np.float32)
    k3[0, 0] = 1.0

    k4 = np.zeros((20, n_out), dtype=np.float32)
    b4 = np.zeros((n_out,), dtype=np.float32)
    k4[0, 0] = -1000.0

    diag_weights = coin_model.CoinWeights(
        kernels=(k0, k1, k2, k3, k4),
        biases=(b0, b1, b2, b3, b4),
        weights_sha256='diag',
        label_mode=label_mode,
    )
    # Input -1.0 scales by 0.2**4 = 0.0016 across the 4 LeakyReLU layers,
    # then * -1000.0 -> +1.6 -> sigmoid(1.6).
    x = np.zeros((1, n_in), dtype=np.float32)
    x[0, 0] = -1.0
    expected_out_0 = float(1.0 / (1.0 + np.exp(-1.6)))
    out_tf = coin_model.forward_tf(x, diag_weights, 16)
    self.assertEqual(out_tf.shape, (1, n_out))
    self.assertAlmostEqual(float(out_tf[0, 0]), expected_out_0, places=6)

    rng = np.random.default_rng(123)
    rand_x = rng.uniform(0.0, 1.0, size=(300, n_in)).astype(np.float32)
    synth_w = _make_synthetic_weights(99, label_mode)
    pred_b17 = coin_model.forward_tf(rand_x, synth_w, 17)
    pred_b300 = coin_model.forward_tf(rand_x, synth_w, 300)
    self.assertEqual(pred_b17.shape, (300, n_out))
    self.assertEqual(pred_b300.shape, (300, n_out))
    np.testing.assert_allclose(pred_b17, pred_b300, atol=1e-6, rtol=1e-6)

  def test_split_npz_bundle_loading_and_manifest_integrity(
      self,
  ) -> None:
    """Verifies SPLIT `.npz` bundle loading and manifest integrity checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
      olr_w = _make_synthetic_weights(88, 'OLR')
      rsr_w = _make_synthetic_weights(89, 'RSR')
      olr_cfg = _make_valid_train_config('OLR')
      rsr_cfg = _make_valid_train_config('RSR')
      olr_cfg['weight_mode'] = 'clipped_inverse_freq'

      split_bundle_dir = os.path.join(tmpdir, 'coin_v1.1')
      olr_manifest = _write_synthetic_npz_subbundle(
          os.path.join(split_bundle_dir, 'olr'), olr_w, olr_cfg, 'coin_v1.1_olr'
      )
      rsr_manifest = _write_synthetic_npz_subbundle(
          os.path.join(split_bundle_dir, 'rsr'), rsr_w, rsr_cfg, 'coin_v1.1_rsr'
      )
      root_manifest_path = os.path.join(split_bundle_dir, 'MODEL_MANIFEST.json')
      with open(root_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'bundle_mode': 'SPLIT',
                'model_version': 'coin_v1.1',
                'olr_manifest': olr_manifest,
                'rsr_manifest': rsr_manifest,
            },
            f,
            indent=2,
            sort_keys=True,
        )

      split_bundle = coin_model.load_split_model_bundle(split_bundle_dir)
      loaded_olr_manifest = cast(
          Mapping[str, object], split_bundle.manifest['olr_manifest']
      )
      loaded_rsr_manifest = cast(
          Mapping[str, object], split_bundle.manifest['rsr_manifest']
      )
      self.assertEqual(loaded_olr_manifest['model_version'], 'coin_v1.1_olr')
      self.assertEqual(loaded_rsr_manifest['model_version'], 'coin_v1.1_rsr')
      for actual, expected in zip(
          split_bundle.olr_weights.kernels
          + split_bundle.olr_weights.biases
          + split_bundle.rsr_weights.kernels
          + split_bundle.rsr_weights.biases,
          olr_w.kernels + olr_w.biases + rsr_w.kernels + rsr_w.biases,
          strict=True,
      ):
        np.testing.assert_array_equal(actual, expected)

      with self.assertRaises(FileNotFoundError):
        coin_model.load_split_model_bundle(
            os.path.join(split_bundle_dir, 'olr')
        )

      # Verify missing coin_weights.npz fails fast with FileNotFoundError.
      missing_npz_dir = os.path.join(tmpdir, 'missing_npz')
      os.makedirs(missing_npz_dir, exist_ok=True)
      with open(
          os.path.join(missing_npz_dir, 'train_config.json'),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump(olr_cfg, f)
      with self.assertRaises(FileNotFoundError):
        coin_model.load_model_bundle(missing_npz_dir)

      # Verify tampered tensors_sha256 in subsystem manifest fails fast.
      olr_manifest_path = os.path.join(
          split_bundle_dir, 'olr', 'MODEL_MANIFEST.json'
      )
      with open(olr_manifest_path, 'r', encoding='utf-8') as f:
        orig_olr_manifest = json.load(f)
      bad_olr_manifest = dict(orig_olr_manifest)
      bad_olr_manifest['tensors_sha256'] = 'f' * 64
      with open(olr_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(bad_olr_manifest, f)
      with self.assertRaisesRegex(ValueError, 'NPZ tensor SHA-256 mismatch'):
        coin_model.load_model_bundle(os.path.join(split_bundle_dir, 'olr'))
      with open(olr_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(orig_olr_manifest, f)

      # Verify tampered root MODEL_MANIFEST.json sub-manifest digest fails fast.
      with open(root_manifest_path, 'r', encoding='utf-8') as f:
        tampered = json.load(f)
      tampered['olr_manifest']['weights_sha256'] = '0' * 64
      with open(root_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(tampered, f)
      with self.assertRaisesRegex(ValueError, 'SPLIT manifest mismatch'):
        coin_model.load_split_model_bundle(split_bundle_dir)

      # Verify missing sub-manifest field fails closed rather than skipping.
      del tampered['olr_manifest']['weights_sha256']
      with open(root_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(tampered, f)
      with self.assertRaises(KeyError):
        coin_model.load_split_model_bundle(split_bundle_dir)

  def test_feature_processors_match_training_formulas(self) -> None:
    """Verifies processors match TF formulas and invert cleanly."""
    rng = np.random.default_rng(55)
    vals = rng.normal(70.0, 20.0, size=(200,)).astype(np.float32)
    mean_v = 70.0
    std_v = 20.0

    pre_np = coin_features.standardize_preprocess(vals, mean_v, std_v)
    np.testing.assert_allclose(
        coin_features.standardize_preprocess(
            np.array(
                [
                    mean_v,
                    mean_v + 5.5 * std_v,
                    mean_v - 5.5 * std_v,
                    mean_v + 100.0 * std_v,
                    mean_v - 100.0 * std_v,
                ],
                dtype=np.float32,
            ),
            mean_v,
            std_v,
        ),
        np.array([0.5, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )

    clip_std = 5.5 * std_v
    pre_tf = tf.clip_by_value(
        ((vals - mean_v) + clip_std) / (2.0 * clip_std), 0.0, 1.0
    ).numpy()
    np.testing.assert_allclose(pre_np, pre_tf, atol=1e-6)

    # ZeroMinNormalize: verify both x_min == 0.0 (mean < 5.5 * std) and
    # x_min > 0.0 (mean > 5.5 * std).
    for mean_lbl, std_lbl in ((100.0, 150.0), (250.0, 40.0)):
      x_min, x_max = coin_features.zero_min_normalize_minmax(mean_lbl, std_lbl)
      mid_val = 0.5 * (x_min + x_max)
      probe = np.array(
          [x_min - 50.0, x_min, mid_val, x_max, x_max + 50.0],
          dtype=np.float32,
      )
      norm = coin_features.zero_min_normalize_preprocess(
          probe, mean_lbl, std_lbl
      )
      np.testing.assert_allclose(
          norm,
          np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32),
          atol=1e-6,
      )
      phys = coin_features.zero_min_normalize_postprocess(
          norm, mean_lbl, std_lbl
      )
      np.testing.assert_allclose(
          phys,
          np.array([x_min, x_min, mid_val, x_max, x_max], dtype=np.float32),
          atol=1e-4,
      )

    with self.assertRaisesRegex(ValueError, 'stddev_val'):
      coin_features.standardize_preprocess(vals, mean_v, 0.0)
    with self.assertRaisesRegex(ValueError, 'Degenerate'):
      coin_features.zero_min_normalize_minmax(-55.0, 10.0)

    sza_vals = np.linspace(0.5, 179.5, 50, dtype=np.float32)
    cos_vals = coin_features.cosine_degrees_preprocess(sza_vals)
    sza_rt = coin_features.cosine_degrees_postprocess(cos_vals)
    np.testing.assert_allclose(sza_rt, sza_vals, atol=1e-3)

  def test_fill_value_decodes_to_nan(self) -> None:
    """Verifies packed int16 _FillValue decodes to float32 NaN."""
    with tempfile.TemporaryDirectory() as tmpdir:
      nc_path = os.path.join(
          tmpdir,
          'OR_ABI-L1b-RadF-M6C13_G16_s20191521700396_e20191521709474_c20191521709531.nc',
      )
      rad = np.full((8, 8), 80.0, dtype=np.float32)
      rad[0, 0] = np.nan
      rad[3, 4] = np.nan
      _write_synthetic_abi_nc(
          nc_path,
          13,
          rad,
          'ABI Mode 6',
          -75.0,
          'mW m-2 sr-1 (cm-1)-1',
          True,
          'G16',
      )
      decoded, tl_id, _, _, _ = goes_abi_l1b.load_and_downscale_channel_nc(
          nc_path, 13, (8, 8), False
      )
      self.assertEqual(tl_id, 'ABI Mode 6')
      self.assertEqual(decoded.dtype, np.float32)
      self.assertEqual(int(np.sum(np.isnan(decoded))), 2)
      self.assertTrue(np.isnan(decoded[0, 0]))
      self.assertTrue(np.isnan(decoded[3, 4]))
      self.assertAlmostEqual(float(decoded[1, 1]), 80.0, places=2)

  @parameterized.named_parameters(
      ('factor_2', 2),
      ('factor_4', 4),
  )
  def test_downscale_block_mean(self, factor: int) -> None:
    """Verifies 2x2 and 4x4 block means propagate NaNs and reject bad shapes."""
    rng = np.random.default_rng(7 + factor)
    target_shape = (4, 6)
    arr = rng.uniform(
        10.0, 100.0, size=(target_shape[0] * factor, target_shape[1] * factor)
    ).astype(np.float32)
    arr[1, 1] = np.nan  # Inside block (0, 0)

    down = goes_abi_l1b.downscale_block_mean_to_target(arr, target_shape)
    self.assertEqual(down.shape, target_shape)
    self.assertEqual(down.dtype, np.float32)
    self.assertTrue(np.isnan(down[0, 0]))
    expected_01 = float(
        np.mean(arr[0:factor, factor : 2 * factor], dtype=np.float64)
    )
    self.assertAlmostEqual(float(down[0, 1]), expected_01, places=5)

    with self.assertRaisesRegex(ValueError, 'Downscale factor'):
      goes_abi_l1b.downscale_block_mean_to_target(
          np.ones((12, 12), dtype=np.float32), (4, 4)
      )
    with self.assertRaisesRegex(ValueError, 'Downscale factor'):
      goes_abi_l1b.downscale_block_mean_to_target(
          np.ones((8, 16), dtype=np.float32), (4, 4)
      )
    with self.assertRaisesRegex(ValueError, 'evenly divisible'):
      goes_abi_l1b.downscale_block_mean_to_target(
          np.ones((10, 10), dtype=np.float32), (4, 4)
      )
    with self.assertRaises(ZeroDivisionError):
      goes_abi_l1b.downscale_block_mean_to_target(
          np.ones((4, 4), dtype=np.float32), (0, 4)
      )

  def test_fixed_grid_geometry_and_in_range_count(self) -> None:
    """Verifies GOES East 2 km full-disk orientation, nadir, and pixel count."""
    with self.assertRaises(NotImplementedError):
      fixed_grid.build_area_def(
          'himawari', (8, 8), coin_constants.GOES_AREA_EXTENT_METERS
      )

    area_def = fixed_grid.build_area_def(
        coin_constants.SATELLITE_NAME,
        coin_constants.FULL_DISK_2KM_SHAPE,
        coin_constants.GOES_AREA_EXTENT_METERS,
    )
    geom = fixed_grid.compute_grid_geometry(
        area_def,
        coin_constants.SUBSATELLITE_LAT_DEGREES,
        coin_constants.SUBSATELLITE_LNG_DEGREES,
        coin_constants.MAX_EARTH_CENTRAL_ANGLE_RADIANS,
    )
    self.assertEqual(geom.area_def.shape, coin_constants.FULL_DISK_2KM_SHAPE)
    self.assertEqual(geom.n_in_range, 21199940)
    mid_r = coin_constants.FULL_DISK_2KM_SHAPE[0] // 2
    mid_c = coin_constants.FULL_DISK_2KM_SHAPE[1] // 2
    self.assertAlmostEqual(float(geom.lats[mid_r, mid_c]), 0.0, delta=0.02)
    self.assertAlmostEqual(float(geom.lngs[mid_r, mid_c]), -75.0, delta=0.02)
    self.assertGreater(
        float(geom.lats[mid_r - 100, mid_c]),
        float(geom.lats[mid_r + 100, mid_c]),
    )
    self.assertLess(
        float(geom.lngs[mid_r, mid_c - 100]),
        float(geom.lngs[mid_r, mid_c + 100]),
    )

    with self.assertRaisesRegex(ValueError, 'share one shape'):
      fixed_grid.earth_central_angle_radians(
          0.0, -75.0, np.zeros((4,)), np.zeros((4, 1))
      )

  def test_fixed_grid_caching_and_immutability(self) -> None:
    """Verifies GridGeometry cache reuse, array immutability, and eviction."""
    self.addCleanup(fixed_grid.clear_grid_geometry_cache)
    fixed_grid.clear_grid_geometry_cache()
    geom1 = fixed_grid.build_goes_east_grid(
        (12, 12), coin_constants.GOES_AREA_EXTENT_METERS
    )
    geom2 = fixed_grid.build_goes_east_grid(
        (12, 12), coin_constants.GOES_AREA_EXTENT_METERS
    )
    self.assertIs(geom1, geom2)
    self.assertFalse(geom1.lats.flags.writeable)
    self.assertFalse(geom1.lngs.flags.writeable)
    self.assertFalse(geom1.in_range_mask.flags.writeable)
    with self.assertRaises(ValueError):
      geom1.lats[0, 0] = 0.0

    fixed_grid.clear_grid_geometry_cache()
    geom3 = fixed_grid.build_goes_east_grid(
        (12, 12), coin_constants.GOES_AREA_EXTENT_METERS
    )
    self.assertIsNot(geom3, geom1)
    np.testing.assert_array_equal(geom3.lats, geom1.lats)
    np.testing.assert_array_equal(geom3.lngs, geom1.lngs)
    np.testing.assert_array_equal(geom3.in_range_mask, geom1.in_range_mask)

  def test_admission_allowlist_and_duplicate_selection(self) -> None:
    """Verifies G16/G19 acceptance and rejection of G18, Mode 4, and bad metadata."""
    g19_meta = goes_abi_l1b.parse_goes_filename(
        'OR_ABI-L1b-RadF-M6C01_G19_s20251001200000_e20251001209000_c20251001210000.nc'
    )
    self.assertEqual(g19_meta.platform_id, 'G19')
    with self.assertRaisesRegex(ValueError, 'platform_ID'):
      goes_abi_l1b.parse_goes_filename(
          'OR_ABI-L1b-RadF-M6C01_G18_s20251001200000_e20251001209000_c20251001210000.nc'
      )
    with self.assertRaisesRegex(ValueError, 'timeline_id / scan mode'):
      goes_abi_l1b.parse_goes_filename(
          'OR_ABI-L1b-RadF-M4C01_G16_s20191521700396_e20191521705000_c20191521705100.nc'
      )

    partial_files = [
        f'OR_ABI-L1b-RadF-M6C{ch:02d}_G16_s20191521700396_e20191521709474_c20191521709531.nc'
        for ch in range(1, 16)
    ]
    with self.assertRaisesRegex(ValueError, 'missing channels'):
      goes_abi_l1b.select_scantime_files(partial_files, 's20191521700396')

    all_16 = [
        f'dir/OR_ABI-L1b-RadF-M6C{ch:02d}_G16_s20191521700396_e20191521709474_c20191521709531.nc'
        for ch in range(1, 17)
    ]
    mixed_platforms = list(all_16[:-1]) + [
        'dir/OR_ABI-L1b-RadF-M6C16_G19_s20191521700396_e20191521709474_c20191521709531.nc'
    ]
    with self.assertRaisesRegex(ValueError, 'Inconsistent platform IDs'):
      goes_abi_l1b.select_scantime_files(mixed_platforms, 's20191521700396')

    newer_c01 = 'dir/OR_ABI-L1b-RadF-M6C01_G16_s20191521700396_e20191521709474_c20191521800000.nc'
    unrelated_nc = 'dir/coin_output_olr.nc'
    selected = goes_abi_l1b.select_scantime_files(
        all_16 + [newer_c01, unrelated_nc], 's20191521700396'
    )
    self.assertEqual(selected[1], newer_c01)

    with tempfile.TemporaryDirectory() as tmpdir:
      g19_nc = os.path.join(
          tmpdir,
          'OR_ABI-L1b-RadF-M6C13_G19_s20251521700204_e20251521709523_c20251521709571.nc',
      )
      _write_synthetic_abi_nc(
          g19_nc,
          13,
          np.full((8, 8), 70.0, dtype=np.float32),
          'ABI Mode 6',
          -75.0,
          'mW m-2 sr-1 (cm-1)-1',
          False,
          'G19',
      )
      g19_decoded, g19_tl, _, _, _ = goes_abi_l1b.load_and_downscale_channel_nc(
          g19_nc, 13, (8, 8), False
      )
      self.assertEqual(g19_tl, 'ABI Mode 6')
      self.assertEqual(g19_decoded.shape, (8, 8))

      mismatch_nc = os.path.join(
          tmpdir,
          'OR_ABI-L1b-RadF-M6C13_G19_s20251521710204_e20251521719523_c20251521719571.nc',
      )
      _write_synthetic_abi_nc(
          mismatch_nc,
          13,
          np.full((8, 8), 70.0, dtype=np.float32),
          'ABI Mode 6',
          -75.0,
          'mW m-2 sr-1 (cm-1)-1',
          False,
          'G16',
      )
      with self.assertRaisesRegex(ValueError, 'disagrees with filename'):
        goes_abi_l1b.load_and_downscale_channel_nc(
            mismatch_nc, 13, (8, 8), False
        )

      for idx, bad_lon in enumerate((-89.5, float('nan'))):
        bad_lon_nc = os.path.join(
            tmpdir,
            f'OR_ABI-L1b-RadF-M3C13_G16_s201720012000{idx}0_e20172001210000_c20172001211000.nc',
        )
        _write_synthetic_abi_nc(
            bad_lon_nc,
            13,
            np.full((8, 8), 70.0, dtype=np.float32),
            'ABI Mode 3',
            bad_lon,
            'mW m-2 sr-1 (cm-1)-1',
            False,
            'G16',
        )
        with self.assertRaisesRegex(
            ValueError, 'longitude_of_projection_origin'
        ):
          goes_abi_l1b.load_and_downscale_channel_nc(
              bad_lon_nc, 13, (8, 8), False
          )

  @parameterized.named_parameters(
      ('rsr', 'RSR'),
      ('olr', 'OLR'),
  )
  def test_night_rsr_bypass_boundary_by_mode(self, label_mode: str) -> None:
    """Verifies strict SZA > 90.0 deg night bypass boundary per mode."""
    sza_test_deg = np.array([89.99, 90.0, 90.01], dtype=np.float32)
    cos_sza = coin_features.cosine_degrees_preprocess(sza_test_deg)
    cfg = _make_valid_train_config(label_mode)
    n_in = len(coin_constants.SORTED_INPUT_NAMES_BY_MODE[label_mode])

    model_in = np.zeros((3, n_in), dtype=np.float32)
    model_in[:, coin_constants.SZA_INDEX_BY_MODE[label_mode]] = cos_sza
    row_vals = [0.7] if label_mode == 'RSR' else [0.2]
    norm_preds = np.tile(np.array(row_vals, dtype=np.float32), (3, 1))
    out = coin_model.apply_night_rsr_bypass_and_postprocess(
        norm_preds, model_in, cfg
    )
    if label_mode == 'RSR':
      expected_rsr_day = float(
          coin_features.zero_min_normalize_postprocess(
              np.array([0.7], dtype=np.float32),
              float(cfg['ceres_rsr_mean']),
              float(cfg['ceres_rsr_stddev']),
          )[0]
      )
      np.testing.assert_allclose(
          out['rsr'],
          np.array([expected_rsr_day, expected_rsr_day, 0.0], dtype=np.float32),
          atol=1e-4,
      )
      bad_cos_in = model_in.copy()
      bad_cos_in[0, coin_constants.SZA_INDEX_BY_MODE[label_mode]] = np.inf
      with self.assertRaisesRegex(ValueError, 'Non-finite solar zenith angle'):
        coin_model.apply_night_rsr_bypass_and_postprocess(
            norm_preds, bad_cos_in, cfg
        )
    else:
      self.assertNotIn('rsr', out)
    if label_mode == 'OLR':
      expected_olr = float(
          coin_features.zero_min_normalize_postprocess(
              np.array([0.2], dtype=np.float32),
              float(cfg['ceres_olr_mean']),
              float(cfg['ceres_olr_stddev']),
          )[0]
      )
      np.testing.assert_allclose(
          out['olr'],
          np.full((3,), expected_olr, dtype=np.float32),
          atol=1e-4,
      )
    else:
      self.assertNotIn('olr', out)

    nan_preds = norm_preds.copy()
    nan_preds[2, 0] = np.nan
    with self.assertRaisesRegex(ValueError, 'Non-finite normalized_preds'):
      coin_model.apply_night_rsr_bypass_and_postprocess(
          nan_preds, model_in, cfg
      )

  def test_finite_input_gate(self) -> None:
    """Verifies that a single NaN in any feature triggers an error."""
    cfg = _make_valid_train_config('OLR')
    area_def = fixed_grid.build_area_def(
        coin_constants.SATELLITE_NAME,
        (8, 8),
        coin_constants.GOES_AREA_EXTENT_METERS,
    )
    lats = np.zeros((8, 8), dtype=np.float64)
    lngs = np.full((8, 8), -75.0, dtype=np.float64)
    lats[2, 2] = np.nan
    in_range = np.ones((8, 8), dtype=bool)
    bad_geom = fixed_grid.GridGeometry(
        lats=lats,
        lngs=lngs,
        in_range_mask=in_range,
        n_in_range=64,
        area_def=area_def,
    )
    channels = {
        ch: np.full((8, 8), 20.0, dtype=np.float32)
        for ch in coin_constants.GOES_CHANNELS
    }
    dt = datetime.datetime(2019, 6, 1, 17, 0, tzinfo=datetime.timezone.utc)
    with self.assertRaisesRegex(ValueError, 'Finite-input gate failed'):
      coin_features.build_feature_matrix(cfg, channels, bad_geom, dt)

  def test_split_finite_masking(self) -> None:
    """Verifies per-subsystem finite masking (LW ch 8..16 vs SW ch 1..7)."""
    area_def = fixed_grid.build_area_def(
        coin_constants.SATELLITE_NAME,
        (8, 8),
        coin_constants.GOES_AREA_EXTENT_METERS,
    )
    lats = np.zeros((8, 8), dtype=np.float64)
    lngs = np.full((8, 8), -75.0, dtype=np.float64)
    in_range = np.ones((8, 8), dtype=bool)
    geom = fixed_grid.GridGeometry(
        lats=lats,
        lngs=lngs,
        in_range_mask=in_range,
        n_in_range=64,
        area_def=area_def,
    )

    channels = {
        ch: np.full((8, 8), 20.0, dtype=np.float32)
        for ch in coin_constants.GOES_CHANNELS
    }
    channels[2][1, 2] = np.nan
    channels[13][4, 5] = np.nan
    channels[1][6, 6] = np.nan
    channels[8][6, 6] = np.nan

    dt = datetime.datetime(2019, 6, 1, 17, 0, tzinfo=datetime.timezone.utc)
    olr_cfg = _make_valid_train_config('OLR')
    rsr_cfg = _make_valid_train_config('RSR')

    split_prep = coin_features.build_split_feature_matrices(
        olr_cfg, rsr_cfg, channels, geom, dt
    )
    self.assertEqual(split_prep.olr_prepared.n_valid, 62)
    self.assertEqual(split_prep.rsr_prepared.n_valid, 62)
    self.assertEqual(split_prep.n_valid_union, 63)
    self.assertEqual(split_prep.olr_prepared.model_input.shape, (62, 15))
    self.assertEqual(split_prep.rsr_prepared.model_input.shape, (62, 13))

    olr_w = _make_synthetic_weights(11, 'OLR')
    rsr_w = _make_synthetic_weights(12, 'RSR')
    preds = coin_model.predict_split_on_grid(
        split_prep,
        olr_w,
        olr_cfg,
        rsr_w,
        rsr_cfg,
        (8, 8),
        32,
    )
    self.assertEqual(preds.n_valid_olr, 62)
    self.assertEqual(preds.n_valid_rsr, 62)
    self.assertEqual(preds.n_valid, 63)

    self.assertTrue(np.isfinite(preds.predicted_olr[1, 2]))
    self.assertTrue(np.isnan(preds.predicted_rsr[1, 2]))
    self.assertEqual(
        int(preds.inference_mask[1, 2]), coin_constants.MASK_OLR_BIT
    )

    self.assertTrue(np.isnan(preds.predicted_olr[4, 5]))
    self.assertTrue(np.isfinite(preds.predicted_rsr[4, 5]))
    self.assertEqual(
        int(preds.inference_mask[4, 5]), coin_constants.MASK_RSR_BIT
    )

    self.assertTrue(np.isnan(preds.predicted_olr[6, 6]))
    self.assertTrue(np.isnan(preds.predicted_rsr[6, 6]))
    self.assertTrue(np.isnan(preds.solar_zenith_deg[6, 6]))
    self.assertEqual(int(preds.inference_mask[6, 6]), 0)

    self.assertTrue(np.isfinite(preds.predicted_olr[0, 0]))
    self.assertTrue(np.isfinite(preds.predicted_rsr[0, 0]))
    self.assertEqual(
        int(preds.inference_mask[0, 0]), coin_constants.MASK_BOTH_BITS
    )

  def test_config_schema_validation(self) -> None:
    """Verifies strict validation of train_config.json for OLR/RSR."""
    for mode in ('OLR', 'RSR'):
      valid = _make_valid_train_config(mode)
      self.assertEqual(
          coin_features.validate_train_config(valid)['label_mode'], mode
      )

    sat_name = coin_constants.SATELLITE_NAME
    missing_lw = _make_valid_train_config('OLR')
    del missing_lw[f'{sat_name}_channel_13_mean']
    with self.assertRaisesRegex(ValueError, f'{sat_name}_channel_13_mean'):
      coin_features.validate_train_config(missing_lw)

    minimal_olr = _make_valid_train_config('OLR')
    del minimal_olr[f'{sat_name}_channel_1_mean']
    del minimal_olr[f'{sat_name}_channel_1_stddev']
    del minimal_olr['ceres_rsr_mean']
    del minimal_olr['ceres_rsr_stddev']
    self.assertEqual(
        coin_features.validate_train_config(minimal_olr)['label_mode'], 'OLR'
    )

    missing_sw = _make_valid_train_config('RSR')
    del missing_sw[f'{sat_name}_channel_1_mean']
    with self.assertRaisesRegex(ValueError, f'{sat_name}_channel_1_mean'):
      coin_features.validate_train_config(missing_sw)

    extra = _make_valid_train_config('OLR')
    extra['unknown_hyperparam'] = 123
    with self.assertRaisesRegex(ValueError, 'unknown_hyperparam'):
      coin_features.validate_train_config(extra)

    bad_bool = _make_valid_train_config('OLR')
    bad_bool['dropout'] = False
    with self.assertRaisesRegex(ValueError, 'dropout'):
      coin_features.validate_train_config(bad_bool)

    for unsupported_mode in ('DUAL', 'SURFACE_FLUX'):
      bad_mode = _make_valid_train_config('OLR')
      bad_mode['label_mode'] = unsupported_mode
      with self.assertRaisesRegex(ValueError, 'label_mode'):
        coin_features.validate_train_config(bad_mode)

    nested = _make_valid_train_config('OLR')
    nested['args'] = {'save_path': 'unused'}
    with self.assertRaisesRegex(ValueError, 'Nested dictionary'):
      coin_features.validate_train_config(nested)

  def test_radiance_range_guard(self) -> None:
    """Verifies rejection of Kelvin brightness temperature and +/-inf radiances."""
    with tempfile.TemporaryDirectory() as tmpdir:
      bt_nc = os.path.join(
          tmpdir,
          'OR_ABI-L1b-RadF-M6C13_G16_s20191521700396_e20191521709474_c20191521709531.nc',
      )
      bt_vals = np.linspace(250.0, 290.0, 64, dtype=np.float32).reshape(8, 8)
      _write_synthetic_abi_nc(
          bt_nc,
          13,
          bt_vals,
          'ABI Mode 6',
          -75.0,
          'mW m-2 sr-1 (cm-1)-1',
          False,
          'G16',
      )
      with self.assertRaisesRegex(ValueError, 'outside physical bounds'):
        goes_abi_l1b.load_and_downscale_channel_nc(bt_nc, 13, (8, 8), False)

      inf_nc = os.path.join(
          tmpdir,
          'OR_ABI-L1b-RadF-M6C13_G16_s20191521800396_e20191521809474_c20191521809531.nc',
      )
      inf_vals = np.full((8, 8), 70.0, dtype=np.float32)
      inf_vals[0, 0] = np.inf
      _write_synthetic_abi_nc(
          inf_nc,
          13,
          inf_vals,
          'ABI Mode 6',
          -75.0,
          'mW m-2 sr-1 (cm-1)-1',
          False,
          'G16',
      )
      with self.assertRaisesRegex(ValueError, 'infinite value'):
        goes_abi_l1b.load_and_downscale_channel_nc(inf_nc, 13, (8, 8), False)

  @parameterized.named_parameters(
      ('netcdf4', 'nc'),
      ('zarr_v2', 'zarr'),
  )
  def test_derive_and_write_flux_output_datasets(self, ext: str) -> None:
    """Verifies path derivation, numeric serialization, Zarr v2, and overwrite."""
    olr_p, rsr_p = write_output.derive_flux_output_paths(f'out/scan.{ext}')
    self.assertEqual(olr_p, f'out/scan_olr.{ext}')
    self.assertEqual(rsr_p, f'out/scan_rsr.{ext}')
    with self.assertRaisesRegex(ValueError, '_olr'):
      write_output.derive_flux_output_paths(f'out/scan_olr.{ext}')
    with self.assertRaisesRegex(ValueError, '_rsr'):
      write_output.derive_flux_output_paths(f'out/scan_rsr.{ext}')
    with self.assertRaisesRegex(ValueError, 'Unsupported'):
      write_output.derive_flux_output_paths('out/scan.parquet')

    shape = (4, 4)
    pred_olr = np.linspace(210.0, 270.0, 16, dtype=np.float32).reshape(shape)
    pred_rsr = np.linspace(50.0, 350.0, 16, dtype=np.float32).reshape(shape)
    sza = np.linspace(15.0, 65.0, 16, dtype=np.float32).reshape(shape)
    olr_mask = np.ones(shape, dtype=np.uint8)
    rsr_mask = np.ones(shape, dtype=np.uint8)

    pred_rsr[0, 1] = np.nan
    rsr_mask[0, 1] = 0
    pred_olr[1, 0] = np.nan
    olr_mask[1, 0] = 0
    pred_olr[2, 0] = np.nan
    olr_mask[2, 0] = 0
    pred_olr[3, 3] = np.nan
    pred_rsr[3, 3] = np.nan
    sza[3, 3] = np.nan
    olr_mask[3, 3] = 0
    rsr_mask[3, 3] = 0
    combined_mask = (
        (olr_mask * np.uint8(coin_constants.MASK_OLR_BIT))
        | (rsr_mask * np.uint8(coin_constants.MASK_RSR_BIT))
    ).astype(np.uint8)

    preds = coin_model.GridPredictions(
        predicted_rsr=pred_rsr,
        predicted_olr=pred_olr,
        solar_zenith_deg=sza,
        olr_inference_mask=olr_mask,
        rsr_inference_mask=rsr_mask,
        inference_mask=combined_mask,
        n_in_range=15,
        n_valid_olr=13,
        n_valid_rsr=14,
        n_valid=15,
    )
    dt = datetime.datetime(
        2019, 6, 1, 17, 0, 39, 5, tzinfo=datetime.timezone.utc
    )
    ds_olr_mem = write_output.build_flux_output_dataset(
        'olr',
        preds,
        coin_constants.GOES_AREA_EXTENT_METERS,
        dt,
        's20191521700396',
        'ABI Mode 6',
        'split_v1_olr',
        'sha_olr',
        'SPLIT',
    )
    ds_rsr_mem = write_output.build_flux_output_dataset(
        'rsr',
        preds,
        coin_constants.GOES_AREA_EXTENT_METERS,
        dt,
        's20191521700396',
        'ABI Mode 6',
        'split_v1_rsr',
        'sha_rsr',
        'SPLIT',
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      base_out = os.path.join(tmpdir, f'flux_test.{ext}')
      # Write twice to exercise atomic overwrite and verify zero leftover temps.
      _ = write_output.write_flux_datasets_atomic(
          ds_olr_mem, ds_rsr_mem, base_out
      )
      olr_written, rsr_written = write_output.write_flux_datasets_atomic(
          ds_olr_mem, ds_rsr_mem, base_out
      )
      remaining = sorted(os.listdir(tmpdir))
      self.assertEqual(
          remaining, [f'flux_test_olr.{ext}', f'flux_test_rsr.{ext}']
      )
      if ext == 'zarr':
        self.assertTrue(os.path.isfile(os.path.join(olr_written, '.zmetadata')))
        self.assertTrue(os.path.isfile(os.path.join(olr_written, '.zgroup')))

      open_fn = (
          (lambda p: xarray.open_dataset(p, engine='netcdf4'))
          if ext == 'nc'
          else (lambda p: xarray.open_zarr(p, consolidated=True))
      )
      with open_fn(olr_written) as ds_o, open_fn(rsr_written) as ds_r:
        np.testing.assert_allclose(
            ds_o['predicted_olr'].values, pred_olr, equal_nan=True
        )
        np.testing.assert_allclose(
            ds_r['predicted_rsr'].values, pred_rsr, equal_nan=True
        )
        np.testing.assert_allclose(
            ds_o['solar_zenith_deg'].values, sza, equal_nan=True
        )
        self.assertEqual(
            ds_o['t'].values, np.datetime64('2019-06-01T17:00:39.000005000')
        )
        self.assertEqual(int(ds_o['inference_mask'].values[0, 1]), 1)
        self.assertEqual(int(ds_o['inference_mask'].values[1, 0]), 0)
        self.assertEqual(int(ds_r['inference_mask'].values[0, 1]), 0)
        self.assertEqual(int(ds_r['inference_mask'].values[1, 0]), 1)
        self.assertEqual(
            int(ds_o['combined_inference_bitmask'].values[0, 1]),
            coin_constants.MASK_OLR_BIT,
        )
        self.assertEqual(
            int(ds_o['combined_inference_bitmask'].values[1, 0]),
            coin_constants.MASK_RSR_BIT,
        )
        self.assertEqual(
            int(ds_o['combined_inference_bitmask'].values[0, 0]),
            coin_constants.MASK_BOTH_BITS,
        )
        self.assertEqual(
            int(ds_o['combined_inference_bitmask'].values[3, 3]), 0
        )
        self.assertEqual(ds_o.attrs['n_valid_pixels'], 13)
        self.assertEqual(ds_o.attrs['n_valid_union_pixels'], 15)
        self.assertEqual(ds_o.attrs['model_version'], 'split_v1_olr')
        self.assertEqual(ds_o.attrs['weights_sha256'], 'sha_olr')
        self.assertEqual(ds_r.attrs['n_valid_pixels'], 14)
        self.assertEqual(ds_r.attrs['n_valid_union_pixels'], 15)
        self.assertEqual(ds_r.attrs['model_version'], 'split_v1_rsr')
        self.assertEqual(ds_r.attrs['weights_sha256'], 'sha_rsr')

  @parameterized.named_parameters(
      ('nc_split', 'nc'),
      ('zarr_split', 'zarr'),
  )
  def test_end_to_end_miniature_scan(self, ext: str) -> None:
    """End-to-end miniature SPLIT pipeline run for NetCDF4 and Zarr v2."""
    target_shape = (12, 12)
    timekey = 's20191521700396'
    with tempfile.TemporaryDirectory() as tmpdir:
      scan_dir = os.path.join(tmpdir, 'abi_scan')
      os.makedirs(scan_dir, exist_ok=True)
      for ch in coin_constants.GOES_CHANNELS:
        factor = coin_constants.CHANNEL_DOWNSCALE_FACTORS[ch]
        native_shape = (target_shape[0] * factor, target_shape[1] * factor)
        rad = np.linspace(
            float(ch),
            float(ch) + 12.0,
            native_shape[0] * native_shape[1],
            dtype=np.float32,
        ).reshape(native_shape)
        if ch == 4:
          rad[5, 5] = np.nan
        elif ch == 13:
          rad[6, 5] = np.nan
          rad[7, 5] = np.nan
        units = 'W m-2 sr-1 um-1' if ch <= 6 else 'mW m-2 sr-1 (cm-1)-1'
        fname = (
            f'OR_ABI-L1b-RadF-M6C{ch:02d}_G16_{timekey}_'
            'e20191521709474_c20191521709531.nc'
        )
        _write_synthetic_abi_nc(
            os.path.join(scan_dir, fname),
            ch,
            rad,
            'ABI Mode 6',
            -75.0,
            units,
            False,
            'G16',
        )

      model_dir = os.path.join(tmpdir, 'model')
      for sub, mode_str, seed in (('olr', 'OLR', 201), ('rsr', 'RSR', 202)):
        sub_dir = os.path.join(model_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        with open(
            os.path.join(sub_dir, 'train_config.json'), 'w', encoding='utf-8'
        ) as f:
          json.dump(_make_valid_train_config(mode_str), f)
        synth_w = _make_synthetic_weights(seed, mode_str)
        np.savez(
            os.path.join(sub_dir, 'coin_weights.npz'),
            **{f'kernel_{i}': synth_w.kernels[i] for i in range(5)},
            **{f'bias_{i}': synth_w.biases[i] for i in range(5)},
        )

      out_path = os.path.join(tmpdir, f'output.{ext}')
      source = goes_abi_l1b.LocalDirectoryScanSource(scan_dir)
      summary = run_single_scantime.run_single_scantime_inference(
          timekey,
          model_dir,
          out_path,
          64,
          scan_dir,
          source,
          target_shape,
          False,
          2,
      )
      self.assertEqual(summary.scan_timekey, timekey)
      self.assertEqual(summary.timeline_id, 'ABI Mode 6')
      self.assertEqual(summary.bundle_mode, 'SPLIT')
      self.assertGreater(summary.n_valid_olr, 0)
      self.assertGreater(summary.n_valid_rsr, 0)
      self.assertGreater(summary.n_valid_union, 0)
      self.assertEqual(summary.n_valid_olr, summary.n_valid_union - 2)
      self.assertEqual(summary.n_valid_rsr, summary.n_valid_union - 1)

      open_fn = (
          (lambda p: xarray.open_dataset(p, engine='netcdf4'))
          if ext == 'nc'
          else (lambda p: xarray.open_zarr(p, consolidated=True))
      )
      with (
          open_fn(summary.olr_output_path) as ds_olr,
          open_fn(summary.rsr_output_path) as ds_rsr,
      ):
        for (
            ds,
            flux_type,
            present_var,
            absent_var,
            exp_ver,
            exp_sha,
            exp_valid,
        ) in (
            (
                ds_olr,
                'olr',
                'predicted_olr',
                'predicted_rsr',
                summary.olr_model_version,
                summary.olr_weights_sha256,
                summary.n_valid_olr,
            ),
            (
                ds_rsr,
                'rsr',
                'predicted_rsr',
                'predicted_olr',
                summary.rsr_model_version,
                summary.rsr_weights_sha256,
                summary.n_valid_rsr,
            ),
        ):
          for req_var in (
              present_var,
              'solar_zenith_deg',
              'inference_mask',
              'combined_inference_bitmask',
              'goes_imager_projection',
              'x_image_bounds',
              'y_image_bounds',
          ):
            self.assertIn(req_var, ds.variables)
          self.assertNotIn(absent_var, ds.variables)
          vals = ds[present_var].values
          mask = ds['inference_mask'].values
          self.assertEqual(vals.shape, target_shape)
          self.assertEqual(int(np.sum(mask == 1)), exp_valid)
          self.assertTrue(np.all(np.isfinite(vals[mask == 1])))
          self.assertTrue(np.all(np.isnan(vals[mask == 0])))
          self.assertTrue(np.all(vals[mask == 1] >= 0.0))
          self.assertEqual(ds.attrs['flux_type'], flux_type)
          self.assertEqual(ds.attrs['bundle_mode'], 'SPLIT')
          self.assertEqual(ds.attrs['model_version'], exp_ver)
          self.assertEqual(ds.attrs['weights_sha256'], exp_sha)
          self.assertEqual(ds.attrs['scan_timekey'], timekey)
          self.assertEqual(ds.attrs['timeline_id'], 'ABI Mode 6')
          self.assertEqual(ds.attrs['time_convention'], 'scan_start')

        self.assertEqual(int(ds_olr['inference_mask'].values[5, 5]), 1)
        self.assertEqual(int(ds_rsr['inference_mask'].values[5, 5]), 0)
        self.assertEqual(
            int(ds_olr['combined_inference_bitmask'].values[5, 5]),
            coin_constants.MASK_OLR_BIT,
        )
        self.assertEqual(int(ds_olr['inference_mask'].values[6, 5]), 0)
        self.assertEqual(int(ds_rsr['inference_mask'].values[6, 5]), 1)
        self.assertEqual(
            int(ds_rsr['combined_inference_bitmask'].values[6, 5]),
            coin_constants.MASK_RSR_BIT,
        )

  def test_datetime_from_timekey_and_gcs_crc32c_verification(self) -> None:
    """Verifies timekey validation and GcsAnonymousScanSource CRC32C checks."""
    leap_dt = goes_abi_l1b.datetime_from_timekey('s20203661200000')
    self.assertEqual((leap_dt.year, leap_dt.month, leap_dt.day), (2020, 12, 31))

    with self.assertRaisesRegex(ValueError, 'non-leap year'):
      goes_abi_l1b.datetime_from_timekey('s20193661200000')
    with self.assertRaisesRegex(ValueError, 'second=60'):
      goes_abi_l1b.datetime_from_timekey('s20191521203600')

    # RFC 3720 CRC-32C check value for b'123456789' is 0xE3069283 ('4waSgw==').
    payload = b'123456789'
    expected_crc_b64 = '4waSgw=='

    class _FakeBlob:

      def __init__(self, name: str, crc32c: str, size: int) -> None:
        self.name = name
        self.crc32c = crc32c
        self.size = size

      def download_to_filename(self, filename: str) -> None:
        with open(filename, 'wb') as f:
          f.write(payload)

    class _FakeClient:

      def __init__(self, expected_bucket: str, blobs: list[_FakeBlob]) -> None:
        self._expected_bucket = expected_bucket
        self._blobs = blobs
        self.closed = False

      def bucket(self, name: str) -> str:
        return name

      def list_blobs(self, bucket: str, prefix: str) -> list[_FakeBlob]:
        if bucket != self._expected_bucket:
          raise ValueError(f'Unexpected bucket {bucket!r}')
        return [b for b in self._blobs if b.name.startswith(prefix)]

      def close(self) -> None:
        self.closed = True

    timekey = 's20191521700396'
    prefix = 'ABI-L1b-RadF/2019/152/17/'
    good_blobs = [
        _FakeBlob(
            f'{prefix}OR_ABI-L1b-RadF-M6C{ch:02d}_G16_{timekey}_'
            'e20191521709474_c20191521709531.nc',
            expected_crc_b64,
            len(payload),
        )
        for ch in coin_constants.GOES_CHANNELS
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
      ok_dir = os.path.join(tmpdir, 'ok')
      fake_client = _FakeClient('test-bucket', good_blobs)
      with mock.patch.object(
          goes_abi_l1b.storage.Client,
          'create_anonymous_client',
          return_value=fake_client,
      ):
        source = goes_abi_l1b.GcsAnonymousScanSource('test-bucket', 4)
        staged = source.fetch_scan_files(timekey, ok_dir)
      self.assertLen(staged, 16)
      self.assertTrue(fake_client.closed)

      bad_dir = os.path.join(tmpdir, 'bad')
      bad_crc_blobs = list(good_blobs)
      bad_crc_blobs[0] = _FakeBlob(good_blobs[0].name, 'bad_crc=', len(payload))
      bad_client = _FakeClient('test-bucket', bad_crc_blobs)
      with mock.patch.object(
          goes_abi_l1b.storage.Client,
          'create_anonymous_client',
          return_value=bad_client,
      ):
        with self.assertRaisesRegex(IOError, 'CRC32C mismatch'):
          source.fetch_scan_files(timekey, bad_dir)
      self.assertTrue(bad_client.closed)
      self.assertNotIn(
          os.path.basename(good_blobs[0].name), os.listdir(bad_dir)
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='extra_positional_arg',
          argv=['run_single_scantime', 'extra'],
          flag_overrides={},
          expected_regex='Unexpected positional',
      ),
      dict(
          testcase_name='missing_required_flags',
          argv=['run_single_scantime'],
          flag_overrides={
              'scan_timekey': None,
              'model_dir': None,
              'output_path': None,
              'batch_size': None,
              'work_dir': None,
              'input_source': None,
              'max_decode_workers': None,
          },
          expected_regex='--scan_timekey',
      ),
      dict(
          testcase_name='gcs_missing_bucket',
          argv=['run_single_scantime'],
          flag_overrides={
              'scan_timekey': 's20191521700447',
              'model_dir': 'model',
              'output_path': 'out.nc',
              'batch_size': 64,
              'work_dir': 'work',
              'input_source': 'gcs',
              'gcs_bucket': None,
              'max_decode_workers': 2,
          },
          expected_regex='--gcs_bucket is required',
      ),
      dict(
          testcase_name='local_with_gcs_bucket_set',
          argv=['run_single_scantime'],
          flag_overrides={
              'scan_timekey': 's20191521700447',
              'model_dir': 'model',
              'output_path': 'out.nc',
              'batch_size': 64,
              'work_dir': 'work',
              'input_source': 'local',
              'gcs_bucket': 'gcp-public-data-goes-16',
              'max_decode_workers': 2,
          },
          expected_regex='--gcs_bucket must not be set',
      ),
  )
  def test_run_single_scantime_cli_validation_errors(
      self,
      argv: list[str],
      flag_overrides: dict[str, object],
      expected_regex: str,
  ) -> None:
    """Verifies UsageError on invalid CLI invocations of run_single_scantime."""
    with flagsaver.flagsaver(**flag_overrides):
      with self.assertRaisesRegex(app.UsageError, expected_regex):
        run_single_scantime.main(argv)

  def test_train_config_aux_keys_and_args_flattening(self) -> None:
    """Verifies auxiliary config keys, nested `args` stripping, and guards."""
    new_aux_keys = {
        'check_l2_sentinel_labels': True,
        'early_stopping_patience': ARBITRARY_INT,
        'min_learning_rate': ARBITRARY_FLOAT,
        'shuffle_buffer_size': ARBITRARY_INT,
        'steps_per_execution': ARBITRARY_INT,
        'tfrecord_read_buffer_bytes': ARBITRARY_INT,
        'train_interleave_block_length': ARBITRARY_INT,
        'train_interleave_cycle_length': ARBITRARY_INT,
        'train_prefetch_batches': ARBITRARY_INT,
        'validation_shuffle_seed': ARBITRARY_INT,
    }

    def build_raw_cfg(mode: str) -> dict[str, object]:
      cfg = _make_valid_train_config(mode)
      cfg.update(new_aux_keys)
      cfg['args'] = {
          'save_path': f'synthetic_model_out/{mode}',
          'train_config': {'path': f'synthetic_configs/{mode}.json'},
      }
      return cfg

    with tempfile.TemporaryDirectory() as tmpdir:
      bad_args_scalar = build_raw_cfg('OLR')
      bad_args_scalar['args'] = ARBITRARY_INT
      bad_args_path = os.path.join(tmpdir, 'bad_args_scalar.json')
      with open(bad_args_path, 'w', encoding='utf-8') as f:
        json.dump(bad_args_scalar, f)
      with self.assertRaisesRegex(ValueError, 'args'):
        coin_features.load_train_config(bad_args_path)

      bad_other_nested = build_raw_cfg('OLR')
      bad_other_nested['other_nested'] = {'foo': 'bar'}
      bad_nested_path = os.path.join(tmpdir, 'bad_other_nested.json')
      with open(bad_nested_path, 'w', encoding='utf-8') as f:
        json.dump(bad_other_nested, f)
      with self.assertRaisesRegex(ValueError, 'Nested dictionary'):
        coin_features.load_train_config(bad_nested_path)

      for mode in ('OLR', 'RSR'):
        cfg_path = os.path.join(tmpdir, f'{mode.lower()}_train_config.json')
        with open(cfg_path, 'w', encoding='utf-8') as f:
          json.dump(build_raw_cfg(mode), f)
        loaded = coin_features.load_train_config(cfg_path)
        self.assertNotIn('args', loaded)
        self.assertEqual(loaded['label_mode'], mode)
        for aux_k, aux_v in new_aux_keys.items():
          self.assertEqual(loaded[aux_k], aux_v)

  @parameterized.named_parameters(
      ('olr', 'OLR', 15),
      ('rsr', 'RSR', 13),
  )
  def test_solar_azimuth_sin_cos_continuity_and_unit_circle(
      self, mode: str, expected_cols: int
  ) -> None:
    """Verifies sin^2 + cos^2 == 1 and continuity across 0 <-> 2*pi via public API."""
    # At 2019-06-01 17:00:44.7 UTC, subsolar latitude is ~ +22.012 deg. At
    # latitude 0.0 deg (south of the subsolar latitude), crossing the exact
    # solar-noon meridian (hour_angle == 0) wraps raw solar azimuth across
    # 0 <-> 2*pi (due North).
    dt = datetime.datetime(
        2019, 6, 1, 17, 0, 44, 700000, tzinfo=datetime.timezone.utc
    )
    sub_lng = -math.degrees(
        float(solar_geometry.hour_angle_radians(0.0, dt.timestamp()))
    )
    lats = np.zeros((1, 4), dtype=np.float64)
    lngs = np.array(
        [[sub_lng - 20.0, sub_lng - 1e-4, sub_lng + 1e-4, sub_lng + 20.0]],
        dtype=np.float64,
    )
    area_def = fixed_grid.build_area_def(
        coin_constants.SATELLITE_NAME,
        (1, 4),
        coin_constants.GOES_AREA_EXTENT_METERS,
    )
    geom = fixed_grid.GridGeometry(
        lats=lats,
        lngs=lngs,
        in_range_mask=np.ones((1, 4), dtype=bool),
        n_in_range=4,
        area_def=area_def,
    )
    channels = {
        ch: np.full((1, 4), 25.0, dtype=np.float32)
        for ch in coin_constants.GOES_CHANNELS
    }
    raw_saz = solar_geometry.azimuth_angle_radians(lats, lngs, dt.timestamp())
    # Confirm pixels 1 and 2 straddle the 2*pi <-> 0 wrap (> 6.0 rad jump).
    self.assertGreater(abs(float(raw_saz[0, 1] - raw_saz[0, 2])), 6.0)

    cfg = _make_valid_train_config(mode)
    prep = coin_features.build_feature_matrix(cfg, channels, geom, dt)
    self.assertEqual(prep.model_input.shape, (4, expected_cols))
    sat_name = coin_constants.SATELLITE_NAME
    names = coin_constants.SORTED_INPUT_NAMES_BY_MODE[mode]
    sin_idx = names.index(f'{sat_name}_sin_solar_azimuths')
    cos_idx = names.index(f'{sat_name}_cos_solar_azimuths')

    unit_circle = np.square(prep.model_input[:, sin_idx]) + np.square(
        prep.model_input[:, cos_idx]
    )
    np.testing.assert_allclose(
        unit_circle, np.ones((4,), dtype=np.float32), atol=1e-6
    )
    # Periodic (sin, cos) features are continuous across the wrap boundary.
    np.testing.assert_allclose(
        prep.model_input[1], prep.model_input[2], atol=2e-5
    )
    self.assertAlmostEqual(float(prep.model_input[1, cos_idx]), 1.0, places=4)
    self.assertAlmostEqual(float(prep.model_input[1, sin_idx]), 0.0, places=3)

  @parameterized.named_parameters(
      ('olr', 'OLR'),
      ('rsr', 'RSR'),
  )
  def test_longitude_positive_unwrap_standardization(self, mode: str) -> None:
    """Verifies positive longitudes (+179.0) unwrap to -181.0 before scaling."""
    area_def = fixed_grid.build_area_def(
        coin_constants.SATELLITE_NAME,
        (2, 2),
        coin_constants.GOES_AREA_EXTENT_METERS,
    )
    lats = np.zeros((2, 2), dtype=np.float64)
    lngs = np.array([[-179.0, -75.0], [0.0, 179.0]], dtype=np.float64)
    geom = fixed_grid.GridGeometry(
        lats=lats,
        lngs=lngs,
        in_range_mask=np.ones((2, 2), dtype=bool),
        n_in_range=4,
        area_def=area_def,
    )
    channels = {
        ch: np.full((2, 2), 20.0, dtype=np.float32)
        for ch in coin_constants.GOES_CHANNELS
    }
    dt = datetime.datetime(2020, 1, 1, 17, 0, tzinfo=datetime.timezone.utc)

    expected_unwrapped = np.array(
        [-179.0, -75.0, 0.0, -181.0], dtype=np.float32
    )
    sat_name = coin_constants.SATELLITE_NAME
    cfg = _make_valid_train_config(mode)
    prep = coin_features.build_feature_matrix(cfg, channels, geom, dt)
    lng_idx = coin_constants.SORTED_INPUT_NAMES_BY_MODE[mode].index(
        f'{sat_name}_lngs'
    )
    expected_std = coin_features.standardize_preprocess(
        expected_unwrapped,
        float(cfg[f'{sat_name}_lngs_mean']),
        float(cfg[f'{sat_name}_lngs_stddev']),
    )
    np.testing.assert_allclose(
        prep.model_input[:, lng_idx], expected_std, rtol=0.0, atol=1e-7
    )
    self.assertLess(
        abs(
            float(prep.model_input[3, lng_idx])
            - float(prep.model_input[0, lng_idx])
        ),
        0.02,
    )


if __name__ == '__main__':
  absltest.main()

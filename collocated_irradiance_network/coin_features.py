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

"""Feature pre-masking, standardization, and matrix assembly for COIN inference.

Pre-masks valid pixels (N_valid <= 21,199,940) independently per split subsystem
(longwave channels 8--16 for OLR; shortwave channels 1--7 for RSR), unwraps
positive longitudes, computes periodic (sin, cos) solar azimuth features, and
standardizes float32 inputs in lexicographical SORTED_INPUT_NAMES_BY_MODE order.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import json

import numpy as np

from collocated_irradiance_network import coin_constants
from collocated_irradiance_network import fixed_grid
from collocated_irradiance_network import solar_geometry


@dataclasses.dataclass(frozen=True)
class PreparedFeatures:
  """Pre-masked, standardized feature matrix and pixel indices."""

  model_input: np.ndarray
  row_indices: np.ndarray
  col_indices: np.ndarray
  valid_mask: np.ndarray
  n_in_range: int
  n_valid: int
  label_mode: str


@dataclasses.dataclass(frozen=True)
class SplitPreparedFeatures:
  """Pre-masked OLR and RSR feature matrices with shared solar geometry."""

  olr_prepared: PreparedFeatures
  rsr_prepared: PreparedFeatures
  solar_zenith_deg_grid: np.ndarray
  n_valid_union: int


def validate_train_config(
    raw_config,
):
  """Validates a COIN train_config dictionary against the strict schema.

  Raises ValueError on missing keys, unexpected keys, nested dicts (e.g.
  unflattened runner `args` mappings), unsupported label_mode, or non-positive
  stddevs.

  Args:
    raw_config: Mapping loaded from train_config.json.

  Returns:
    Validated copy of `raw_config` as a flat dictionary.
  """
  for k, v in raw_config.items():
    if isinstance(v, Mapping):
      raise ValueError(
          f'Nested dictionary key {k!r} is not allowed in flat train_config.'
      )

  allowed_keys = (
      set(coin_constants.REQUIRED_STANDARDIZATION_KEYS)
      | set(coin_constants.REQUIRED_ARCHITECTURE_KEYS.keys())
      | {'label_mode'}
      | set(coin_constants.ALLOWED_AUXILIARY_CONFIG_KEYS)
  )
  extra_keys = sorted(set(raw_config.keys()) - allowed_keys)
  if extra_keys:
    raise ValueError(f'Unexpected key(s) in train_config: {extra_keys}')

  raw_label_mode = raw_config['label_mode']
  if (
      not isinstance(raw_label_mode, str)
      or raw_label_mode not in coin_constants.ALLOWED_LABEL_MODES
  ):
    raise ValueError(
        'Invalid label_mode in train_config: expected one of '
        f'{sorted(coin_constants.ALLOWED_LABEL_MODES)}, '
        f'got {raw_label_mode!r}'
    )
  label_mode = raw_label_mode

  for (
      arch_key,
      expected_val,
  ) in coin_constants.REQUIRED_ARCHITECTURE_KEYS.items():
    actual_val = raw_config[arch_key]
    if isinstance(actual_val, bool) or actual_val != expected_val:
      raise ValueError(
          f'Invalid value for {arch_key!r} in train_config: expected '
          f'{expected_val!r}, got {actual_val!r}'
      )

  sat_name = coin_constants.SATELLITE_NAME
  required_std_bases = (
      [
          f'{sat_name}_channel_{ch}'
          for ch in coin_constants.CHANNELS_BY_MODE[label_mode]
      ]
      + [f'{sat_name}_lats', f'{sat_name}_lngs']
      + list(coin_constants.LABEL_NAMES_BY_MODE[label_mode])
  )
  required_std_keys = [f'{name}_mean' for name in required_std_bases] + [
      f'{name}_stddev' for name in required_std_bases
  ]
  for std_key in required_std_keys:
    if std_key not in raw_config:
      raise ValueError(
          f'Missing required standardization key {std_key!r} for '
          f'label_mode={label_mode!r} in train_config.'
      )

  for std_key in coin_constants.REQUIRED_STANDARDIZATION_KEYS:
    if std_key not in raw_config:
      continue
    val = raw_config[std_key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
      raise ValueError(
          f'Standardization key {std_key!r} must be float/int, got {val!r}'
      )
    fval = float(val)
    if not np.isfinite(fval):
      raise ValueError(
          f'Standardization key {std_key!r} must be finite, got {fval}'
      )
    if std_key.endswith('_stddev') and fval <= 0.0:
      raise ValueError(
          f'Standardization stddev key {std_key!r} must be > 0, got {fval}'
      )

  return dict(raw_config)


def load_train_config(config_path):
  """Loads and validates `train_config.json` from `config_path`."""
  with open(config_path, 'r', encoding='utf-8') as f:
    raw_config = json.load(f)
  if (
      isinstance(raw_config, Mapping)
      and 'args' in raw_config
      and isinstance(raw_config['args'], Mapping)
  ):
    flat_config = {k: v for k, v in raw_config.items() if k != 'args'}
  else:
    flat_config = raw_config
  return validate_train_config(flat_config)


def standardize_preprocess(
    values, mean_val, stddev_val
):
  """Centers, normalizes to [0, 1] over +/- 5.5 stddevs, and clips in float32.

  Args:
    values: Unprocessed input array.
    mean_val: Feature training mean.
    stddev_val: Feature training standard deviation.

  Returns:
    Standardized float32 array clipped to [0, 1].
  """
  arr = np.asarray(values, dtype=np.float32)
  mean_f32 = np.float32(mean_val)
  clip_stddev = np.float32(
      coin_constants.STANDARDIZATION_MAX_STDDEVS * float(stddev_val)
  )
  denom = np.float32(2.0) * clip_stddev
  if not np.isfinite(mean_f32):
    raise ValueError(f'mean_val must be finite in float32, got {mean_val!r}')
  if not np.isfinite(denom) or denom <= np.float32(0.0):
    raise ValueError(
        'stddev_val must yield a finite positive float32 normalization '
        f'denominator, got stddev_val={stddev_val!r}'
    )
  centered = arr - mean_f32
  standardized = (centered + clip_stddev) / denom
  return np.clip(standardized, np.float32(0.0), np.float32(1.0)).astype(
      np.float32
  )


def zero_min_normalize_minmax(
    mean_val, stddev_val
):
  """Returns `(x_min, x_max)` bounds for non-negative output normalization.

  Used at inference to denormalize `[0, 1]` MLP predictions (`predicted_olr`
  and `predicted_rsr`) back to physical W/m^2, and to encode the `0.0 W/m^2`
  nighttime RSR bypass value into the same normalized scale.

  Args:
    mean_val: Output target mean from `train_config.json` (W/m^2).
    stddev_val: Output target standard deviation from `train_config.json`
      (W/m^2).
  """
  mean_f64 = float(mean_val)
  stddev_f64 = float(stddev_val)
  if (
      not np.isfinite(mean_f64)
      or not np.isfinite(stddev_f64)
      or stddev_f64 <= 0.0
  ):
    raise ValueError(
        f'Invalid output normalization stats: mean={mean_val!r}, '
        f'stddev={stddev_val!r}'
    )
  x_min = max(
      0.0,
      mean_f64 - (coin_constants.STANDARDIZATION_MAX_STDDEVS * stddev_f64),
  )
  x_max = mean_f64 + (coin_constants.STANDARDIZATION_MAX_STDDEVS * stddev_f64)
  span_f32 = np.float32(x_max - x_min)
  if (
      not np.isfinite(np.float32(x_min))
      or not np.isfinite(np.float32(x_max))
      or not np.isfinite(span_f32)
      or span_f32 <= np.float32(0.0)
  ):
    raise ValueError(
        f'Degenerate or inverted [x_min, x_max]=[{x_min}, {x_max}] in '
        f'float32 for mean={mean_val!r}, stddev={stddev_val!r}'
    )
  return x_min, x_max


def zero_min_normalize_preprocess(
    values, mean_val, stddev_val
):
  """Clips physical irradiance to `[x_min, x_max]` and normalizes to [0, 1].

  Used at inference to encode the `0.0 W/m^2` nighttime RSR bypass constant
  into the MLP's normalized `[0, 1]` output space.

  Args:
    values: Physical flux values in W/m^2.
    mean_val: Output target mean from `train_config.json` (W/m^2).
    stddev_val: Output target standard deviation from `train_config.json`
      (W/m^2).

  Returns:
    Normalized float32 array in [0, 1].
  """
  x_min, x_max = zero_min_normalize_minmax(mean_val, stddev_val)
  arr = np.asarray(values, dtype=np.float32)
  clipped = np.clip(arr, np.float32(x_min), np.float32(x_max))
  return ((clipped - np.float32(x_min)) / np.float32(x_max - x_min)).astype(
      np.float32
  )


def zero_min_normalize_postprocess(
    values, mean_val, stddev_val
):
  """Denormalizes `[0, 1]` MLP predictions to physical W/m^2 in float32.

  Args:
    values: Normalized MLP predictions in [0, 1].
    mean_val: Output target mean from `train_config.json` (W/m^2).
    stddev_val: Output target standard deviation from `train_config.json`
      (W/m^2).

  Returns:
    Physical irradiance float32 array in W/m^2.
  """
  x_min, x_max = zero_min_normalize_minmax(mean_val, stddev_val)
  arr = np.asarray(values, dtype=np.float32)
  return ((arr * np.float32(x_max - x_min)) + np.float32(x_min)).astype(
      np.float32
  )


def cosine_degrees_preprocess(degrees):
  """Applies cosine to an input in degrees in float32.

  Args:
    degrees: Angle(s) in degrees.

  Returns:
    Cosine of `degrees` as a float32 array.
  """
  arr = np.asarray(degrees, dtype=np.float32)
  return np.cos(np.deg2rad(arr)).astype(np.float32)


def cosine_degrees_postprocess(cos_values):
  """Inverts `cosine_degrees_preprocess` to degrees in [0, 180] in float32.

  Args:
    cos_values: Cosine values in [-1, 1].

  Returns:
    Angle(s) in degrees in [0, 180] as a float32 array.
  """
  arr = np.asarray(cos_values, dtype=np.float32)
  clipped = np.clip(arr, np.float32(-1.0), np.float32(1.0))
  return np.rad2deg(np.arccos(clipped)).astype(np.float32)


def _finite_channel_mask(
    channels,
    required_channels,
    grid_shape,
):
  """Returns boolean mask where all `required_channels` match shape and are finite."""
  all_channels_finite = np.ones(grid_shape, dtype=bool)
  for ch in required_channels:
    rad = channels[ch]
    if rad.shape != grid_shape:
      raise ValueError(
          f'Channel {ch} shape {rad.shape} does not match grid shape '
          f'{grid_shape}'
      )
    all_channels_finite &= np.isfinite(rad)
  return all_channels_finite


def _solar_angles_float32(
    grid_geometry,
    mask,
    utc_timestamp,
):
  """Returns float32 (zenith degrees, azimuth radians) at pixels in `mask`."""
  lats = grid_geometry.lats[mask]
  lngs = grid_geometry.lngs[mask]
  sza_deg = solar_geometry.zenith_angle_degrees(lats, lngs, utc_timestamp)
  saz_rad = solar_geometry.azimuth_angle_radians(lats, lngs, utc_timestamp)
  # There's not a good reason for one to be in degrees and the other in radians;
  # They were just added as features at different times and now it would be
  # inconvenient to change.
  return (
      np.asarray(sza_deg, dtype=np.float32),
      np.asarray(saz_rad, dtype=np.float32),
  )


def _assemble_prepared_features_from_mask(
    validated_config,
    label_mode,
    channels,
    grid_geometry,
    valid_mask,
    sza_deg_valid,
    saz_rad_valid,
    cos_doy_scalar,
):
  """Assembles a mode-specific PreparedFeatures matrix over `valid_mask`."""
  row_indices, col_indices = np.nonzero(valid_mask)
  row_indices = row_indices.astype(np.int32)
  col_indices = col_indices.astype(np.int32)
  n_valid = int(row_indices.shape[0])

  sat_name = coin_constants.SATELLITE_NAME

  sorted_input_names = coin_constants.SORTED_INPUT_NAMES_BY_MODE[label_mode]
  model_input = np.empty((n_valid, len(sorted_input_names)), dtype=np.float32)
  for idx, feat_name in enumerate(sorted_input_names):
    if feat_name == 'cos_day_of_year':
      processed = np.full((n_valid,), cos_doy_scalar, dtype=np.float32)
    elif feat_name == f'{sat_name}_sin_solar_azimuths':
      processed = np.sin(saz_rad_valid).astype(np.float32)
    elif feat_name == f'{sat_name}_cos_solar_azimuths':
      processed = np.cos(saz_rad_valid).astype(np.float32)
    elif feat_name == f'{sat_name}_solar_zeniths':
      processed = cosine_degrees_preprocess(sza_deg_valid)
    elif feat_name in coin_constants.STANDARDIZED_INPUT_FEATURES:
      if feat_name == f'{sat_name}_lats':
        vals_to_standardize = grid_geometry.lats[valid_mask].astype(np.float32)
      elif feat_name == f'{sat_name}_lngs':
        raw_lngs = grid_geometry.lngs[valid_mask].astype(np.float32)
        # Shift eastern longitudes by -360 so the disk is continuous across
        # the antimeridian.
        vals_to_standardize = np.where(
            raw_lngs > np.float32(0.0),
            raw_lngs - np.float32(360.0),
            raw_lngs,
        )
      else:
        ch_num = int(feat_name.removeprefix(f'{sat_name}_channel_'))
        vals_to_standardize = channels[ch_num][valid_mask]
      processed = standardize_preprocess(
          vals_to_standardize,
          float(validated_config[f'{feat_name}_mean']),
          float(validated_config[f'{feat_name}_stddev']),
      )
    else:
      raise ValueError(
          f'Unhandled feature in {label_mode} input list: {feat_name}'
      )

    non_finite_feat = int(np.sum(~np.isfinite(processed)))
    if non_finite_feat > 0:
      raise ValueError(
          f'Finite-input gate failed for feature {feat_name!r}: found '
          f'{non_finite_feat} non-finite values out of {n_valid}.'
      )
    model_input[:, idx] = processed

  return PreparedFeatures(
      model_input=model_input,
      row_indices=row_indices,
      col_indices=col_indices,
      valid_mask=valid_mask,
      n_in_range=grid_geometry.n_in_range,
      n_valid=n_valid,
      label_mode=label_mode,
  )


def build_feature_matrix(
    train_config,
    channels,
    grid_geometry,
    scan_start_datetime,
):
  """Builds the pre-masked, standardized float32 feature matrix for `label_mode`.

  Validates that all required channels for `label_mode` match the grid shape,
  computes the combined in-range and finite-channel mask over
  `CHANNELS_BY_MODE[label_mode]`, evaluates solar geometry only on valid
  pixels, standardizes each feature into
  `SORTED_INPUT_NAMES_BY_MODE[label_mode]` order, and enforces a strict
  finite-input check.

  Args:
    train_config: Validated COIN training configuration dictionary.
    channels: Mapping of channel number to 2D float32 radiance arrays.
    grid_geometry: Precomputed GridGeometry for the target 2D grid.
    scan_start_datetime: Timezone-aware UTC scan start datetime.

  Returns:
    `PreparedFeatures` ready for MLP evaluation.
  """
  validated_config = validate_train_config(train_config)
  label_mode = str(validated_config['label_mode'])
  utc_timestamp = solar_geometry.validate_utc_datetime(scan_start_datetime)

  valid_mask = grid_geometry.in_range_mask & _finite_channel_mask(
      channels,
      coin_constants.CHANNELS_BY_MODE[label_mode],
      grid_geometry.lats.shape,
  )
  if not np.any(valid_mask):
    raise ValueError(
        f'Zero valid pixels remain for label_mode={label_mode!r} after '
        'intersecting in_range_mask and finite-channel mask.'
    )
  sza_deg, saz_rad = _solar_angles_float32(
      grid_geometry, valid_mask, utc_timestamp
  )
  return _assemble_prepared_features_from_mask(
      validated_config,
      label_mode,
      channels,
      grid_geometry,
      valid_mask,
      sza_deg,
      saz_rad,
      float(solar_geometry.cos_day_of_year(utc_timestamp)),
  )


def build_split_feature_matrices(
    olr_train_config,
    rsr_train_config,
    channels,
    grid_geometry,
    scan_start_datetime,
):
  """Builds separate OLR (15-col) and RSR (13-col) feature matrices in SPLIT mode.

  Computes `olr_valid_mask` over longwave channels 8..16 and `rsr_valid_mask`
  over shortwave channels 1..7, evaluates solar geometry once on the union of
  valid pixels (`olr_valid_mask | rsr_valid_mask`), and populates both
  subsystem feature matrices.

  Args:
    olr_train_config: Validated COIN training configuration with
      label_mode=='OLR'.
    rsr_train_config: Validated COIN training configuration with
      label_mode=='RSR'.
    channels: Mapping of channel number (1..16) to 2D float32 radiance arrays.
    grid_geometry: Precomputed GridGeometry for the target 2D grid.
    scan_start_datetime: Timezone-aware UTC scan start datetime.

  Returns:
    `SplitPreparedFeatures` containing OLR and RSR `PreparedFeatures`.
  """
  validated_config_by_mode = {
      'OLR': validate_train_config(olr_train_config),
      'RSR': validate_train_config(rsr_train_config),
  }
  for mode, validated_config in validated_config_by_mode.items():
    if validated_config['label_mode'] != mode:
      raise ValueError(
          f'{mode.lower()}_train_config must have label_mode={mode!r}, got '
          f'{validated_config["label_mode"]!r}'
      )

  utc_timestamp = solar_geometry.validate_utc_datetime(scan_start_datetime)
  grid_shape = grid_geometry.lats.shape
  valid_mask_by_mode = {
      mode: (
          grid_geometry.in_range_mask
          & _finite_channel_mask(
              channels, coin_constants.CHANNELS_BY_MODE[mode], grid_shape
          )
      )
      for mode in ('OLR', 'RSR')
  }
  for mode, mask in valid_mask_by_mode.items():
    if not np.any(mask):
      raise ValueError(
          f'Zero valid pixels remain for label_mode={mode!r} after '
          'intersecting in_range_mask and finite-channel mask.'
      )

  union_mask = valid_mask_by_mode['OLR'] | valid_mask_by_mode['RSR']
  sza_union, saz_union = _solar_angles_float32(
      grid_geometry, union_mask, utc_timestamp
  )
  solar_zenith_deg_grid = np.full(grid_shape, np.nan, dtype=np.float32)
  solar_zenith_deg_grid[union_mask] = sza_union
  solar_azimuth_rad_grid = np.full(grid_shape, np.nan, dtype=np.float32)
  solar_azimuth_rad_grid[union_mask] = saz_union
  cos_doy_scalar = float(solar_geometry.cos_day_of_year(utc_timestamp))

  prepared_by_mode = {
      mode: _assemble_prepared_features_from_mask(
          validated_config_by_mode[mode],
          mode,
          channels,
          grid_geometry,
          valid_mask,
          solar_zenith_deg_grid[valid_mask],
          solar_azimuth_rad_grid[valid_mask],
          cos_doy_scalar,
      )
      for mode, valid_mask in valid_mask_by_mode.items()
  }
  return SplitPreparedFeatures(
      olr_prepared=prepared_by_mode['OLR'],
      rsr_prepared=prepared_by_mode['RSR'],
      solar_zenith_deg_grid=solar_zenith_deg_grid,
      n_valid_union=int(np.sum(union_mask)),
  )

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

"""Explicit 5-layer split COIN MLP (TensorFlow) and night RSR bypass.

Loads portable `.npz` weight bundles (with SHA-256 verification) for split
OLR/RSR bundles (`olr/` and `rsr/`), executes the 5-layer MLP with explicit
matrix multiplications and LeakyReLU(alpha=0.2) in a compiled `@tf.function`
and applies the nighttime shortwave bypass (where solar zenith angle > 90 deg
forces RSR output to 0 instead of allowing physically implausible positive RSR
values at night.)
"""

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import os

import numpy as np
import tensorflow as tf

from collocated_irradiance_network import coin_constants
from collocated_irradiance_network import coin_features


@dataclasses.dataclass(frozen=True)
class CoinWeights:
  """Validated 5-layer MLP weights and hash checksum."""

  kernels: tuple[np.ndarray, Ellipsis]
  biases: tuple[np.ndarray, Ellipsis]
  weights_sha256: str
  label_mode: str

  def __post_init__(self):
    validate_weights(self.kernels, self.biases, self.label_mode)


@dataclasses.dataclass(frozen=True)
class GridPredictions:
  """2D full-disk (5424, 5424) predictions and masks."""

  predicted_rsr: np.ndarray
  predicted_olr: np.ndarray
  solar_zenith_deg: np.ndarray
  olr_inference_mask: np.ndarray
  rsr_inference_mask: np.ndarray
  inference_mask: np.ndarray
  n_in_range: int
  n_valid_olr: int
  n_valid_rsr: int
  n_valid: int


@dataclasses.dataclass(frozen=True)
class SplitModelBundle:
  """Loaded SPLIT (OLR + RSR) model bundle."""

  olr_weights: CoinWeights
  olr_config: dict[str, object]
  rsr_weights: CoinWeights
  rsr_config: dict[str, object]
  manifest: dict[str, object]


def file_sha256(filepath):
  """Returns the hex SHA-256 digest of a file on disk."""
  hasher = hashlib.sha256()
  with open(filepath, 'rb') as f:
    while chunk := f.read(1024 * 1024):
      hasher.update(chunk)
  return hasher.hexdigest()


def tensors_sha256(
    kernels, biases
):
  """Returns deterministic hex SHA-256 over little-endian float32 weight bytes.

  This additional hash checksum allows for verifying the weights values,
  independently of the serialization file format which file_sha256() checks.

  Args:
    kernels: Sequence of kernel tensors.
    biases: Sequence of bias tensors.

  Returns:
    Hex SHA-256 digest over the concatenated little-endian float32 bytes of the
    input tensors.
  """
  hasher = hashlib.sha256()
  for k, b in zip(kernels, biases, strict=True):
    hasher.update(np.asarray(k, dtype='<f4').tobytes())
    hasher.update(np.asarray(b, dtype='<f4').tobytes())
  return hasher.hexdigest()


def validate_weights(
    kernels,
    biases,
    label_mode,
):
  """Validates shapes, dtypes, and finiteness of the 5 MLP layers."""
  expected_k_shapes = coin_constants.EXPECTED_KERNEL_SHAPES_BY_MODE[label_mode]
  expected_b_shapes = coin_constants.EXPECTED_BIAS_SHAPES_BY_MODE[label_mode]

  validated_kernels: list[np.ndarray] = []
  validated_biases: list[np.ndarray] = []
  for idx, (raw_k, raw_b, exp_k_shape, exp_b_shape) in enumerate(
      zip(kernels, biases, expected_k_shapes, expected_b_shapes, strict=True)
  ):
    k = np.asarray(raw_k)
    b = np.asarray(raw_b)
    if k.dtype != np.float32 or b.dtype != np.float32:
      raise ValueError(
          f'Layer {idx} weights must be float32, got {k.dtype} and {b.dtype}'
      )
    if k.shape != exp_k_shape:
      raise ValueError(
          f'Layer {idx} kernel shape mismatch for label_mode={label_mode!r}: '
          f'expected {exp_k_shape}, got {k.shape}'
      )
    if b.shape != exp_b_shape:
      raise ValueError(
          f'Layer {idx} bias shape mismatch for label_mode={label_mode!r}: '
          f'expected {exp_b_shape}, got {b.shape}'
      )
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(b)):
      raise ValueError(f'Layer {idx} weights contain non-finite values.')
    validated_kernels.append(k)
    validated_biases.append(b)

  return tuple(validated_kernels), tuple(validated_biases)


def load_weights_from_npz(
    npz_path,
    expected_sha256,
    label_mode,
):
  """Loads and validates COIN MLP weights from a `.npz` archive."""
  actual_sha256 = file_sha256(npz_path)
  if expected_sha256 is not None and actual_sha256 != expected_sha256:
    raise ValueError(
        f'SHA-256 mismatch for {npz_path}: expected {expected_sha256}, '
        f'got {actual_sha256}'
    )

  n_layers = len(coin_constants.EXPECTED_KERNEL_SHAPES_BY_MODE[label_mode])
  expected_keys = {f'kernel_{idx}' for idx in range(n_layers)} | {
      f'bias_{idx}' for idx in range(n_layers)
  }
  with np.load(npz_path, allow_pickle=False) as archive:
    if set(archive.files) != expected_keys:
      raise ValueError(
          f'Unexpected or missing keys in {npz_path}: expected '
          f'{sorted(expected_keys)}, found {sorted(archive.files)}'
      )
    kernels = [np.asarray(archive[f'kernel_{idx}']) for idx in range(n_layers)]
    biases = [np.asarray(archive[f'bias_{idx}']) for idx in range(n_layers)]

  val_k, val_b = validate_weights(kernels, biases, label_mode)
  return CoinWeights(
      kernels=val_k,
      biases=val_b,
      weights_sha256=actual_sha256,
      label_mode=label_mode,
  )


def load_model_bundle(
    model_dir,
):
  """Loads an atomic (weights, train_config, manifest) `.npz` bundle.

  Expects `coin_weights.npz`, `train_config.json`, and an optional
  `MODEL_MANIFEST.json` inside `model_dir`.

  Args:
    model_dir: Directory containing `coin_weights.npz` and `train_config.json`.

  Returns:
    Tuple of `(weights, train_config, manifest)`.
  """
  config_path = os.path.join(model_dir, 'train_config.json')
  train_config = coin_features.load_train_config(config_path)
  label_mode = str(train_config['label_mode'])
  config_sha256 = file_sha256(config_path)

  manifest_path = os.path.join(model_dir, 'MODEL_MANIFEST.json')
  if os.path.isfile(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
      loaded_manifest = json.load(f)
    if loaded_manifest['label_mode'] != label_mode:
      raise ValueError(
          f'label_mode mismatch in {model_dir}: manifest has '
          f'{loaded_manifest["label_mode"]!r}, train_config has {label_mode!r}'
      )
    if loaded_manifest['train_config_sha256'] != config_sha256:
      raise ValueError(
          'train_config.json SHA-256 mismatch against manifest in '
          f'{model_dir}: expected {loaded_manifest["train_config_sha256"]}, '
          f'got {config_sha256}'
      )
    model_version = str(loaded_manifest['model_version'])
    expected_weights_sha256: str | None = str(loaded_manifest['weights_sha256'])
    expected_tensors_sha256: str | None = str(loaded_manifest['tensors_sha256'])
    disk_manifest: dict[str, object] | None = loaded_manifest
  else:
    model_version = os.path.basename(os.path.normpath(model_dir))
    expected_weights_sha256 = None
    expected_tensors_sha256 = None
    disk_manifest = None

  npz_path = os.path.join(model_dir, 'coin_weights.npz')
  weights = load_weights_from_npz(
      npz_path, expected_weights_sha256, label_mode
  )
  actual_tensor_sha = tensors_sha256(weights.kernels, weights.biases)
  if (
      expected_tensors_sha256 is not None
      and actual_tensor_sha != expected_tensors_sha256
  ):
    raise ValueError(
        'NPZ tensor SHA-256 mismatch: expected '
        f'{expected_tensors_sha256}, got {actual_tensor_sha}'
    )

  if disk_manifest is not None:
    manifest: dict[str, object] = disk_manifest
  else:
    manifest = {
        'bundle_mode': 'SINGLE',
        'label_mode': label_mode,
        'model_version': model_version,
        'weights_sha256': weights.weights_sha256,
        'tensors_sha256': actual_tensor_sha,
        'train_config_sha256': config_sha256,
    }
  return weights, train_config, manifest


def load_split_model_bundle(model_dir):
  """Loads and validates a `SPLIT` (`olr/` and `rsr/`) COIN model bundle.

  Expects `olr/` (`label_mode == "OLR"`) and `rsr/` (`label_mode == "RSR"`)
  subdirectories under `model_dir`, plus an optional root `MODEL_MANIFEST.json`.

  Args:
    model_dir: Root directory of the split model bundle.

  Returns:
    `SplitModelBundle` with validated OLR and RSR weights, configs, and
    manifest.
  """
  olr_w, olr_cfg, olr_manifest = load_model_bundle(
      os.path.join(model_dir, 'olr')
  )
  rsr_w, rsr_cfg, rsr_manifest = load_model_bundle(
      os.path.join(model_dir, 'rsr')
  )
  if olr_w.label_mode != 'OLR':
    raise ValueError(
        f'Expected olr/ subdirectory in {model_dir} to have '
        f'label_mode="OLR", got {olr_w.label_mode!r}'
    )
  if rsr_w.label_mode != 'RSR':
    raise ValueError(
        f'Expected rsr/ subdirectory in {model_dir} to have '
        f'label_mode="RSR", got {rsr_w.label_mode!r}'
    )

  manifest_path = os.path.join(model_dir, 'MODEL_MANIFEST.json')
  if os.path.isfile(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
      final_manifest: dict[str, object] = json.load(f)
    if final_manifest['bundle_mode'] != 'SPLIT':
      raise ValueError(
          f'Root MODEL_MANIFEST.json in {model_dir} must specify '
          f'bundle_mode="SPLIT", got {final_manifest["bundle_mode"]!r}.'
      )
    for key, sub_name, actual_sub in (
        ('olr_manifest', 'olr', olr_manifest),
        ('rsr_manifest', 'rsr', rsr_manifest),
    ):
      expected_sub = final_manifest[key]
      for field in (
          'model_version',
          'weights_sha256',
          'tensors_sha256',
          'train_config_sha256',
      ):
        if actual_sub[field] != expected_sub[field]:
          raise ValueError(
              f'SPLIT manifest mismatch for {sub_name}/{field} in '
              f'{model_dir}: expected {expected_sub[field]!r}, '
              f'got {actual_sub[field]!r}'
          )
  else:
    final_manifest = {
        'bundle_mode': 'SPLIT',
        'model_version': os.path.basename(os.path.normpath(model_dir)),
        'olr_manifest': olr_manifest,
        'rsr_manifest': rsr_manifest,
    }

  return SplitModelBundle(
      olr_weights=olr_w,
      olr_config=olr_cfg,
      rsr_weights=rsr_w,
      rsr_config=rsr_cfg,
      manifest=final_manifest,
  )


@tf.function(reduce_retracing=True)
def _tf_mlp_step(
    batch_x,
    kernels,
    biases,
):
  """Compiled TensorFlow graph for a single batch of the 5-layer MLP."""
  hidden = batch_x
  for kernel, bias in zip(kernels[:-1], biases[:-1], strict=True):
    hidden = tf.nn.leaky_relu(
        tf.linalg.matmul(hidden, kernel) + bias,
        alpha=coin_constants.LEAKY_RELU_ALPHA,
    )
  return tf.math.sigmoid(tf.linalg.matmul(hidden, kernels[-1]) + biases[-1])


def forward_tf(
    features,
    weights,
    batch_size,
):
  """Executes the 5-layer COIN MLP using compiled TensorFlow matmuls.

  Args:
    features: 2D float32 array of shape (N, D_in) matching `weights.label_mode`.
    weights: Validated CoinWeights.
    batch_size: Positive integer batch size.

  Returns:
    2D float32 array of shape `(N, D_out)` containing normalized predictions.
  """
  if batch_size <= 0:
    raise ValueError(f'batch_size must be positive, got {batch_size}')

  expected_n_out = len(coin_constants.LABEL_NAMES_BY_MODE[weights.label_mode])
  k_tensors = tuple(tf.constant(k, dtype=tf.float32) for k in weights.kernels)
  b_tensors = tuple(tf.constant(b, dtype=tf.float32) for b in weights.biases)

  n_samples = features.shape[0]
  outputs = np.empty((n_samples, expected_n_out), dtype=np.float32)

  for start in range(0, n_samples, batch_size):
    end = min(start + batch_size, n_samples)
    batch_tensor = tf.convert_to_tensor(features[start:end])
    outputs[start:end] = _tf_mlp_step(
        batch_tensor, k_tensors, b_tensors
    ).numpy()

  return outputs


def apply_night_rsr_bypass_and_postprocess(
    normalized_preds,
    model_input,
    train_config,
):
  """Applies nighttime shortwave bypass in normalized space and unstandardizes.

  Reconstructs solar_zenith_degrees from `model_input[:, SZA_INDEX_BY_MODE]`,
  forces normalized RSR to `standardized_zero` where
  `solar_zenith_degrees > 90.0` (strict `>`), and applies
  `zero_min_normalize_postprocess` to each active output head (`'rsr'` or
  `'olr'`).

  Args:
    normalized_preds: (N, D_out) float32 array of raw sigmoid outputs.
    model_input: (N, D_in) float32 standardized input feature matrix.
    train_config: Validated COIN training configuration dictionary.

  Returns:
    Dictionary mapping active display name (`'rsr'` or `'olr'`) to 1D float32
    physical irradiance array in W/m^2.
  """
  label_mode = str(train_config['label_mode'])
  expected_n_out = len(coin_constants.LABEL_NAMES_BY_MODE[label_mode])
  expected_n_in = len(coin_constants.SORTED_INPUT_NAMES_BY_MODE[label_mode])
  if (
      normalized_preds.shape != (model_input.shape[0], expected_n_out)
      or model_input.shape != (model_input.shape[0], expected_n_in)
  ):
    raise ValueError(
        f'Shape mismatch for label_mode={label_mode!r}: '
        f'normalized_preds={normalized_preds.shape}, '
        f'model_input={model_input.shape}'
    )
  if not np.all(np.isfinite(normalized_preds)):
    raise ValueError(
        'Non-finite normalized_preds encountered in '
        'apply_night_rsr_bypass_and_postprocess.'
    )

  result: dict[str, np.ndarray] = {}
  for col, label_name in enumerate(
      coin_constants.LABEL_NAMES_BY_MODE[label_mode]
  ):
    mean_val = float(train_config[f'{label_name}_mean'])
    stddev_val = float(train_config[f'{label_name}_stddev'])
    raw_label_norm = normalized_preds[:, col]
    if label_name == 'ceres_rsr':
      cos_sza = model_input[:, coin_constants.SZA_INDEX_BY_MODE[label_mode]]
      if not np.all(np.isfinite(cos_sza)):
        raise ValueError(
            'Non-finite solar zenith angle encountered in '
            'apply_night_rsr_bypass_and_postprocess.'
        )
      if np.any(np.abs(cos_sza) > np.float32(1.0 + 1e-5)):
        raise ValueError(
            'Out-of-range cos(solar zenith angle) outside [-1, 1] in '
            'apply_night_rsr_bypass_and_postprocess.'
        )
      sza_deg = coin_features.cosine_degrees_postprocess(cos_sza)
      is_night = sza_deg > np.float32(
          coin_constants.NIGHT_SZA_THRESHOLD_DEGREES
      )
      standardized_zero = coin_features.zero_min_normalize_preprocess(
          np.array([0.0], dtype=np.float32), mean_val, stddev_val
      )[0]
      effective_norm = np.where(is_night, standardized_zero, raw_label_norm)
    else:
      effective_norm = raw_label_norm
    result[label_name.removeprefix('ceres_')] = (
        coin_features.zero_min_normalize_postprocess(
            effective_norm, mean_val, stddev_val
        )
    )
  return result


def _scatter_to_grid(
    values,
    prepared,
    grid_shape,
):
  """Scatters per-pixel `values` onto a NaN-filled float32 grid."""
  if prepared.valid_mask.shape != grid_shape:
    raise ValueError(
        f'prepared.valid_mask shape {prepared.valid_mask.shape} does not '
        f'match grid_shape {grid_shape}'
    )
  grid = np.full(grid_shape, np.nan, dtype=np.float32)
  grid[prepared.row_indices, prepared.col_indices] = values
  return grid


def _build_grid_predictions(
    olr_prepared,
    predicted_olr_valid,
    rsr_prepared,
    predicted_rsr_valid,
    solar_zenith_deg,
    n_valid_union,
    grid_shape,
):
  """Scatters OLR and RSR predictions onto the grid with per-flux masks."""
  predicted_rsr = _scatter_to_grid(
      predicted_rsr_valid, rsr_prepared, grid_shape
  )
  predicted_olr = _scatter_to_grid(
      predicted_olr_valid, olr_prepared, grid_shape
  )
  olr_inference_mask = np.zeros(grid_shape, dtype=np.uint8)
  olr_inference_mask[olr_prepared.row_indices, olr_prepared.col_indices] = 1
  rsr_inference_mask = np.zeros(grid_shape, dtype=np.uint8)
  rsr_inference_mask[rsr_prepared.row_indices, rsr_prepared.col_indices] = 1
  inference_mask = (
      (olr_inference_mask * np.uint8(coin_constants.MASK_OLR_BIT))
      | (rsr_inference_mask * np.uint8(coin_constants.MASK_RSR_BIT))
  ).astype(np.uint8)
  return GridPredictions(
      predicted_rsr=predicted_rsr,
      predicted_olr=predicted_olr,
      solar_zenith_deg=solar_zenith_deg,
      olr_inference_mask=olr_inference_mask,
      rsr_inference_mask=rsr_inference_mask,
      inference_mask=inference_mask,
      n_in_range=olr_prepared.n_in_range,
      n_valid_olr=olr_prepared.n_valid,
      n_valid_rsr=rsr_prepared.n_valid,
      n_valid=n_valid_union,
  )


def predict_split_on_grid(
    split_prepared,
    olr_weights,
    olr_config,
    rsr_weights,
    rsr_config,
    grid_shape,
    batch_size,
):
  """Runs split OLR and RSR COIN inference and scatters predictions to the grid.

  Args:
    split_prepared: SplitPreparedFeatures with OLR (15-col) and RSR (13-col).
    olr_weights: Validated OLR CoinWeights.
    olr_config: Validated OLR training configuration dictionary.
    rsr_weights: Validated RSR CoinWeights.
    rsr_config: Validated RSR training configuration dictionary.
    grid_shape: 2D grid shape, e.g. (5424, 5424).
    batch_size: Positive integer batch size.

  Returns:
    `GridPredictions` containing 2D `float32` OLR/RSR arrays and `uint8` masks.
  """
  olr_norm = forward_tf(
      split_prepared.olr_prepared.model_input,
      olr_weights,
      batch_size,
  )
  rsr_norm = forward_tf(
      split_prepared.rsr_prepared.model_input,
      rsr_weights,
      batch_size,
  )
  pred_olr_valid = apply_night_rsr_bypass_and_postprocess(
      olr_norm, split_prepared.olr_prepared.model_input, olr_config
  )['olr']
  pred_rsr_valid = apply_night_rsr_bypass_and_postprocess(
      rsr_norm, split_prepared.rsr_prepared.model_input, rsr_config
  )['rsr']

  return _build_grid_predictions(
      split_prepared.olr_prepared,
      pred_olr_valid,
      split_prepared.rsr_prepared,
      pred_rsr_valid,
      split_prepared.solar_zenith_deg_grid,
      split_prepared.n_valid_union,
      grid_shape,
  )

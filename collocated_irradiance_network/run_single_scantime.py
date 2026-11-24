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

"""Runs COIN TOA upwelling irradiance inference on a single GOES East scantime."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import resource
import time
from typing import cast

from absl import app
from absl import flags
from absl import logging

from collocated_irradiance_network import coin_constants
from collocated_irradiance_network import coin_features
from collocated_irradiance_network import coin_model
from collocated_irradiance_network import fixed_grid
from collocated_irradiance_network import goes_abi_l1b
from collocated_irradiance_network import write_output

_SCAN_TIMEKEY = flags.DEFINE_string(
    'scan_timekey',
    None,
    'GOES East scan start timekey, e.g. s20191521700447 or s20180211630395.',
)
_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    None,
    'Directory containing a SPLIT model bundle (olr/ and rsr/ subdirectories).',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Base output path ending in .nc (NetCDF4) or .zarr (Zarr v2); writes '
    '<stem>_olr.{nc|zarr} and <stem>_rsr.{nc|zarr}.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    None,
    'Number of pixels per inference batch (e.g. 262144).',
)
_WORK_DIR = flags.DEFINE_string(
    'work_dir',
    None,
    'Local directory for staging or reading the 16 ABI L1b NetCDF files.',
)
_INPUT_SOURCE = flags.DEFINE_enum(
    'input_source',
    None,
    ['gcs', 'local'],
    'Source for ABI L1b NetCDF files ("gcs" or "local").',
)
_GCS_BUCKET = flags.DEFINE_string(
    'gcs_bucket',
    None,
    'GCS bucket name when --input_source=gcs '
    '(e.g. gcp-public-data-goes-16 or gcp-public-data-goes-19).',
)
_MAX_DECODE_WORKERS = flags.DEFINE_integer(
    'max_decode_workers',
    None,
    'Maximum parallel threads for decoding and downscaling NetCDF bands'
    ' (1..16).',
)


@dataclasses.dataclass(frozen=True)
class InferenceRunSummary:
  """Summary metrics and provenance from a single-scantime inference run."""

  olr_output_path: str
  rsr_output_path: str
  scan_timekey: str
  timeline_id: str
  bundle_mode: str
  olr_model_version: str
  rsr_model_version: str
  olr_weights_sha256: str
  rsr_weights_sha256: str
  n_in_range: int
  n_valid_olr: int
  n_valid_rsr: int
  n_valid_union: int
  peak_rss_kb: int
  wall_seconds_total: float


def run_single_scantime_inference(
    scan_timekey,
    model_dir,
    output_path,
    batch_size,
    work_dir,
    scan_source,
    target_shape,
    enforce_native_shape,
    max_decode_workers,
):
  """Executes end-to-end COIN inference for a single GOES East scantime.

  Args:
    scan_timekey: GOES East scan start timekey (e.g. 's20191521700447').
    model_dir: Directory containing a SPLIT (`olr/` and `rsr/`) model bundle.
    output_path: Base destination `.nc` or `.zarr` path (writes `<stem>_olr` and
      `<stem>_rsr`).
    batch_size: Positive integer batch size for MLP execution.
    work_dir: Local staging directory for downloaded `.nc` files.
    scan_source: Implementation of `ScanFileSource` for staging `.nc` files.
    target_shape: Target 2D grid shape, e.g. (5424, 5424).
    enforce_native_shape: If True, requires raw ABI `Rad` arrays to match
      full-disk native per-channel resolutions before downscaling.
    max_decode_workers: Bounded parallel workers for NetCDF decode (1..16).

  Returns:
    `InferenceRunSummary` with output file paths, pixel counts, and timings.
  """
  t0 = time.monotonic()

  # Validate arguments, output path stem, and model bundle before network I/O.
  if batch_size <= 0:
    raise ValueError(f'batch_size must be positive, got {batch_size!r}')
  if max_decode_workers < 1 or max_decode_workers > 16:
    raise ValueError(
        f'max_decode_workers must be in [1, 16], got {max_decode_workers!r}'
    )
  _ = goes_abi_l1b.datetime_from_timekey(scan_timekey)
  write_output.derive_flux_output_paths(output_path)
  bundle = coin_model.load_split_model_bundle(model_dir)
  logging.info('Loaded SPLIT COIN model bundle from %s', model_dir)

  t_stage = time.monotonic()
  channel_files = scan_source.fetch_scan_files(scan_timekey, work_dir)
  logging.info(
      'Staged 16 ABI L1b files for %s in %.2f s',
      scan_timekey,
      time.monotonic() - t_stage,
  )

  t_decode = time.monotonic()
  scantime_rad = goes_abi_l1b.load_scantime_radiances(
      channel_files,
      target_shape,
      enforce_native_shape,
      max_decode_workers,
  )
  logging.info(
      'Decoded and downscaled 16 channels (%s) in %.2f s',
      scantime_rad.timeline_id,
      time.monotonic() - t_decode,
  )

  t_geo = time.monotonic()
  area_extent = fixed_grid.area_extent_from_image_bounds(
      scantime_rad.x_image_bounds,
      scantime_rad.y_image_bounds,
      scantime_rad.perspective_point_height,
  )
  grid_geo = fixed_grid.build_goes_east_grid(target_shape, area_extent)
  logging.info(
      'Resolved grid geometry (n_in_range=%d) in %.2f s',
      grid_geo.n_in_range,
      time.monotonic() - t_geo,
  )

  t_feat = time.monotonic()
  split_prepared = coin_features.build_split_feature_matrices(
      bundle.olr_config,
      bundle.rsr_config,
      scantime_rad.channels,
      grid_geo,
      scantime_rad.scan_start_datetime,
  )
  logging.info(
      'Assembled SPLIT feature matrices (n_valid_olr=%d, n_valid_rsr=%d, '
      'n_valid_union=%d) in %.2f s',
      split_prepared.olr_prepared.n_valid,
      split_prepared.rsr_prepared.n_valid,
      split_prepared.n_valid_union,
      time.monotonic() - t_feat,
  )

  t_infer = time.monotonic()
  predictions = coin_model.predict_split_on_grid(
      split_prepared,
      bundle.olr_weights,
      bundle.olr_config,
      bundle.rsr_weights,
      bundle.rsr_config,
      target_shape,
      batch_size,
  )
  logging.info(
      'Completed TensorFlow SPLIT inference in %.2f s',
      time.monotonic() - t_infer,
  )
  olr_manifest = cast(Mapping[str, object], bundle.manifest['olr_manifest'])
  rsr_manifest = cast(Mapping[str, object], bundle.manifest['rsr_manifest'])
  olr_model_version = str(olr_manifest['model_version'])
  rsr_model_version = str(rsr_manifest['model_version'])
  olr_weights_sha256 = bundle.olr_weights.weights_sha256
  rsr_weights_sha256 = bundle.rsr_weights.weights_sha256

  t_write = time.monotonic()
  olr_dataset, rsr_dataset = (
      write_output.build_flux_output_dataset(
          flux_type,
          predictions,
          area_extent,
          scantime_rad.scan_start_datetime,
          scantime_rad.timekey,
          scantime_rad.timeline_id,
          model_version,
          weights_sha256,
          'SPLIT',
      )
      for flux_type, model_version, weights_sha256 in (
          ('olr', olr_model_version, olr_weights_sha256),
          ('rsr', rsr_model_version, rsr_weights_sha256),
      )
  )
  olr_final_path, rsr_final_path = write_output.write_flux_datasets_atomic(
      olr_dataset, rsr_dataset, output_path
  )
  logging.info(
      'Wrote OLR and RSR datasets to %s and %s in %.2f s',
      olr_final_path,
      rsr_final_path,
      time.monotonic() - t_write,
  )

  peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  wall_total = time.monotonic() - t0
  logging.info(
      'Finished scantime %s in %.2f s (peak RSS=%d KB)',
      scantime_rad.timekey,
      wall_total,
      peak_rss_kb,
  )

  return InferenceRunSummary(
      olr_output_path=olr_final_path,
      rsr_output_path=rsr_final_path,
      scan_timekey=scantime_rad.timekey,
      timeline_id=scantime_rad.timeline_id,
      bundle_mode='SPLIT',
      olr_model_version=olr_model_version,
      rsr_model_version=rsr_model_version,
      olr_weights_sha256=olr_weights_sha256,
      rsr_weights_sha256=rsr_weights_sha256,
      n_in_range=predictions.n_in_range,
      n_valid_olr=predictions.n_valid_olr,
      n_valid_rsr=predictions.n_valid_rsr,
      n_valid_union=predictions.n_valid,
      peak_rss_kb=peak_rss_kb,
      wall_seconds_total=wall_total,
  )


def main(argv):
  """CLI entry point for single-scantime COIN inference."""
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected positional arguments: {argv[1:]}')
  if (
      _SCAN_TIMEKEY.value is None
      or _MODEL_DIR.value is None
      or _OUTPUT_PATH.value is None
      or _BATCH_SIZE.value is None
      or _WORK_DIR.value is None
      or _INPUT_SOURCE.value is None
      or _MAX_DECODE_WORKERS.value is None
  ):
    raise app.UsageError(
        '--scan_timekey, --model_dir, --output_path, --batch_size, '
        '--work_dir, --input_source, and --max_decode_workers are required.'
    )

  if _INPUT_SOURCE.value == 'gcs':
    if not _GCS_BUCKET.value:
      raise app.UsageError('--gcs_bucket is required when --input_source=gcs.')
    scan_source: goes_abi_l1b.ScanFileSource = (
        goes_abi_l1b.GcsAnonymousScanSource(
            _GCS_BUCKET.value, _MAX_DECODE_WORKERS.value
        )
    )
  elif _INPUT_SOURCE.value == 'local':
    if _GCS_BUCKET.value is not None:
      raise app.UsageError(
          '--gcs_bucket must not be set when --input_source=local.'
      )
    scan_source = goes_abi_l1b.LocalDirectoryScanSource(_WORK_DIR.value)
  else:
    raise app.UsageError(f'Unsupported --input_source={_INPUT_SOURCE.value!r}')

  summary = run_single_scantime_inference(
      _SCAN_TIMEKEY.value,
      _MODEL_DIR.value,
      _OUTPUT_PATH.value,
      _BATCH_SIZE.value,
      _WORK_DIR.value,
      scan_source,
      coin_constants.FULL_DISK_2KM_SHAPE,
      True,
      _MAX_DECODE_WORKERS.value,
  )
  logging.info(
      'InferenceRunSummary: %s',
      json.dumps(dataclasses.asdict(summary), sort_keys=True),
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'scan_timekey',
      'model_dir',
      'output_path',
      'batch_size',
      'work_dir',
      'input_source',
      'max_decode_workers',
  ])
  app.run(main)

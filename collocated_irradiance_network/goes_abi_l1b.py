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

"""GOES ABI L1b RadF staging, admission checks, decoding, and downscaling.

Discovers and stages the 16 ABI-L1b-RadF NetCDF4 bands from public Google Cloud
Storage (e.g., gs://gcp-public-data-goes-16 or gs://gcp-public-data-goes-19) or
a local directory, enforces GOES East platform, projection, and physical
radiance range admission checks, and performs NaN-propagating float64 block-mean
downscaling onto the 5424 x 5424 2 km grid.
"""

import base64
from collections.abc import Sequence
import concurrent.futures
import contextlib
import dataclasses
import datetime
import os
import re
import threading
from typing import Protocol
import uuid

from absl import logging
from google.cloud import storage
import netCDF4  # pylint: disable=unused-import
import numpy as np
import xarray

from collocated_irradiance_network import coin_constants

_GOES_RADF_FILENAME_RE = re.compile(
    r'^OR_ABI-L1b-RadF-M(\d+)C(\d{2})_([A-Z0-9]+)_(s\d{14})_(e\d{14})_(c\d{14})\.nc$'
)
_TIMEKEY_RE = re.compile(r'^s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)$')

_FORBIDDEN_RAD_UNITS = frozenset({'k', 'kelvin', 'degk', '1', '%', 'percent'})


@dataclasses.dataclass(frozen=True)
class GoesFileMetadata:
  """Parsed metadata from a GOES ABI L1b RadF filename."""

  filename: str
  scan_mode: int
  channel: int
  platform_id: str
  timekey: str
  end_timekey: str
  creation_timekey: str
  scan_start_datetime: datetime.datetime


@dataclasses.dataclass(frozen=True)
class ScantimeRadiances:
  """Downscaled 2 km radiances and metadata for all 16 GOES ABI channels."""

  channels: dict[int, np.ndarray]
  scan_start_datetime: datetime.datetime
  timekey: str
  scan_mode: int
  timeline_id: str
  x_image_bounds: np.ndarray
  y_image_bounds: np.ndarray
  perspective_point_height: float


class ScanFileSource(Protocol):
  """Protocol for staging the 16 ABI L1b RadF NetCDF files for a scantime."""

  def fetch_scan_files(
      self, scan_timekey, work_dir
  ):
    """Returns mapping of channel (1..16) to local NetCDF file path."""
    Ellipsis


def datetime_from_timekey(timekey):
  """Parses a GOES timekey (e.g. 's20180211630395') into a UTC datetime.

  Args:
    timekey: 15-character string starting with 's' followed by YYYYDDDHHMMSST.

  Returns:
    Timezone-aware UTC `datetime.datetime`.
  """
  match = _TIMEKEY_RE.match(timekey)
  if match is None:
    raise ValueError(
        f'Invalid GOES timekey format (expected sYYYYDDDHHMMSST): {timekey!r}'
    )
  year = int(match.group(1))
  day_of_year = int(match.group(2))
  hour = int(match.group(3))
  minute = int(match.group(4))
  second = int(match.group(5))
  tenth_second = int(match.group(6))

  if day_of_year < 1 or day_of_year > 366:
    raise ValueError(f'day_of_year out of range [1, 366]: {day_of_year}')
  if hour > 23 or minute > 59 or second > 59:
    raise ValueError(
        f'Invalid time components in timekey {timekey!r}: '
        f'hour={hour}, minute={minute}, second={second}'
    )

  base_dt = datetime.datetime(year - 1, 12, 31, tzinfo=datetime.timezone.utc)
  result_dt = base_dt + datetime.timedelta(
      days=day_of_year,
      hours=hour,
      minutes=minute,
      seconds=second,
      milliseconds=tenth_second * 100,
  )
  if result_dt.year != year:
    raise ValueError(
        f'day_of_year={day_of_year} is invalid for non-leap year {year} '
        f'in timekey {timekey!r}'
    )
  return result_dt


def gcs_hour_prefix_from_timekey(timekey):
  """Returns the ABI-L1b-RadF/{YYYY}/{DDD}/{HH}/ prefix for `timekey`."""
  dt = datetime_from_timekey(timekey)
  day_of_year = dt.timetuple().tm_yday
  return f'ABI-L1b-RadF/{dt.year:04d}/{day_of_year:03d}/{dt.hour:02d}/'


def parse_goes_filename(filename):
  """Parses and validates a GOES East ABI L1b RadF filename.

  Args:
    filename: Basename of the NetCDF file.

  Returns:
    Parsed `GoesFileMetadata` instance.
  """
  basename = os.path.basename(filename)
  match = _GOES_RADF_FILENAME_RE.match(basename)
  if match is None:
    raise ValueError(
        f'Filename does not match GOES ABI-L1b-RadF pattern: {basename!r}'
    )
  scan_mode = int(match.group(1))
  channel = int(match.group(2))
  platform_id = match.group(3)
  timekey = match.group(4)
  end_timekey = match.group(5)
  creation_timekey = match.group(6)

  if platform_id not in coin_constants.ALLOWED_PLATFORM_IDS:
    raise ValueError(
        f'Unsupported platform_ID in filename {basename!r}: expected one of '
        f'{sorted(coin_constants.ALLOWED_PLATFORM_IDS)}, got {platform_id!r}'
    )
  if scan_mode not in coin_constants.ALLOWED_SCAN_MODES:
    raise ValueError(
        f'Unsupported timeline_id / scan mode M{scan_mode} in filename'
        f' {basename!r}; allowed modes:'
        f' {sorted(coin_constants.ALLOWED_SCAN_MODES)}'
    )
  if channel not in coin_constants.GOES_CHANNELS:
    raise ValueError(
        f'Invalid ABI channel {channel} in filename {basename!r}; '
        'expected 1..16.'
    )

  scan_start_dt = datetime_from_timekey(timekey)
  return GoesFileMetadata(
      filename=basename,
      scan_mode=scan_mode,
      channel=channel,
      platform_id=platform_id,
      timekey=timekey,
      end_timekey=end_timekey,
      creation_timekey=creation_timekey,
      scan_start_datetime=scan_start_dt,
  )


def select_scantime_files(
    file_paths, target_timekey
):
  """Selects exactly 16 channel files for `target_timekey`.

  If duplicate reprocessed files exist for the same channel and `s` timekey,
  deterministically selects the file with the latest `c` creation timestamp.

  Args:
    file_paths: Candidate file paths or blob names.
    target_timekey: Exact scan start timekey (e.g. 's20180211630395').

  Returns:
    Dictionary mapping channel number (1..16) to the selected file path.
  """
  _ = datetime_from_timekey(target_timekey)
  selected_by_channel: dict[int, tuple[GoesFileMetadata, str]] = {}
  observed_modes: set[int] = set()
  observed_platforms: set[str] = set()

  for path in file_paths:
    basename = os.path.basename(path)
    if not basename.endswith('.nc'):
      continue
    match = _GOES_RADF_FILENAME_RE.match(basename)
    if match is None or match.group(4) != target_timekey:
      continue
    meta = parse_goes_filename(basename)
    observed_modes.add(meta.scan_mode)
    observed_platforms.add(meta.platform_id)
    if meta.channel in selected_by_channel:
      prev_meta, _ = selected_by_channel[meta.channel]
      if meta.creation_timekey == prev_meta.creation_timekey:
        raise ValueError(
            f'Duplicate files for channel {meta.channel} and timekey '
            f'{target_timekey} with identical creation timestamp '
            f'{meta.creation_timekey}: {basename!r}'
        )
      if meta.creation_timekey > prev_meta.creation_timekey:
        logging.info(
            'Selecting newer reprocessed file %s over %s for channel %d',
            basename,
            prev_meta.filename,
            meta.channel,
        )
        selected_by_channel[meta.channel] = (meta, path)
    else:
      selected_by_channel[meta.channel] = (meta, path)

  if len(observed_modes) > 1:
    raise ValueError(
        f'Inconsistent scan modes {sorted(observed_modes)} found for '
        f'timekey {target_timekey}'
    )
  if len(observed_platforms) > 1:
    raise ValueError(
        f'Inconsistent platform IDs {sorted(observed_platforms)} found for '
        f'timekey {target_timekey}'
    )

  missing_channels = [
      ch for ch in coin_constants.GOES_CHANNELS if ch not in selected_by_channel
  ]
  if missing_channels:
    raise ValueError(
        f'Incomplete scan for timekey {target_timekey}: missing channels '
        f'{missing_channels} (found {sorted(selected_by_channel.keys())})'
    )

  return {ch: pair[1] for ch, pair in sorted(selected_by_channel.items())}


def downscale_block_mean_to_target(
    radiance, target_shape
):
  """Downscales 2D radiance array via NaN-propagating block mean in float64.

  Numerically matches `skimage.transform.downscale_local_mean` used during COIN
  training by averaging over float64 spatial blocks and casting the final result
  once to float32. Reimplemented here to avoid the skimage dependency.

  Args:
    radiance: 2D float32 numpy array.
    target_shape: Target (rows, cols) shape.

  Returns:
    Downscaled 2D float32 numpy array of shape `target_shape`.
  """
  target_rows, target_cols = target_shape
  curr_rows, curr_cols = radiance.shape
  if curr_rows == target_rows and curr_cols == target_cols:
    return np.asarray(radiance, dtype=np.float32)

  if curr_rows % target_rows != 0 or curr_cols % target_cols != 0:
    raise ValueError(
        f'Array shape {radiance.shape} is not evenly divisible by '
        f'target_shape {target_shape}'
    )
  factor_r = curr_rows // target_rows
  factor_c = curr_cols // target_cols
  if factor_r != factor_c or factor_r not in (2, 4):
    raise ValueError(
        'Downscale factor must be isotropic and in {2, 4}, got '
        f'({factor_r}, {factor_c}) for shape {radiance.shape} -> {target_shape}'
    )

  reshaped = radiance.reshape(target_rows, factor_r, target_cols, factor_c)
  return reshaped.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _validate_dataset_attributes(
    ds, filepath, expected_channel
):
  """Validates global and projection attributes on an open ABI L1b Dataset."""
  platform_id = str(ds.attrs['platform_ID'])
  if platform_id not in coin_constants.ALLOWED_PLATFORM_IDS:
    raise ValueError(
        f'Invalid platform_ID in {filepath}: expected one of '
        f'{sorted(coin_constants.ALLOWED_PLATFORM_IDS)}, got {platform_id!r}'
    )
  fname_match = _GOES_RADF_FILENAME_RE.match(os.path.basename(filepath))
  if fname_match is not None and platform_id != fname_match.group(3):
    raise ValueError(
        f'platform_ID attribute {platform_id!r} disagrees with filename '
        f'platform {fname_match.group(3)!r} in {filepath}'
    )

  scene_id = str(ds.attrs['scene_id'])
  if scene_id != coin_constants.REQUIRED_SCENE_ID:
    raise ValueError(
        f'Invalid scene_id in {filepath}: expected '
        f'{coin_constants.REQUIRED_SCENE_ID!r}, got {scene_id!r}'
    )

  timeline_id = str(ds.attrs['timeline_id'])
  if timeline_id not in coin_constants.ALLOWED_TIMELINE_IDS:
    raise ValueError(
        f'Invalid timeline_id in {filepath}: expected one of '
        f'{sorted(coin_constants.ALLOWED_TIMELINE_IDS)}, got {timeline_id!r}'
    )

  proj_attrs = ds[coin_constants.GOES_PROJECTION_KEY].attrs
  for attr_name, expected, tol in (
      (
          'longitude_of_projection_origin',
          coin_constants.SUBSATELLITE_LNG_DEGREES,
          1e-3,
      ),
      (
          'latitude_of_projection_origin',
          coin_constants.SUBSATELLITE_LAT_DEGREES,
          1e-3,
      ),
      (
          'perspective_point_height',
          coin_constants.GOES_PERSPECTIVE_POINT_HEIGHT,
          1.0,
      ),
      ('semi_major_axis', coin_constants.GOES_SEMI_MAJOR_AXIS, 1.0),
      ('semi_minor_axis', coin_constants.GOES_SEMI_MINOR_AXIS, 1.0),
  ):
    actual = float(np.asarray(proj_attrs[attr_name]).item())
    if not np.isfinite(actual) or not (abs(actual - expected) <= tol):
      raise ValueError(
          f'Invalid {attr_name} in {filepath}: expected {expected}, '
          f'got {actual}'
      )

  sweep = str(proj_attrs['sweep_angle_axis'])
  if sweep != coin_constants.GOES_SWEEP_ANGLE_AXIS:
    raise ValueError(
        f'Invalid sweep_angle_axis in {filepath}: expected '
        f'{coin_constants.GOES_SWEEP_ANGLE_AXIS!r}, got {sweep!r}'
    )
  h = float(np.asarray(proj_attrs['perspective_point_height']).item())

  x_vals = np.asarray(ds['x'].values, dtype=np.float64)
  y_vals = np.asarray(ds['y'].values, dtype=np.float64)
  if (
      x_vals.ndim != 1
      or x_vals.size < 2
      or not np.all(np.isfinite(x_vals))
      or not np.all(np.diff(x_vals) > 0.0)
  ):
    raise ValueError(f'Coordinate x is not strictly ascending in {filepath}')
  if (
      y_vals.ndim != 1
      or y_vals.size < 2
      or not np.all(np.isfinite(y_vals))
      or not np.all(np.diff(y_vals) < 0.0)
  ):
    raise ValueError(
        f'Coordinate y is not strictly descending (north-up) in {filepath}'
    )

  x_bounds = np.asarray(ds['x_image_bounds'].values, dtype=np.float64).reshape(
      -1
  )
  y_bounds = np.asarray(ds['y_image_bounds'].values, dtype=np.float64).reshape(
      -1
  )
  if (
      x_bounds.shape != (2,)
      or y_bounds.shape != (2,)
      or not np.all(np.isfinite(x_bounds))
      or not np.all(np.isfinite(y_bounds))
      or not (x_bounds[0] < x_bounds[1])
      or not (y_bounds[0] > y_bounds[1])
  ):
    raise ValueError(
        f'Invalid image_bounds ordering in {filepath}: '
        f'x_image_bounds={x_bounds}, y_image_bounds={y_bounds}'
    )

  dx = (x_bounds[1] - x_bounds[0]) / float(x_vals.size)
  expected_x = (
      x_bounds[0] + (np.arange(x_vals.size, dtype=np.float64) + 0.5) * dx
  )
  if not np.allclose(x_vals, expected_x, rtol=0.0, atol=1e-5):
    raise ValueError(
        'Coordinate x does not match uniform cell centers of x_image_bounds '
        f'in {filepath}'
    )
  dy = (y_bounds[1] - y_bounds[0]) / float(y_vals.size)
  expected_y = (
      y_bounds[0] + (np.arange(y_vals.size, dtype=np.float64) + 0.5) * dy
  )
  if not np.allclose(y_vals, expected_y, rtol=0.0, atol=1e-5):
    raise ValueError(
        'Coordinate y does not match uniform cell centers of y_image_bounds '
        f'in {filepath}'
    )

  band_arr = np.asarray(ds['band_id'].values).reshape(-1)
  if band_arr.size != 1 or int(band_arr[0]) != expected_channel:
    raise ValueError(
        f'band_id mismatch in {filepath}: expected [{expected_channel}], '
        f'got {band_arr.tolist()}'
    )

  return timeline_id, x_bounds, y_bounds, h


def _validate_radiance_values(
    rad,
    rad_attrs,
    expected_channel,
    filepath,
):
  """Validates that decoded `Rad` contains physical radiances, not BT or refl."""
  units_str = str(rad_attrs['units']).strip().lower()
  if units_str in _FORBIDDEN_RAD_UNITS:
    raise ValueError(
        f'Forbidden Rad units {rad_attrs["units"]!r} in {filepath}; '
        'expected spectral radiance, not brightness temperature or '
        'reflectance.'
    )

  if np.any(np.isinf(rad)):
    n_inf = int(np.sum(np.isinf(rad)))
    raise ValueError(
        f'Decoded Rad array for channel {expected_channel} in {filepath} '
        f'contains {n_inf} infinite value(s).'
    )

  finite_mask = np.isfinite(rad)
  if not np.any(finite_mask):
    raise ValueError(
        f'Decoded Rad array for channel {expected_channel} in {filepath} '
        'is 100% NaN.'
    )

  finite_vals = rad[finite_mask]
  min_val = float(np.min(finite_vals))
  max_val = float(np.max(finite_vals))
  allowed_min, allowed_max = coin_constants.CHANNEL_RADIANCE_MIN_MAX[
      expected_channel
  ]
  if min_val < allowed_min or max_val > allowed_max:
    raise ValueError(
        f'Channel {expected_channel} radiance in {filepath} outside physical '
        f'bounds [{allowed_min}, {allowed_max}]: observed '
        f'min={min_val:.4f}, max={max_val:.4f}'
    )


def load_and_downscale_channel_nc(
    filepath,
    expected_channel,
    target_shape,
    enforce_native_shape,
):
  """Loads one ABI L1b NetCDF file, validates metadata, and downscales to 2 km.

  Args:
    filepath: Local filesystem path to the `.nc` file.
    expected_channel: Expected ABI channel number (1..16).
    target_shape: Target 2D grid shape, e.g. (5424, 5424).
    enforce_native_shape: If True, requires raw `Rad` shape to match
      `coin_constants.CHANNEL_NATIVE_SHAPES[expected_channel]`.

  Returns:
    Tuple of (downscaled_rad_float32, timeline_id, x_image_bounds,
    y_image_bounds, perspective_point_height).
  """
  with coin_constants.NETCDF4_LOCK:
    with xarray.open_dataset(
        filepath, engine='netcdf4', mask_and_scale=True
    ) as ds:
      timeline_id, x_bounds, y_bounds, h = _validate_dataset_attributes(
          ds, filepath, expected_channel
      )
      rad_da = ds['Rad']
      if rad_da.dims != ('y', 'x'):
        raise ValueError(
            f'Expected Rad dims ("y", "x") in {filepath}, got {rad_da.dims}'
        )
      if not np.issubdtype(rad_da.dtype, np.floating):
        raise ValueError(
            f'Expected floating Rad dtype after CF decode in {filepath}, '
            f'got {rad_da.dtype}'
        )
      rad_attrs = dict(rad_da.attrs)
      rad_raw = np.asarray(rad_da.values, dtype=np.float32)

  if enforce_native_shape:
    expected_native = coin_constants.CHANNEL_NATIVE_SHAPES[expected_channel]
    if rad_raw.shape != expected_native:
      raise ValueError(
          f'Channel {expected_channel} native shape mismatch in {filepath}: '
          f'expected {expected_native}, got {rad_raw.shape}'
      )

  _validate_radiance_values(rad_raw, rad_attrs, expected_channel, filepath)
  downscaled = downscale_block_mean_to_target(rad_raw, target_shape)
  del rad_raw
  return downscaled, timeline_id, x_bounds, y_bounds, h


def load_scantime_radiances(
    channel_filepaths,
    target_shape,
    enforce_native_shape,
    max_decode_workers,
):
  """Loads, validates, and downscales all 16 ABI channels for a single scantime.

  Uses bounded thread parallelism (`max_decode_workers`) so at most a few native
  0.5 km / 1 km arrays coexist in memory before downscaling.

  Args:
    channel_filepaths: Mapping of channel (1..16) to local `.nc` file path.
    target_shape: Target 2D grid shape, e.g. (5424, 5424).
    enforce_native_shape: If True, enforces native full-disk channel dimensions.
    max_decode_workers: Maximum parallel decode threads (1..16).

  Returns:
    `ScantimeRadiances` containing the 16 downscaled 2 km radiance arrays.

  Raises:
    ValueError: If channel files, timekeys, scan modes, or attributes mismatch.
  """
  if set(channel_filepaths.keys()) != set(coin_constants.GOES_CHANNELS):
    raise ValueError(
        'channel_filepaths must contain exactly channels 1..16, got '
        f'{sorted(channel_filepaths.keys())}'
    )

  first_meta = parse_goes_filename(channel_filepaths[1])
  for ch, path in channel_filepaths.items():
    meta = parse_goes_filename(path)
    if meta.channel != ch:
      raise ValueError(
          f'Channel key {ch} does not match filename channel {meta.channel}: '
          f'{path}'
      )
    if meta.timekey != first_meta.timekey:
      raise ValueError(
          f'Mismatched timekeys across channels: {first_meta.timekey} vs '
          f'{meta.timekey}'
      )
    if meta.scan_mode != first_meta.scan_mode:
      raise ValueError(
          f'Mismatched scan modes across channels: {first_meta.scan_mode} vs '
          f'{meta.scan_mode}'
      )
    if meta.platform_id != first_meta.platform_id:
      raise ValueError(
          f'Mismatched platform IDs across channels: '
          f'{first_meta.platform_id!r} vs {meta.platform_id!r}'
      )

  def _decode_one(
      ch,
  ):
    return ch, load_and_downscale_channel_nc(
        channel_filepaths[ch], ch, target_shape, enforce_native_shape
    )

  if max_decode_workers == 1:
    decoded_by_channel = dict(
        _decode_one(ch) for ch in coin_constants.GOES_CHANNELS
    )
  else:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_decode_workers
    ) as executor:
      futures = [
          executor.submit(_decode_one, ch)
          for ch in coin_constants.GOES_CHANNELS
      ]
      decoded_by_channel = dict(
          fut.result() for fut in concurrent.futures.as_completed(futures)
      )

  channels_out = {
      ch: decoded_by_channel[ch][0] for ch in coin_constants.GOES_CHANNELS
  }
  timeline_ids = {
      decoded_by_channel[ch][1] for ch in coin_constants.GOES_CHANNELS
  }
  if len(timeline_ids) != 1:
    raise ValueError(
        'Inconsistent timeline_id attributes across channels: '
        f'{sorted(timeline_ids)}'
    )
  timeline_id = next(iter(timeline_ids))
  expected_timeline = f'ABI Mode {first_meta.scan_mode}'
  if timeline_id != expected_timeline:
    raise ValueError(
        f'timeline_id {timeline_id!r} disagrees with filename scan mode '
        f'M{first_meta.scan_mode} (expected {expected_timeline!r})'
    )
  _, _, ref_x_bounds, ref_y_bounds, ref_h = decoded_by_channel[13]
  for ch in coin_constants.GOES_CHANNELS:
    _, _, ch_x_bounds, ch_y_bounds, ch_h = decoded_by_channel[ch]
    if not (
        np.allclose(ch_x_bounds, ref_x_bounds, rtol=0.0, atol=1e-6)
        and np.allclose(ch_y_bounds, ref_y_bounds, rtol=0.0, atol=1e-6)
    ):
      raise ValueError(
          f'Inconsistent image_bounds across channels (ch 13 vs ch {ch}): '
          f'x={ref_x_bounds} vs {ch_x_bounds}, '
          f'y={ref_y_bounds} vs {ch_y_bounds}'
      )
    if abs(ch_h - ref_h) > 1e-3:
      raise ValueError(
          'Inconsistent perspective_point_height across channels '
          f'(ch 13 vs ch {ch}): {ref_h} vs {ch_h}'
      )

  return ScantimeRadiances(
      channels=channels_out,
      scan_start_datetime=first_meta.scan_start_datetime,
      timekey=first_meta.timekey,
      scan_mode=first_meta.scan_mode,
      timeline_id=timeline_id,
      x_image_bounds=ref_x_bounds,
      y_image_bounds=ref_y_bounds,
      perspective_point_height=ref_h,
  )


class LocalDirectoryScanSource:
  """ScanFileSource that discovers ABI L1b RadF files in a local directory."""

  def __init__(self, source_dir):
    self._source_dir = source_dir

  def fetch_scan_files(
      self, scan_timekey, work_dir
  ):
    """Selects and returns the 16 channel files from `self._source_dir`."""
    del work_dir
    entries = [
        os.path.join(self._source_dir, name)
        for name in sorted(os.listdir(self._source_dir))
    ]
    return select_scantime_files(entries, scan_timekey)


def _compute_crc32c_base64(filepath):
  """Computes base64-encoded CRC32C of `filepath` using google_crc32c."""
  # Imported lazily since google-crc32c is only required for GCS downloads.
  import google_crc32c  # pylint: disable=g-import-not-at-top

  checksum = google_crc32c.Checksum()
  with open(filepath, 'rb') as f:
    while chunk := f.read(1024 * 1024):
      checksum.update(chunk)
  return base64.b64encode(checksum.digest()).decode('ascii')


class GcsAnonymousScanSource:
  """Downloads the 16 ABI L1b RadF files from the public GCS mirror."""

  def __init__(self, bucket_name, max_download_workers):
    self._bucket_name = bucket_name
    self._max_download_workers = max_download_workers

  def fetch_scan_files(
      self, scan_timekey, work_dir
  ):
    """Lists and downloads the 16 channel `.nc` files for `scan_timekey`."""
    if not os.path.isdir(work_dir):
      os.makedirs(work_dir, exist_ok=True)

    prefix = gcs_hour_prefix_from_timekey(scan_timekey)
    with contextlib.closing(storage.Client.create_anonymous_client()) as client:
      bucket = client.bucket(self._bucket_name)

      blobs_by_name: dict[str, storage.Blob] = {}
      for blob in client.list_blobs(bucket, prefix=prefix):
        basename = os.path.basename(blob.name)
        if scan_timekey in basename and basename.endswith('.nc'):
          blobs_by_name[blob.name] = blob

      if not blobs_by_name:
        raise FileNotFoundError(
            f'No ABI-L1b-RadF blobs found in gs://{self._bucket_name}/{prefix} '
            f'for scan_timekey {scan_timekey}'
        )

      selected_blob_names = select_scantime_files(
          sorted(blobs_by_name.keys()), scan_timekey
      )

      def _download_channel(ch):
        blob_name = selected_blob_names[ch]
        blob = blobs_by_name[blob_name]
        local_path = os.path.join(work_dir, os.path.basename(blob_name))
        temp_path = (
            f'{local_path}.part.{os.getpid()}.{threading.get_ident()}.'
            f'{uuid.uuid4().hex}'
        )
        try:
          blob.download_to_filename(temp_path)
          if blob.size is None:
            raise IOError(f'Missing GCS blob size metadata for {blob_name}')
          actual_size = os.path.getsize(temp_path)
          if actual_size != blob.size:
            raise IOError(
                f'Downloaded size mismatch for {blob_name}: expected '
                f'{blob.size}, got {actual_size}'
            )
          if blob.crc32c is None:
            raise IOError(f'Missing GCS blob crc32c metadata for {blob_name}')
          computed_crc = _compute_crc32c_base64(temp_path)
          if computed_crc != blob.crc32c:
            raise IOError(
                f'CRC32C mismatch for {blob_name}: expected {blob.crc32c}, '
                f'got {computed_crc}'
            )
          os.replace(temp_path, local_path)
        finally:
          if os.path.exists(temp_path):
            os.unlink(temp_path)
        return ch, local_path

      local_files: dict[int, str] = {}
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=self._max_download_workers
      ) as executor:
        futures = [
            executor.submit(_download_channel, ch)
            for ch in coin_constants.GOES_CHANNELS
        ]
        for fut in concurrent.futures.as_completed(futures):
          ch_id, path = fut.result()
          local_files[ch_id] = path

      return dict(sorted(local_files.items()))

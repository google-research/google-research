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

"""Two-file CF-1.7 xarray.Dataset assembly and atomic serialization (.nc/.zarr).

Writes separate `<stem>_olr.{nc|zarr}` and `<stem>_rsr.{nc|zarr}` NetCDF4 or
Zarr v2 outputs with GOES-R fixed-grid projection metadata, cryptographic model
provenance, solar zenith angle, and per-flux/combined valid-pixel masks.
"""

import datetime
import os
import shutil
import uuid

import netCDF4  # pylint: disable=unused-import
import numpy as np
import xarray
import zarr  # pylint: disable=unused-import

from collocated_irradiance_network import coin_constants
from collocated_irradiance_network import coin_model
from collocated_irradiance_network import solar_geometry


def derive_flux_output_paths(output_path):
  """Derives `<stem>_olr.{nc|zarr}` and `<stem>_rsr.{nc|zarr}` from `output_path`.

  Args:
    output_path: Base output path ending in `.nc` or `.zarr`.

  Returns:
    Tuple of `(olr_output_path, rsr_output_path)`.
  """
  if output_path.endswith('.nc'):
    ext = '.nc'
  elif output_path.endswith('.zarr'):
    ext = '.zarr'
  else:
    raise ValueError(
        f'Unsupported output_path extension for {output_path!r}; '
        'must end in ".nc" or ".zarr".'
    )

  stem_path = output_path[: -len(ext)]
  stem_name = os.path.basename(stem_path)
  if not stem_name:
    raise ValueError(
        f'Invalid output_path with empty filename stem: {output_path!r}'
    )
  if stem_name.endswith('_olr') or stem_name.endswith('_rsr'):
    raise ValueError(
        f'output_path stem {stem_name!r} must not already end with "_olr" or '
        '"_rsr"; pass the base stem path so both files can be derived.'
    )

  return f'{stem_path}_olr{ext}', f'{stem_path}_rsr{ext}'


def _build_grid_coords_and_projection(
    shape,
    area_extent_meters,
    scan_start_datetime,
):
  """Builds GOES-R fixed-grid coordinates, bounds, and projection DataArray."""
  ext = np.asarray(area_extent_meters, dtype=np.float64)
  if (
      ext.shape != (4,)
      or not np.all(np.isfinite(ext))
      or not (ext[0] < ext[2])
      or not (ext[1] < ext[3])
  ):
    raise ValueError(
        'Invalid area_extent_meters (expected finite x_min < x_max and '
        f'y_min < y_max): {area_extent_meters}'
    )
  rows, cols = shape
  h = coin_constants.GOES_PERSPECTIVE_POINT_HEIGHT

  x_bounds = np.array([ext[0] / h, ext[2] / h], dtype=np.float64)
  # ABI y_image_bounds is stored [north_max, south_min].
  y_bounds = np.array([ext[3] / h, ext[1] / h], dtype=np.float64)

  x_len = x_bounds[1] - x_bounds[0]
  x_pixel_size = x_len / float(cols)
  x_centers = np.linspace(
      x_bounds[0] + (x_pixel_size / 2.0),
      x_bounds[1] - (x_pixel_size / 2.0),
      num=cols,
      dtype=np.float32,
  )

  y_len = y_bounds[1] - y_bounds[0]
  y_pixel_size = y_len / float(rows)
  y_centers = np.linspace(
      y_bounds[0] + (y_pixel_size / 2.0),
      y_bounds[1] - (y_pixel_size / 2.0),
      num=rows,
      dtype=np.float32,
  )

  proj_attrs = {
      'long_name': 'GOES-R ABI fixed grid projection',
      'grid_mapping_name': 'geostationary',
      'perspective_point_height': coin_constants.GOES_PERSPECTIVE_POINT_HEIGHT,
      'semi_major_axis': coin_constants.GOES_SEMI_MAJOR_AXIS,
      'semi_minor_axis': coin_constants.GOES_SEMI_MINOR_AXIS,
      'inverse_flattening': coin_constants.GOES_INVERSE_FLATTENING,
      'latitude_of_projection_origin': coin_constants.SUBSATELLITE_LAT_DEGREES,
      'longitude_of_projection_origin': coin_constants.SUBSATELLITE_LNG_DEGREES,
      'sweep_angle_axis': coin_constants.GOES_SWEEP_ANGLE_AXIS,
  }
  goes_imager_projection = xarray.DataArray(
      data=np.array(np.iinfo(np.int32).min, dtype=np.int32),
      attrs=proj_attrs,
  )

  x_coord = xarray.DataArray(
      data=x_centers,
      dims=['x'],
      attrs={
          'units': 'rad',
          'axis': 'X',
          'long_name': 'GOES fixed grid projection x-coordinate',
          'standard_name': 'projection_x_coordinate',
      },
  )
  y_coord = xarray.DataArray(
      data=y_centers,
      dims=['y'],
      attrs={
          'units': 'rad',
          'axis': 'Y',
          'long_name': 'GOES fixed grid projection y-coordinate',
          'standard_name': 'projection_y_coordinate',
      },
  )
  t_coord = np.datetime64(scan_start_datetime.replace(tzinfo=None), 'ns')
  return (
      x_coord,
      y_coord,
      goes_imager_projection,
      x_bounds,
      y_bounds,
      t_coord,
  )


def build_flux_output_dataset(
    flux_type,
    predictions,
    area_extent_meters,
    scan_start_datetime,
    scan_timekey,
    timeline_id,
    model_version,
    weights_sha256,
    bundle_mode,
):
  """Builds a single-flux (`olr` or `rsr`) CF-1.7 compliant xarray.Dataset.

  Writes a single flux variable (`predicted_olr` or `predicted_rsr`) per dataset
  with CF-1.7 geostationary projection metadata and cryptographic provenance.

  Args:
    flux_type: Either `'olr'` or `'rsr'`.
    predictions: GridPredictions with 2D float32 RSR, OLR, SZA, and uint8 masks.
    area_extent_meters: (x_min, y_min, x_max, y_max) in geostationary meters.
    scan_start_datetime: Timezone-aware UTC scan start datetime.
    scan_timekey: GOES scan start timekey (e.g. 's20180211630395').
    timeline_id: ABI timeline string ('ABI Mode 3' or 'ABI Mode 6').
    model_version: Model version identifier for this flux subsystem.
    weights_sha256: Hex SHA-256 digest of the weights for this flux subsystem.
    bundle_mode: Bundle mode string ('SPLIT').

  Returns:
    CF-1.7 compliant `xarray.Dataset` for the requested `flux_type`.
  """
  if flux_type not in ('olr', 'rsr'):
    raise ValueError(
        f'Unsupported flux_type {flux_type!r}; expected "olr" or "rsr".'
    )

  _ = solar_geometry.validate_utc_datetime(scan_start_datetime)
  shape = predictions.predicted_olr.shape

  (
      x_coord,
      y_coord,
      goes_imager_projection,
      x_bounds,
      y_bounds,
      t_coord,
  ) = _build_grid_coords_and_projection(
      shape, area_extent_meters, scan_start_datetime
  )

  if flux_type == 'olr':
    flux_data = predictions.predicted_olr
    long_name = 'Predicted Outgoing Longwave Radiation'
    standard_name = 'toa_outgoing_longwave_flux'
    sub_mask_arr = predictions.olr_inference_mask
    n_valid_pixels = int(predictions.n_valid_olr)
  else:
    flux_data = predictions.predicted_rsr
    long_name = 'Predicted Reflected Shortwave Radiation'
    standard_name = 'toa_outgoing_shortwave_flux'
    sub_mask_arr = predictions.rsr_inference_mask
    n_valid_pixels = int(predictions.n_valid_rsr)

  flux_var_name = f'predicted_{flux_type}'
  flux_da = xarray.DataArray(
      data=flux_data,
      dims=['y', 'x'],
      attrs={
          'long_name': long_name,
          'standard_name': standard_name,
          'units': 'W m-2',
          'grid_mapping': coin_constants.GOES_PROJECTION_KEY,
      },
  )
  sza_da = xarray.DataArray(
      data=predictions.solar_zenith_deg,
      dims=['y', 'x'],
      attrs={
          'long_name': 'Calculated Solar Zenith Angle',
          'standard_name': 'solar_zenith_angle',
          'units': 'degrees',
          'grid_mapping': coin_constants.GOES_PROJECTION_KEY,
      },
  )
  mask_da = xarray.DataArray(
      data=sub_mask_arr,
      dims=['y', 'x'],
      attrs={
          'long_name': (
              f'Binary {flux_type.upper()} inference mask '
              '(1 = model evaluated, 0 = excluded)'
          ),
          'grid_mapping': coin_constants.GOES_PROJECTION_KEY,
      },
  )
  combined_bitmask_da = xarray.DataArray(
      data=predictions.inference_mask,
      dims=['y', 'x'],
      attrs={
          'long_name': (
              'Combined inference bitmask '
              '(1 = OLR evaluated, 2 = RSR evaluated, 3 = both, 0 = neither)'
          ),
          'grid_mapping': coin_constants.GOES_PROJECTION_KEY,
      },
  )

  return xarray.Dataset(
      data_vars={
          flux_var_name: flux_da,
          'solar_zenith_deg': sza_da,
          'inference_mask': mask_da,
          'combined_inference_bitmask': combined_bitmask_da,
          coin_constants.GOES_PROJECTION_KEY: goes_imager_projection,
          'x_image_bounds': (['number_of_image_bounds'], x_bounds),
          'y_image_bounds': (['number_of_image_bounds'], y_bounds),
      },
      coords={
          'y': y_coord,
          'x': x_coord,
          't': t_coord,
          'x_image': np.float32(0.5 * (x_bounds[0] + x_bounds[1])),
          'y_image': np.float32(0.5 * (y_bounds[0] + y_bounds[1])),
      },
      attrs={
          'Conventions': 'CF-1.7',
          'title': (
              'Collocated Irradiance Network (COIN) TOA Upwelling Irradiance'
          ),
          'flux_type': flux_type,
          'bundle_mode': bundle_mode,
          'model_version': model_version,
          'weights_sha256': weights_sha256,
          'time_convention': 'scan_start',
          'scan_timekey': scan_timekey,
          'timeline_id': timeline_id,
          'n_in_range_pixels': int(predictions.n_in_range),
          'n_valid_pixels': n_valid_pixels,
          'n_valid_union_pixels': int(predictions.n_valid),
      },
  )


def _remove_path(path):
  """Removes a temporary or backup file, symlink, or directory."""
  if os.path.isdir(path) and not os.path.islink(path):
    shutil.rmtree(path)
  elif os.path.lexists(path):
    os.unlink(path)


def write_dataset_atomic(dataset, output_path):
  """Serializes `dataset` atomically to `.nc` (NetCDF4) or `.zarr` (Zarr v2).

  Writes to a sibling temporary path and renames into place only after all
  data and metadata are flushed, preventing truncated outputs on crash.

  Args:
    dataset: Assembled xarray.Dataset.
    output_path: Destination filesystem path ending in `.nc` or `.zarr`.

  Returns:
    Normalized absolute `output_path`.
  """
  abs_output = os.path.abspath(output_path)
  os.makedirs(os.path.dirname(abs_output), exist_ok=True)

  unique_id = f'{os.getpid()}.{uuid.uuid4().hex}'
  temp_path = f'{abs_output}.tmp.{unique_id}'

  if abs_output.endswith('.nc'):
    try:
      with coin_constants.NETCDF4_LOCK:
        dataset.to_netcdf(temp_path, engine='netcdf4')
      os.replace(temp_path, abs_output)
    finally:
      _remove_path(temp_path)
  elif abs_output.endswith('.zarr'):
    try:
      zarr_encoding = {
          var_name: {'chunks': dataset[var_name].shape}
          for var_name in dataset.data_vars
          if dataset[var_name].ndim == 2
      }
      dataset.to_zarr(
          temp_path,
          mode='w',
          consolidated=True,
          zarr_format=2,
          encoding=zarr_encoding,
      )
      if os.path.lexists(abs_output):
        backup_path = f'{abs_output}.old.{unique_id}'
        os.replace(abs_output, backup_path)
        try:
          os.replace(temp_path, abs_output)
        except BaseException:
          if not os.path.lexists(abs_output) and os.path.lexists(backup_path):
            os.replace(backup_path, abs_output)
          raise
        _remove_path(backup_path)
      else:
        os.replace(temp_path, abs_output)
    finally:
      _remove_path(temp_path)
  else:
    raise ValueError(
        f'Unsupported output_path extension for {output_path!r}; '
        'must end in ".nc" or ".zarr".'
    )

  return abs_output


def write_flux_datasets_atomic(
    olr_dataset,
    rsr_dataset,
    output_path,
):
  """Atomically writes separate `_olr` and `_rsr` datasets for `output_path`.

  Args:
    olr_dataset: Assembled OLR xarray.Dataset.
    rsr_dataset: Assembled RSR xarray.Dataset.
    output_path: Base output path ending in `.nc` or `.zarr`.

  Returns:
    Tuple of `(final_olr_path, final_rsr_path)`.
  """
  olr_path, rsr_path = derive_flux_output_paths(output_path)
  final_olr = write_dataset_atomic(olr_dataset, olr_path)
  final_rsr = write_dataset_atomic(rsr_dataset, rsr_path)
  return final_olr, final_rsr

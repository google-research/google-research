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

# pylint: disable=logging-fstring-interpolation
# pylint: disable=g-importing-member

"""Data processing methods for reading and writing logic."""

from collections import ChainMap
import json
import os
import time

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from united_earth_surface_fluxes.data_processing import vars_config


def _calculate_solar_time_vectorized(
    unix_ns, longitude
):
  """Calculates Local Solar Time for a time array and a longitude array.

  Args:
      unix_ns (np.ndarray): Array of timestamps (can be any shape).
      longitude (np.ndarray): Array of longitudes (can be any shape, but must be
        broadcastable with unix_ns). !!! Range [0, 360] is required !!!

  Returns:
      np.ndarray: Array of LST (shape: the broadcasted shape of unix_ns and
      longitude).

  Note:
      The longitude range must be from 0 to 360 degrees.
      If your longitude is in the range [-180, 180], you can convert it using:
      longitude[longitude < 0] += 360
  """

  try:
    np.broadcast_shapes(unix_ns.shape, longitude.shape)
  except ValueError as e:
    raise ValueError(
        f'The shapes of unix_ns {unix_ns.shape} and longitude'
        f' {longitude.shape} are not broadcastable.'
    ) from e

  unix_ns_shape = unix_ns.shape
  unix_ns_flat = unix_ns.flatten()

  utc_dt = pd.to_datetime(unix_ns_flat, unit='ns', utc=True)
  doy = utc_dt.dayofyear.values

  fractional_utc_hour = (
      utc_dt.hour
      + utc_dt.minute / 60.0
      + (utc_dt.second + utc_dt.microsecond / 1e6) / 3600.0
  ).values

  # Calculate Equation of Time (EoT) in minutes (approximation)
  b_deg = (360 / 365.24) * (doy - 81)
  b_rad = np.radians(b_deg)
  eot_minutes = (
      9.87 * np.sin(2 * b_rad) - 7.53 * np.cos(b_rad) - 1.5 * np.sin(b_rad)
  )

  fractional_utc_hour_reshaped = np.reshape(fractional_utc_hour, unix_ns_shape)
  eot_minutes_reshaped = np.reshape(eot_minutes, unix_ns_shape)

  longitude_correction_minutes = 4.0 * longitude
  time_correction_minutes = longitude_correction_minutes + eot_minutes_reshaped

  lst_hours_prelim = fractional_utc_hour_reshaped + (
      time_correction_minutes / 60.0
  )

  local_solar_time = np.fmod(lst_hours_prelim, 24)
  local_solar_time[local_solar_time < 0] += 24

  return local_solar_time


def load_location_lat_lons(spatial_coordinates_path):
  """Loads mapping references for latitude and longitudes."""

  with tf.io.gfile.GFile(spatial_coordinates_path, 'r') as f:
    lat_intervals = json.load(f)

  records = []
  for lat_str, intervals in lat_intervals.items():
    lat = float(lat_str)
    for start, end in intervals:
      num_steps = int(round((end - start) / vars_config.SPATIAL_RESOLUTION)) + 1
      for i in range(num_steps):
        lon = start + i * vars_config.SPATIAL_RESOLUTION
        records.append((lat, lon))

  records_arr = np.array(records, dtype=np.float64)

  location_data_dict = {
      'all_lat_arr': xr.DataArray(records_arr[:, 0], dims=('locations',)),
      'all_lon_arr': xr.DataArray(records_arr[:, 1], dims=('locations',)),
  }
  return location_data_dict


def _load_era5(era5_zarr_path):
  """Loads ERA5 dataset from a Zarr store."""
  storage_options = (
      {'token': 'anon'} if era5_zarr_path.startswith('gs://') else None
  )
  era5 = xr.open_zarr(
      era5_zarr_path,
      chunks=None,
      consolidated=True,
      storage_options=storage_options,
  )
  era5 = era5.sel(
      time=slice(era5.attrs['valid_time_start'], era5.attrs['valid_time_stop'])
  )
  return era5


def load_era5_for_training(era5_zarr_path):
  era5 = _load_era5(era5_zarr_path)
  era5_with_req_vars = era5[vars_config.REQ_VARS_LIST]
  return era5_with_req_vars


def _calculate_sin_cos_solar_time(time_arr, lon_arr):
  """Calculates the sine and cosine of the local solar time.

  Args:
    time_arr: An array of timestamps.
    lon_arr: An array of longitudes.

  Returns:
    A tuple containing:
      - sine_cosine_solar_time: A numpy array of shape (lon, time, 2)
        containing the sine and cosine of the solar time.
      - solar_time_vars: A list of strings ['solar_time_sin', 'solar_time_cos'].
  """
  solar_time_data = np.round(
      _calculate_solar_time_vectorized(time_arr, lon_arr)
  )

  # Add small values 0.5 to avoid zeroes, just because the ML model doesn't use
  # bias weights.
  solar_time_in_angles = 2 * np.pi * ((solar_time_data + 0.5) / 24)
  sine_cosine_solar_time = np.stack(
      [np.sin(solar_time_in_angles), np.cos(solar_time_in_angles)], axis=-1
  )  # (lon, time, 2)
  solar_time_vars = [
      'solar_time_sin',
      'solar_time_cos',
  ]
  return sine_cosine_solar_time, solar_time_vars


def get_features(
    era5_with_req_vars_at_time_t,
    pad_size,
    loc_data_dict,
    verbose=True,
):
  """Processes requested coordinates to gather static and dynamic features."""
  start_time = time.perf_counter()
  try:
    era5_with_req_vars_at_time_t = era5_with_req_vars_at_time_t.sel(
        latitude=loc_data_dict['all_lat_arr'],
        longitude=loc_data_dict['all_lon_arr'],
    ).compute()
    if verbose:
      logging.info(f'ERA5 indixed: {era5_with_req_vars_at_time_t.dims}')
  except pd.errors.InvalidIndexError as e:
    date_str = (
        era5_with_req_vars_at_time_t.time.values.astype('datetime64[D]')[0]
        .item()
        .strftime('%Y-%m-%d')
    )
    logging.info(
        'An exception was caught with time'
        f' {date_str}: \n'
        f'{era5_with_req_vars_at_time_t=:}\n'
        f"{loc_data_dict['all_lat_arr']=:}\n"
        f"{loc_data_dict['all_lon_arr']=:}",
    )
    raise e

  # --- Time to keep ---
  time_kept = (
      era5_with_req_vars_at_time_t.time.values[pad_size:-pad_size]
      if pad_size > 0
      else era5_with_req_vars_at_time_t.time.values
  )
  if verbose:
    logging.info(
        f'Time kept! {time_kept.shape=:} {time_kept[[0, 1]]}',
    )

  # --- Chain ERA5 and location data ---
  all_data_chain_map = ChainMap(
      era5_with_req_vars_at_time_t,
      loc_data_dict,
  )
  # --- State data ---
  state_data = np.stack(
      [all_data_chain_map[var].values.T for var in vars_config.REQ_VARS_LIST],
      axis=-1,
  )  # (locations, pad_size+time+pad_size, vars)
  if verbose:
    logging.info(f'State data processed! {state_data.shape=:}')
  state_data = state_data[:, pad_size:-pad_size] if pad_size > 0 else state_data
  if verbose:
    logging.info(
        f'State data truncated along time dimension! {state_data.shape=:} '
        f'with time range {time_kept[[0, -1]]}',
    )
  state_is_dyn = np.array([
      vars_config.ERA5_VARS_DICT[var]['is_dynamic']
      for var in vars_config.REQ_VARS_LIST
  ]).astype('bool')
  dyn_state_data = state_data[Ellipsis, state_is_dyn]
  stat_state_data = state_data[:, -1, ~state_is_dyn]
  state_var_names = np.asarray(vars_config.REQ_VARS_LIST)
  dyn_state_var_names = state_var_names[state_is_dyn]
  stat_state_var_names = state_var_names[~state_is_dyn]

  # --- Derived data ---
  priority_vars = [
      '2m_specific_humidity',
      'instantaneous_surface_latent_heat_flux',
      'instantaneous_surface_sensible_heat_flux',
      'instantaneous_surface_net_solar_radiation',
      'instantaneous_surface_net_thermal_radiation',
      'instantaneous_surface_thermal_radiation_downwards',
  ]
  all_data_chain_map = all_data_chain_map.new_child({
      var: vars_config.DERIVED_VARS_DICT[var]['fn'](all_data_chain_map)
      for var in priority_vars
  })
  derived_data = [
      all_data_chain_map[var]
      if var in priority_vars
      else vars_config.DERIVED_VARS_DICT[var]['fn'](all_data_chain_map)
      for var in vars_config.DERIVED_VARS_LIST
  ]

  derived_data = np.stack(
      [d.values.T if hasattr(d, 'values') else d.T for d in derived_data],
      axis=-1,
  )  # (locations, pad_size+time+pad_size, vars)
  if verbose:
    logging.info(f'Derived data processed! {derived_data.shape=:}')
  derived_data = (
      derived_data[:, pad_size:-pad_size] if pad_size > 0 else derived_data
  )
  if verbose:
    logging.info(
        f'Derived data truncated along time dimension! {derived_data.shape=:}',
    )
  derived_is_dyn = np.asarray([
      vars_config.DERIVED_VARS_DICT[var]['is_dynamic']
      for var in vars_config.DERIVED_VARS_LIST
  ]).astype('bool')
  dyn_derived_data = derived_data[Ellipsis, derived_is_dyn]
  stat_derived_data = derived_data[:, -1, ~derived_is_dyn]
  derived_var_names = np.asarray(vars_config.DERIVED_VARS_LIST)
  dyn_derived_var_names = derived_var_names[derived_is_dyn]
  stat_derived_var_names = derived_var_names[~derived_is_dyn]

  dynamic_vars = np.concatenate(
      [dyn_state_data, dyn_derived_data], axis=-1
  ).astype('float32')
  dynamic_var_names = np.concatenate(
      [dyn_state_var_names, dyn_derived_var_names]
  )
  static_vars = np.concatenate(
      [stat_state_data, stat_derived_data], axis=-1
  ).astype('float32')
  static_var_names = np.concatenate(
      [stat_state_var_names, stat_derived_var_names]
  )
  if verbose:
    logging.info(f'New {dynamic_vars.shape=:}')
    logging.info(f'New {static_vars.shape=:}')

  # --- Solar data ---
  solar_time_data, solar_time_vars_names = _calculate_sin_cos_solar_time(
      time_kept,
      loc_data_dict['all_lon_arr'].values[:, np.newaxis],
  )  # (locations, time, 2)
  if verbose:
    logging.info(
        f'Solar time data processed! {solar_time_data.shape=:}',
    )
  dynamic_vars = np.concatenate([dynamic_vars, solar_time_data], axis=-1)
  dynamic_var_names = np.concatenate(
      [dynamic_var_names, solar_time_vars_names], axis=-1
  )
  if verbose:
    logging.info(
        f'Solar time data concatenated! {dynamic_vars.shape=:}',
    )

  # --- Latitude and longitude ---
  lat_lon_arr = np.stack(
      [
          loc_data_dict['all_lat_arr'].values,
          loc_data_dict['all_lon_arr'].values,
      ],
      axis=-1,
  )  # (locations, 2)
  if verbose:
    logging.info(
        f'Latitude longitude data processed! {lat_lon_arr.shape=:}',
    )

  assert dynamic_var_names.shape[0] == dynamic_vars.shape[-1]
  assert static_var_names.shape[0] == static_vars.shape[-1]

  time_str = time_kept.astype('datetime64[D]')[0].item().strftime('%Y-%m-%d')
  assert not np.any(
      np.isnan(dynamic_vars)
  ), f'NaNs in dynamic_vars: {np.where(np.isnan(dynamic_vars))} for {time_str}'
  assert not np.any(
      np.isnan(static_vars)
  ), f'NaNs in static_vars: {np.where(np.isnan(static_vars))} for {time_str}'
  logging.info(
      f'Elapsed time: {(time.perf_counter() - start_time)/60} minutes',
  )

  indices = np.arange(lat_lon_arr.shape[0]).astype(np.int64)

  if verbose:
    logging.info(f'Indices processed! {indices.shape=:}')

  return (
      dynamic_vars.astype('float32'),
      dynamic_var_names,
      static_vars.astype('float32'),
      static_var_names,
      lat_lon_arr.astype('float32'),
      indices,
      time_kept,
  )

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

"""Configuration of ERA5 and derived variables for data processing."""

from functools import partial

import numpy as np
from scipy import interpolate as sp_interpolate
import xarray as xr

SPATIAL_RESOLUTION = 0.25  # 1/4 degree grid resolution


def _calculate_instantaneous_ground_heat_flux(xr_dataset):
  """Calculates ground heat flux in Joule/m^2."""
  ghf = -(
      xr_dataset['instantaneous_surface_net_solar_radiation']
      + xr_dataset['instantaneous_surface_net_thermal_radiation']
      + xr_dataset['instantaneous_surface_sensible_heat_flux']
      + xr_dataset['instantaneous_surface_latent_heat_flux']
  )
  return ghf


def _calculate_instantaneous_surface_thermal_radiation_upwards(xr_dataset):
  """Calculates surface net solar radiation in Watts/m^2."""
  return (
      xr_dataset['instantaneous_surface_net_thermal_radiation']
      - xr_dataset['instantaneous_surface_thermal_radiation_downwards']
  )


def _calculate_specific_humidity(xr_dataset):
  """Calculates specific humidity from dew point temperature and air pressure.

  This function is vectorized and accepts single values, NumPy arrays,
  or pandas Series as input.

  Args:
      xr_dataset: The xarray dataset containing features.

  Returns:
      The specific humidity (q) in kilograms of water vapor per
      kilogram of moist air (kg/kg).
  """
  pressure_pascals = xr_dataset['surface_pressure']

  vapor_pressure = _calculate_vapor_pressure(xr_dataset)

  # --- Step 2: Calculate specific humidity (q) from vapor pressure ---
  # Epsilon is the ratio of the molar mass of water vapor to the molar
  # mass of dry air.
  epsilon = 0.622

  # This is a highly accurate approximation for typical atmospheric conditions.
  specific_humidity = (
      epsilon
      * vapor_pressure
      / (pressure_pascals - (1 - epsilon) * vapor_pressure)
  )
  return specific_humidity


def _calculate_vapor_pressure(xr_dataset):
  dewpoint_kelvin = xr_dataset['2m_dewpoint_temperature']
  dewpoint_celsius = dewpoint_kelvin - 273.15
  # This is a widely used empirical formula (a form of the Magnus equation)
  # to get vapor pressure in Pascals.
  vapor_pressure = 611.2 * np.exp(
      17.67 * dewpoint_celsius / (dewpoint_celsius + 243.5)
  )
  return vapor_pressure


def _calculate_vpd(xr_dataset):
  """Calculates Vapor Pressure Deficit (VPD) in Pascals."""
  temp_k = xr_dataset['2m_temperature']
  dewpoint_k = xr_dataset['2m_dewpoint_temperature']
  temp_c = temp_k - 273.15
  dewpoint_c = dewpoint_k - 273.15
  es = 610.8 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
  ea = 610.8 * np.exp((17.27 * dewpoint_c) / (dewpoint_c + 237.3))
  vpd = es - ea
  vpd = xr.where(vpd > 0, vpd, 0)
  xr.testing.assert_allclose(vpd, vpd.clip(min=0))
  return vpd


def _calculate_snow_cover(xr_dataset):
  """Calculates snow cover fraction from snow depth and snow density.

  SC = min(1, (r_w * snow_depth) / (0.1 * snow_density))
  where:
  - snow_density (rsn) is in kg/m^3
  - snow_depth (sd) is in meters of water equivalent (m w.e.)
  - 1000 is density of water in kg/m^3

  Args:
      xr_dataset: An xarray Dataset containing 'snow_density' and 'snow_depth'.

  Returns:
      xarray.DataArray: Snow cover fraction (0-1).
  """
  snow_depth = xr_dataset['snow_depth']
  snow_density = xr_dataset['snow_density']
  r_w = 1000.0  # Density of water kg/m^3

  # If snow depth is 0, cover is 0.
  # Formula SC=min(1, (r_w*SD) / (0.1*RSN))
  snow_cover = xr.where(
      snow_density > 0,
      (r_w * snow_depth) / (0.1 * snow_density),
      0.0,
  )
  snow_cover = snow_cover.clip(max=1.0)
  return snow_cover


def _convert_energy_to_power(xr_dataset, energy_name):
  """Converts an accumulated energy variable into power."""
  energy_signal = xr_dataset[energy_name].values.T
  t_data = (
      xr_dataset[energy_name]
      .time.values.astype('datetime64[s]')
      .astype('float')
  )
  delta_t = np.diff(t_data, axis=-1)
  delta_t = np.concatenate([delta_t[[0]], delta_t], axis=-1)
  interpolator = sp_interpolate.interp1d(
      # subtract a half hour from the thermal flux variable because the energy
      # is accumulated over the past hour
      t_data - delta_t / 2,
      energy_signal,
      kind='cubic',
      fill_value='extrapolate',
  )
  power_signal = interpolator(t_data) / delta_t

  return power_signal.T  # Return in the original shape


# --- ERA5 Variables ---

ERA5_VARS_DICT = {
    # --- Wind speed ---
    '10m_u_component_of_wind': {'is_dynamic': True},
    '10m_v_component_of_wind': {'is_dynamic': True},
    # --- Temperature ---
    '2m_temperature': {'is_dynamic': True},
    '2m_dewpoint_temperature': {'is_dynamic': True},
    'soil_temperature_level_1': {'is_dynamic': True},
    'soil_temperature_level_2': {'is_dynamic': True},
    'soil_temperature_level_3': {'is_dynamic': True},
    # --- Boundary layer height ---
    'boundary_layer_height': {'is_dynamic': True},
    # --- Vegetation ---
    'type_of_high_vegetation': {'is_dynamic': False},
    'type_of_low_vegetation': {'is_dynamic': False},
    'skin_reservoir_content': {'is_dynamic': True},
    'leaf_area_index_high_vegetation': {'is_dynamic': True},
    'leaf_area_index_low_vegetation': {'is_dynamic': True},
    'high_vegetation_cover': {'is_dynamic': True},
    'low_vegetation_cover': {'is_dynamic': True},
    # --- Pressure ---
    'surface_pressure': {'is_dynamic': True},
    # --- Soil water ---
    'soil_type': {'is_dynamic': False},
    'volumetric_soil_water_layer_1': {'is_dynamic': True},
    'volumetric_soil_water_layer_2': {'is_dynamic': True},
    'volumetric_soil_water_layer_3': {'is_dynamic': True},
    'volumetric_soil_water_layer_4': {'is_dynamic': True},
    # --- Snow data ---
    'snow_density': {'is_dynamic': True},
    'snow_depth': {'is_dynamic': True},
    # --- Energy flux ---
    'surface_latent_heat_flux': {'is_dynamic': True},
    'surface_sensible_heat_flux': {'is_dynamic': True},
    'surface_net_solar_radiation': {'is_dynamic': True},
    'surface_net_thermal_radiation': {'is_dynamic': True},
    'surface_thermal_radiation_downwards': {'is_dynamic': True},
}

REQ_VARS_LIST = list(ERA5_VARS_DICT.keys())

# --- Derived Variables ---

DERIVED_VARS_DICT = {
    '2m_specific_humidity': {
        'fn': _calculate_specific_humidity,
        'is_dynamic': True,
    },
    'vapor_pressure': {'fn': _calculate_vapor_pressure, 'is_dynamic': True},
    'vapor_pressure_deficit': {'fn': _calculate_vpd, 'is_dynamic': True},
    'instantaneous_ground_heat_flux': {
        'fn': _calculate_instantaneous_ground_heat_flux,
        'is_dynamic': True,
    },
    'instantaneous_surface_latent_heat_flux': {
        'fn': partial(
            _convert_energy_to_power, energy_name='surface_latent_heat_flux'
        ),
        'is_dynamic': True,
    },
    'instantaneous_surface_sensible_heat_flux': {
        'fn': partial(
            _convert_energy_to_power, energy_name='surface_sensible_heat_flux'
        ),
        'is_dynamic': True,
    },
    'instantaneous_surface_net_solar_radiation': {
        'fn': partial(
            _convert_energy_to_power, energy_name='surface_net_solar_radiation'
        ),
        'is_dynamic': True,
    },
    'instantaneous_surface_net_thermal_radiation': {
        'fn': partial(
            _convert_energy_to_power,
            energy_name='surface_net_thermal_radiation',
        ),
        'is_dynamic': True,
    },
    'instantaneous_surface_thermal_radiation_downwards': {
        'fn': partial(
            _convert_energy_to_power,
            energy_name='surface_thermal_radiation_downwards',
        ),
        'is_dynamic': True,
    },
    'instantaneous_surface_thermal_radiation_upwards': {
        'fn': _calculate_instantaneous_surface_thermal_radiation_upwards,
        'is_dynamic': True,
    },
    'snow_cover': {
        'fn': _calculate_snow_cover,
        'is_dynamic': True,
    },
}
DERIVED_VARS_LIST = list(DERIVED_VARS_DICT.keys())

assert set(DERIVED_VARS_LIST) & (set(ERA5_VARS_DICT.keys())) == set(), set(
    DERIVED_VARS_LIST
) & (set(ERA5_VARS_DICT.keys()))

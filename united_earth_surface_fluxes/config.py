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
"""Module for config.py."""

# --- Feature Configuration ---

NUM_EXPERTS = 6

DYNAMIC_VARS = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_pressure',
    'vapor_pressure',
    '2m_specific_humidity',
    'boundary_layer_height',
    'instantaneous_surface_net_solar_radiation',
    'instantaneous_surface_thermal_radiation_downwards',
    'soil_temperature_level_1',
    'soil_temperature_level_2',
    'soil_temperature_level_3',
    'vapor_pressure_deficit',
    'skin_reservoir_content',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'solar_time_cos',
    'solar_time_sin',
]
STATIC_VARS = []
VEG_AND_SOIL_TYPES = [
    'type_of_high_vegetation',
    'type_of_low_vegetation',
    'soil_type',
]
TARGET_VARS = [
    'instantaneous_surface_sensible_heat_flux',
    'instantaneous_surface_latent_heat_flux',
    'instantaneous_ground_heat_flux',
    'instantaneous_surface_thermal_radiation_upwards',
]

VEG_PARAMS_PATH = './data/veg_params.csv'
SOIL_PARAMS_PATH = './data/soil_params.csv'
GROUND_HEAT_PARAMS_PATH = './data/ground_heat_params.csv'
PURE_FRACTION_THRESHOLDS = [
    1.0,  # Intercepted Water
    0.9,  # Dry Low Vegetation
    1.0,  # Exposed Snow
    0.99,  # Dry High Vegetation
    0.9,  # Shaded Snow
    1.0,  # Dry Bare ground
]

SOLAR_START_HOUR = 10
SOLAR_END_HOUR = 15

# Sample weights are derived from the training set dates (2024 even-numbered
# months, day 7) based on the inverse square root frequency of each land surface
# type, normalized by the minimum weight.
SAMPLE_WEIGHTS = [2.2769601, 1.1548724, 2.6776066, 1.1804839, 3.0307646, 1.0]

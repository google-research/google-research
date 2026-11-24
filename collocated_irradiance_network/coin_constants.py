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

"""Physical, orbital, fixed-grid, and split-mode (OLR/RSR) constants for COIN."""

import math
import threading

# PyPI manylinux netCDF4/h5py wheels link a non-threadsafe C libhdf5; serialize
# concurrent NetCDF4 dataset reads and writes across threads.
NETCDF4_LOCK = threading.Lock()

# GOES ABI channel numbers (1..16), split into shortwave and longwave bands.
GOES_CHANNELS: tuple[int, Ellipsis] = tuple(range(1, 17))
GOES_SHORTWAVE_CHANNELS: tuple[int, Ellipsis] = (1, 2, 3, 4, 5, 6, 7)
GOES_LONGWAVE_CHANNELS: tuple[int, Ellipsis] = (8, 9, 10, 11, 12, 13, 14, 15, 16)

# A SPLIT bundle pairs a longwave-only OLR model with a shortwave-only RSR
# model.
ALLOWED_LABEL_MODES: frozenset[str] = frozenset({'OLR', 'RSR'})

CHANNELS_BY_MODE: dict[str, tuple[int, Ellipsis]] = {
    'OLR': GOES_LONGWAVE_CHANNELS,
    'RSR': GOES_SHORTWAVE_CHANNELS,
}

# Satellite prefix used in train_config.json and model input feature names.
SATELLITE_NAME: str = 'goes'

# Non-radiance model inputs shared by both OLR and RSR models.
AUXILIARY_INPUT_NAMES: tuple[str, Ellipsis] = (
    'cos_day_of_year',
    f'{SATELLITE_NAME}_cos_solar_azimuths',
    f'{SATELLITE_NAME}_lats',
    f'{SATELLITE_NAME}_lngs',
    f'{SATELLITE_NAME}_sin_solar_azimuths',
    f'{SATELLITE_NAME}_solar_zeniths',
)

# Model input columns follow the lexicographic sorted() order of feature names
# used in training ('cos_day_of_year', '<sat>_channel_1', ...), giving 15 (OLR)
# and 13 (RSR) columns.
SORTED_INPUT_NAMES_BY_MODE: dict[str, tuple[str, Ellipsis]] = {
    mode: tuple(
        sorted(
            [f'{SATELLITE_NAME}_channel_{ch}' for ch in channels]
            + list(AUXILIARY_INPUT_NAMES)
        )
    )
    for mode, channels in CHANNELS_BY_MODE.items()
}

# Column index of <sat>_solar_zeniths, used by the nighttime RSR bypass.
SZA_INDEX_BY_MODE: dict[str, int] = {
    mode: names.index(f'{SATELLITE_NAME}_solar_zeniths')
    for mode, names in SORTED_INPUT_NAMES_BY_MODE.items()
}

# Output label ordering per subsystem.
LABEL_NAMES_BY_MODE: dict[str, tuple[str, Ellipsis]] = {
    'OLR': ('ceres_olr',),
    'RSR': ('ceres_rsr',),
}

# MLP kernel shapes for first_layer_size=525, layer_units_scale_factor=0.34,
# and 4 latent layers.
EXPECTED_KERNEL_SHAPES_BY_MODE: dict[str, tuple[tuple[int, int], Ellipsis]] = {
    'OLR': ((15, 525), (525, 178), (178, 60), (60, 20), (20, 1)),
    'RSR': ((13, 525), (525, 178), (178, 60), (60, 20), (20, 1)),
}
EXPECTED_BIAS_SHAPES_BY_MODE: dict[str, tuple[tuple[int, Ellipsis], Ellipsis]] = {
    mode: tuple((n_out,) for _, n_out in kernel_shapes)
    for mode, kernel_shapes in EXPECTED_KERNEL_SHAPES_BY_MODE.items()
}

# Combined inference bitmask constants (uint8):
# 1 = OLR evaluated, 2 = RSR evaluated, 3 = both evaluated, 0 = neither.
MASK_OLR_BIT: int = 1
MASK_RSR_BIT: int = 2
MASK_BOTH_BITS: int = 3

# Negative slope of LeakyReLU activations in the COIN MLP.
LEAKY_RELU_ALPHA: float = 0.2

# Symmetric clipping half-width in standard deviations (+/- 5.5 sigma) for
# feature standardization and non-negative output normalization.
STANDARDIZATION_MAX_STDDEVS: float = 5.5

# Maximum Earth-central angle (radians) from the subsatellite point included in
# inference.
MAX_EARTH_CENTRAL_ANGLE_RADIANS: float = 1.14

# GOES ABI Full Disk 2 km target grid resolution (rows, cols).
FULL_DISK_2KM_SHAPE: tuple[int, int] = (5424, 5424)

# Downscale factor to reach the 2 km (5424, 5424) target grid:
# band 2 is 0.5 km (4x); bands 1, 3, 5 are 1 km (2x); all others are 2 km (1x).
CHANNEL_DOWNSCALE_FACTORS: dict[int, int] = {
    ch: 4 if ch == 2 else (2 if ch in (1, 3, 5) else 1) for ch in GOES_CHANNELS
}

# Native full-disk shape for each GOES ABI spectral band (channels 1..16).
CHANNEL_NATIVE_SHAPES: dict[int, tuple[int, int]] = {
    ch: (FULL_DISK_2KM_SHAPE[0] * factor, FULL_DISK_2KM_SHAPE[1] * factor)
    for ch, factor in CHANNEL_DOWNSCALE_FACTORS.items()
}

# GOES East subsatellite point and GRS80 geostationary projection parameters.
SUBSATELLITE_LAT_DEGREES: float = 0.0
SUBSATELLITE_LNG_DEGREES: float = -75.0
GOES_PERSPECTIVE_POINT_HEIGHT: float = 35786023.0
GOES_SEMI_MAJOR_AXIS: float = 6378137.0
GOES_SEMI_MINOR_AXIS: float = 6356752.31414
GOES_INVERSE_FLATTENING: float = 298.2572221
GOES_SWEEP_ANGLE_AXIS: str = 'x'
GOES_PROJECTION_KEY: str = 'goes_imager_projection'

# Default GOES East full-disk extent in meters [x_min, y_min, x_max, y_max].
GOES_AREA_EXTENT_METERS: tuple[float, float, float, float] = (
    -5434894.6753,
    -5434894.6753,
    5434894.6753,
    5434894.6753,
)

GOES_PROJ_DICT: dict[str, str] = {
    'proj': 'geos',
    'units': 'm',
    'h': str(GOES_PERSPECTIVE_POINT_HEIGHT),
    'lon_0': str(SUBSATELLITE_LNG_DEGREES),
    'a': str(GOES_SEMI_MAJOR_AXIS),
    'b': str(GOES_SEMI_MINOR_AXIS),
    'sweep': GOES_SWEEP_ANGLE_AXIS,
}

# Solar ephemeris and day-of-year periodicity constants matching COIN training.
DAYS_PER_YEAR_FOR_SOLAR_LIB: float = 365.2448979591837
DAYS_PER_YEAR_FOR_COS_DOY: float = 365.2425
RADIANS_PER_YEAR: float = 2.0 * math.pi / DAYS_PER_YEAR_FOR_SOLAR_LIB
SECONDS_PER_DAY: float = 86400.0
MINUTES_PER_DAY: float = 1440.0
RADIANS_PER_MINUTE: float = 2.0 * math.pi / MINUTES_PER_DAY
AZIMUTH_DENOMINATOR_EPSILON: float = 1e-12

# Solar zenith angle threshold (degrees) above which shortwave RSR is forced
# to 0.
NIGHT_SZA_THRESHOLD_DEGREES: float = 90.0

# Physical radiance range guard (min_allowed, max_allowed) for decoded L1b Rad
# arrays. Prevents accidental substitution of brightness temperature (~180-330 K
# on thermal bands 7-16) or normalized reflectance ([0, 1.5] on solar bands).
# Bounds are in W m-2 sr-1 um-1 (bands 1-6) and mW m-2 sr-1 (cm-1)-1 (7-16).
CHANNEL_RADIANCE_MIN_MAX: dict[int, tuple[float, float]] = {
    1: (-35.0, 1000.0),
    2: (-35.0, 1000.0),
    3: (-35.0, 600.0),
    4: (-15.0, 150.0),
    5: (-15.0, 200.0),
    6: (-15.0, 100.0),
    7: (-15.0, 60.0),
    8: (-15.0, 30.0),
    9: (-15.0, 50.0),
    10: (-15.0, 80.0),
    11: (-15.0, 200.0),
    12: (-15.0, 180.0),
    13: (-15.0, 260.0),
    14: (-15.0, 280.0),
    15: (-15.0, 280.0),
    16: (-15.0, 220.0),
}

# Allowed GOES East ABI scan modes and metadata attributes.
ALLOWED_SCAN_MODES: frozenset[int] = frozenset({3, 6})
ALLOWED_TIMELINE_IDS: frozenset[str] = frozenset({'ABI Mode 3', 'ABI Mode 6'})
ALLOWED_PLATFORM_IDS: frozenset[str] = frozenset({'G16', 'G19'})
REQUIRED_SCENE_ID: str = 'Full Disk'

# Input features standardized via Standardize(mean, stddev).
STANDARDIZED_INPUT_FEATURES: tuple[str, Ellipsis] = tuple(
    f'{SATELLITE_NAME}_channel_{ch}' for ch in GOES_CHANNELS
) + (f'{SATELLITE_NAME}_lats', f'{SATELLITE_NAME}_lngs')

# Required standardization keys in train_config.json (36 input + 4 label keys).
REQUIRED_STANDARDIZATION_KEYS: tuple[str, Ellipsis] = tuple(
    [f'{name}_mean' for name in STANDARDIZED_INPUT_FEATURES]
    + [f'{name}_stddev' for name in STANDARDIZED_INPUT_FEATURES]
    + [
        'ceres_olr_mean',
        'ceres_olr_stddev',
        'ceres_rsr_mean',
        'ceres_rsr_stddev',
    ]
)

# Fixed architecture hyperparameters required in every train_config.json.
REQUIRED_ARCHITECTURE_KEYS: dict[str, object] = {
    'activation': 'leaky_relu',
    'dropout': 0.0,
    'final_layer_activation': 'sigmoid',
    'first_layer_size': 525,
    'layer_units_scale_factor': 0.34,
    'num_latent_layers': 4,
}

# Known auxiliary keys permitted in train_config.json from training runs.
ALLOWED_AUXILIARY_CONFIG_KEYS: frozenset[str] = frozenset({
    'TRAINING_REPLICAS',
    'batch_size',
    'ceres_latitude_mean',
    'ceres_latitude_stddev',
    'ceres_longitude_mean',
    'ceres_longitude_stddev',
    'ceres_relative_azimuth_mean',
    'ceres_relative_azimuth_stddev',
    'ceres_viewing_zenith_mean',
    'ceres_viewing_zenith_stddev',
    'check_l2_sentinel_labels',
    'early_stopping_patience',
    'learning_rate',
    'loss_type',
    'lr_drop_factor',
    'lr_drop_patience',
    'min_learning_rate',
    'num_vis_batches',
    'shuffle_buffer_size',
    'steps_per_epoch',
    'steps_per_execution',
    'tfrecord_read_buffer_bytes',
    'train_epochs',
    'train_interleave_block_length',
    'train_interleave_cycle_length',
    'train_prefetch_batches',
    'validation_shuffle_seed',
    'validation_steps',
    'weight_clip',
    'weight_mode',
})

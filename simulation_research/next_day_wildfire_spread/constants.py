# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Constants for the wildfire data export."""

import immutabledict

INPUT_FEATURES = ('elevation', 'pdsi', 'NDVI', 'pr', 'sph', 'th', 'tmmn',
                  'tmmx', 'vs', 'erc', 'population', 'PrevFireMask')

OUTPUT_FEATURES = ('FireMask',)

# Data statistics computed over `train_ongoing_64_*` and `train_onset_64_*`.
# For each variable, the statistics are ordered in the form:
# `(min_clip, max_clip, mean, std)`
DATA_STATS = immutabledict.immutabledict({
    # 0.1 percentile, 99.9 percentile
    'elevation': (0., 3141., 657., 649.),
    # Pressure
    # 0.1 percentile, 99.9 percentile
    'pdsi': (-6.1, 7.9, 0., 2.7),
    'NDVI': (-9821., 9996., 5158., 2467.),  # min, max
    # Precipitation in mm.
    # Negative values do not make sense, so min is set to 0.
    # 0., 99.9 percentile
    'pr': (0.0, 44.5, 1.7, 4.5),
    # Specific humidity.
    # Negative values do not make sense, so min is set to 0.
    # The range of specific humidity is up to 100% so max is 1.
    'sph': (0., 1., 0.0072, 0.0043),
    # Wind direction in degrees clockwise from north.
    # Thus min set to 0 and max set to 360.
    'th': (0., 360.0, 190.3, 72.6),
    # Min/max temperature in Kelvin.
    # -20 degree C, 99.9 percentile
    'tmmn': (253.2, 298.9, 281.1, 9.0),
    # -20 degree C, 99.9 percentile
    'tmmx': (253.2, 315.1, 295.2, 9.8),
    # Wind velocity.
    # Negative values do not make sense, given there is a wind direction.
    # 0., 99.9 percentile
    'vs': (0.0, 10.0, 3.9, 1.4),
    # NFDRS fire danger index energy release component expressed in BTU's per
    # square foot.
    # Negative values do not make sense. Thus min set to zero.
    # 0., 99.9 percentile
    'erc': (0.0, 106., 37., 21.),
    # Population
    # min, 99.9 percentile
    'population': (0., 2534., 26., 155.),
    # We don't want to normalize the FireMasks.
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
})

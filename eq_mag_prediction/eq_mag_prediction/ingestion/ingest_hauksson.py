# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Ingests the raw Hauksson focal mechanism catalog.

The format appears here:
https://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_focal/SouthernCalifornia_1981-2011_focalmec_Format.pdf
"""

import glob
import math
import os

from absl import app
from absl import flags
import pandas as pd

from tensorflow.io import gfile
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import time_conversions

_RAW_DIRECTORY = flags.DEFINE_string(
    'raw_directory',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/raw/Hauksson'
    ),
    'The directory that contains the raw catalog files.',
)
_INGESTED_FILE = flags.DEFINE_string(
    'ingested_directory',
    os.path.join(
        os.path.dirname(__file__),
        '../..',
        'results/catalogs/ingested/hauksson.csv',
    ),
    'The path to the ingested CSV file.',
)


def parse_file(path):
  """Parses a single raw catalog file."""
  with open(path, 'r') as f:
    return pd.read_csv(
        f,
        delim_whitespace=True,
        header=None,
        names=[
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'event_id',
            'latitude',
            'longitude',
            'depth',
            'magnitude',
            'strike',
            'dip',
            'rake',
            'unused1',
            'unused2',
            'picked_phases',
            'unused3',
            'unused4',
            'unused5',
            'quality',
        ],
    )


def clean_catalog(catalog):
  """Adds useful columns, removes unused ones."""
  time_column = []
  for i in range(len(catalog)):
    second = math.floor(catalog.second.iloc[i])
    if second == 60:
      # It is unclear why these exists. There are ~10 of these in the catalog,
      # all of a fairly small magnitude, so they probably won't have a huge
      # impact.
      second -= 1
    minute = catalog.minute.iloc[i]
    if minute == -1:
      # It is also unclear why these exist. Again, there are ~5 of those, and
      # all have a small magnitude.
      minute += 1
    microsecond = int((catalog.second.iloc[i] % 1) * 1e6)
    time_column.append(
        time_conversions.datetime_utc_to_time(
            year=catalog.year.iloc[i],
            month=catalog.month.iloc[i],
            day=catalog.day.iloc[i],
            hour=catalog.hour.iloc[i],
            minute=minute,
            second=second,
            microsecond=microsecond,
        )
    )
  catalog['time'] = time_column

  catalog['x_utm'], catalog['y_utm'] = data_utils.PROJECTIONS['california'](
      catalog['longitude'].values, catalog['latitude'].values
  )
  catalog = catalog.drop(
      columns=['unused1', 'unused2', 'unused3', 'unused4', 'unused5']
  )
  return catalog


def main(_):
  result = pd.concat(
      [parse_file(path) for path in glob.glob(f'{_RAW_DIRECTORY.value}/*')]
  )

  result = clean_catalog(result)

  with open(_INGESTED_FILE.value, 'wt') as f:
    result.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

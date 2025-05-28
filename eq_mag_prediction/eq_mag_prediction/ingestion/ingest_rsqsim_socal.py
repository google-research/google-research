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

"""Ingests the RSQSim catalog for southern California.

The RSQSim is a simulator meant for long term earthquake simulations.
Details about the model can be found in the article:
https://doi.org/10.1785/0220120105

The raw data was downloaded from:
https://zenodo.org/record/5542222
"""

import glob
from absl import app
from absl import flags
import glob
import numpy as np
import pandas as pd
from tensorflow.io import gfile
from eq_mag_prediction.utilities import data_utils


_RAW_DIRECTORY = flags.DEFINE_string(
    'raw_directory',
    None,
    'The directory the raw catalog.',
)
_INGESTED_FILE = flags.DEFINE_string(
    'ingested_directory',
    None,
    'The path to the ingested CSV file.',
)
CATALOG_LATEST_DATE = (
    1609459200  # 2021, date of catalog generation in epoch time in secs
)


def parse_file(path):
  """Parses a single raw catalog file."""
  with open(path, 'r') as f:
    return pd.read_csv(f)


def clean_catalog(catalog):
  """Adds useful columns, removes unused ones, rename to convention."""

  modified_catalog = catalog.drop(columns=_columns_to_drop(), inplace=False)
  modified_catalog.rename(columns=_columns_mapping(), inplace=True)
  modified_catalog['x_utm'], modified_catalog['y_utm'] = data_utils.PROJECTIONS[
      'california'
  ](modified_catalog['longitude'].values, modified_catalog['latitude'].values)
  modified_catalog.reset_index(inplace=True)
  modified_catalog['time'] = _time_to_epoch_time(
      modified_catalog['time'].values
  )
  return modified_catalog.sort_values(by='time')


def _columns_mapping():
  columns_name_mapping = {
      'Occurrence Time (s)': 'time',
      'Magnitude': 'magnitude',
      'Hypocenter Latitude': 'latitude',
      'Hypocenter Longitude': 'longitude',
      'Hypocenter Depth (km)': 'depth',
  }
  return columns_name_mapping


def _columns_to_drop():
  """Returns a list of columns to drop."""
  columns_to_drop = [
      'Moment (N-m)',
      'Area (m^2)',
      'Number of Participating Elements',
      'Average Slip (m)',
      'Average Element Slip Rate (m/yr)',
      'Centroid Latitude',
      'Centroid Longitude',
      'Centroid Depth (km)',
      'Upper Depth (km)',
      'Lower Depth (km)',
  ]
  return columns_to_drop


def _time_to_epoch_time(time):
  return time - time.max() + CATALOG_LATEST_DATE


def main(_):
  result = pd.concat(
      [parse_file(path) for path in glob.glob(f'{_RAW_DIRECTORY.value}/*')]
  )

  result = clean_catalog(result)

  with open(_INGESTED_FILE.value, 'wt') as f:
    result.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

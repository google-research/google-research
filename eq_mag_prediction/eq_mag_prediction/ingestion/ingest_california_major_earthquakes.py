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

"""Ingests the data of major earthquakes in California.

The raw data was copied from
https://en.wikipedia.org/wiki/List_of_earthquakes_in_California Jan 29 2024.
"""

import os
import re

from absl import app
from absl import flags
import numpy as np
import pandas as pd


_RAW_CATALOG = flags.DEFINE_string(
    'raw_directory',
    None,
    'The directory the raw catalog.',
)
_INGESTED_FILE = flags.DEFINE_string(
    'ingested_directory',
    os.path.join(
        os.path.dirname(__file__),
        '../..',
        'results/catalogs/ingested/major_earthquakes_california.csv',
    ),
    'The path to the ingested CSV file.',
)


def add_time_column(catalog):
  time_logical = [(not s.startswith('18')) for s in catalog['Date'].values]
  catalog = catalog.iloc[time_logical, :]
  catalog = catalog.assign(
      time=[pd.to_datetime(d).timestamp() for d in catalog['Date'].values],
  )
  return catalog


def _get_mag_and_type(search_string):
  mag_regex = r'([\d+\.]+)\s*([a-zA-Z]*)'
  match_coor = re.findall(mag_regex, search_string)
  if match_coor:
    return match_coor[0]
  else:
    return np.nan, np.nan


def add_magnitude_and_magnitude_type_columns(
    catalog,
):
  magnitude, magnitude_type = list(
      zip(*[_get_mag_and_type(m) for m in catalog['Mag.'].values])
  )

  catalog = catalog.assign(
      magnitude=magnitude,
      magnitude_type=magnitude_type,
  )
  return catalog


def add_longitude_latitude_and_depth_columns(
    catalog,
):
  catalog = catalog.assign(
      depth=np.full(len(catalog), np.nan),
      longitude=np.full(len(catalog), np.nan),
      latitude=np.full(len(catalog), np.nan),
  )
  return catalog


def rename_clean_organize_columns(catalog):
  """Renames columns to fit format and removes unnecessary columns."""
  # Rename columns to fit format
  catalog = catalog.rename(
      columns={
          'Name': 'name',
          'Area': 'region',
      }
  )
  # Remove unnecessary columns
  catalog = catalog.drop(
      columns=[
          'Date',
          'Mag.',
          'MMI',
          'Deaths',
          'Injuries',
          'Total damage',
          'Notes',
      ],
      inplace=False,
  )
  # Sort by time
  catalog = catalog.sort_values(
      by='time',
      inplace=False,
  )
  # Reset index column
  catalog.reset_index(inplace=True)
  catalog = catalog.drop(
      columns=[
          'index',
      ],
      inplace=False,
  )
  return catalog


def parse_file(path):
  """Parses a single raw catalog file."""
  with open(path, 'r') as f:
    return pd.read_csv(f)


def main(_):
  raw_catalog = parse_file(_RAW_CATALOG.value)
  catalog = add_time_column(raw_catalog)
  catalog = add_magnitude_and_magnitude_type_columns(catalog)
  catalog = add_longitude_latitude_and_depth_columns(catalog)
  catalog = rename_clean_organize_columns(catalog)
  with open(_INGESTED_FILE.value, 'wt') as f:
    catalog.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

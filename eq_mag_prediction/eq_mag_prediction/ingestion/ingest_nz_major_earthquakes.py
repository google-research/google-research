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

"""Ingests the data of major earthquakes in New Zealand.

The raw data was copied from
https://en.wikipedia.org/wiki/List_of_earthquakes_in_New_Zealand Jan 29 2024.
"""

import datetime
import os

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
        'results/catalogs/ingested/major_earthquakes_nz.csv',
    ),
    'The path to the ingested CSV file.',
)


def _get_time(raw_catalog):
  """Parse time."""
  time = [
      datetime.datetime.strptime(s, '%d %B %Y').timestamp()
      for s in raw_catalog.Date.values
  ]
  return time


def _get_depth(raw_catalog):
  """Parse depth."""
  depth = [
      float(d[:-3]) if isinstance(d, str) else d
      for d in raw_catalog.Depth.values
  ]
  return depth


def _get_latitude(raw_catalog):
  """Parse latitude."""
  latitude = []
  for s in raw_catalog.Latitude.values:
    if isinstance(s, str):
      lat = -np.abs(float(s[: (s.find('°'))]))
      if s.find('N') != -1:
        lat = -lat
      latitude.append(lat)
    else:
      latitude.append(s)
  return latitude


def _get_longitude(raw_catalog):
  """Parse longitude."""
  longitude = []
  for s in raw_catalog.Longitude.values:
    if isinstance(s, str):
      lon = np.abs(float(s[: (s.find('°'))]))
      if s.find('E') != -1:
        pass
      elif s.find('W') != -1:
        lon = -lon
      longitude.append(lon)
    else:
      longitude.append(s)
  return longitude


def _get_magnitude_and_type(
    raw_catalog,
):
  """Parse magnitude and magnitude type."""
  magnitude_type = []
  magnitude = []
  for m in raw_catalog.itertuples():
    if np.isfinite(_try_to_float(m.MW)):
      magnitude.append(float(m.MW))
      magnitude_type.append('MW')
    elif np.isfinite(_try_to_float(m.Mb)):
      magnitude.append(float(m.Mb))
      magnitude_type.append('Mb')
    elif np.isfinite(_try_to_float(m.ML)):
      magnitude.append(float(m.ML))
      magnitude_type.append('ML')
    else:
      magnitude.append(np.nan)
      magnitude_type.append('NaN')
  return magnitude, magnitude_type


def _try_to_float(s):
  try:
    return float(s)
  except ValueError:
    return np.nan


def parse_file(path):
  """Parses a single raw catalog file."""
  with open(path, 'r') as f:
    return pd.read_csv(f)


def _clean_catalog(raw_catalog):
  raw_catalog = raw_catalog.rename(
      columns={
          '(ML)': 'ML',
          '(MW)': 'MW',
          '(Mb)': 'Mb',
      }
  )
  raw_catalog = raw_catalog.drop(range(6), axis=0)
  return raw_catalog


def _construct_new_catalog(raw_catalog):
  """Constructs the parsed fields to a new catalog."""
  time = _get_time(raw_catalog)
  depth = _get_depth(raw_catalog)
  latitude = _get_latitude(raw_catalog)
  longitude = _get_longitude(raw_catalog)
  magnitude, magnitude_type = _get_magnitude_and_type(raw_catalog)
  ingested_cat = pd.DataFrame({
      'time': time,
      'depth': depth,
      'latitude': latitude,
      'longitude': longitude,
      'magnitude': magnitude,
      'magnitude_type': magnitude_type,
      'name': raw_catalog.Location.values,
      'region': raw_catalog.Region.values,
      'note': raw_catalog['Further information'].values,
  })
  return ingested_cat


def main(_):
  raw_catalog = parse_file(_RAW_CATALOG.value)
  raw_catalog = _clean_catalog(raw_catalog)
  ingested_catalog = _construct_new_catalog(raw_catalog)
  with open(_INGESTED_FILE.value, 'wt') as f:
    ingested_catalog.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

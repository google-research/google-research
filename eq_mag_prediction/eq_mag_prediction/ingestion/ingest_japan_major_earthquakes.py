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

"""Ingests the data of major earthquakes in Japan.

The raw data was copied from
https://en.wikipedia.org/wiki/List_of_earthquakes_in_Japan Jan 29 2024.
"""

import datetime
import os
import re

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import pytz


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
        'results/catalogs/ingested/major_earthquakes_japan.csv',
    ),
    'The path to the ingested CSV file.',
)


def _datetime_string_replacement():
  return {
      'March 11, 2011\n05:46:23 UTC\n(14:46 JST)': 'March 11, 2011\n14:46 JST'
  }


def add_time_column(catalog):
  """Adds time column containing epochtime timestamps."""

  for r in catalog.iterrows():
    if r[1]['Date and time'] in _datetime_string_replacement():
      catalog.loc[r[0], 'Date and time'] = _datetime_string_replacement()[
          r[1]['Date and time']
      ]
  catalog['datetime'] = pd.to_datetime(
      catalog['Date and time'], errors='coerce'
  )

  # Remove not-a-times
  catalog = catalog.dropna(subset=['datetime'], inplace=False)

  # Set all entries to Japan timezone
  tz = pytz.timezone('Asia/Tokyo')
  catalog['datetime'] = [
      dt.replace(tzinfo=tz) for dt in catalog['datetime'].values  # pylint: disable=g-tzinfo-replace
  ]

  # Keep only entries after 1900
  datetimes_logical = [
      (pd.to_datetime(t) > datetime.datetime(year=1900, month=1, day=1))
      for t in catalog.loc[:, 'datetime'].values
  ]
  catalog = catalog.iloc[datetimes_logical, :]

  # Add a time column with epoch time
  catalog = catalog.assign(
      time=catalog['datetime'].apply(lambda x: x.timestamp()),
  )
  return catalog


def _corrected_location_strings():
  """Returns a map of Names to corrected coordinates strings."""
  location_mapper = {
      '1923 Great Kantō earthquake': '35°19.6′N 139°8.3′E depth 23 km',
      '1927 North Tango earthquake': '35.63°N 135.01°E depth 10 km',
      '1930 North Izu earthquake': '35.00°N 139.00°E depth 	19 km',
      '1933 Sanriku earthquake': '39°7.7′N 144°7′E depth 20 km',
      '1936 Miyagi earthquake': '38.2°N 142.1°E depth 61 km',
      '1940 Shakotan earthquake': '44.561°N 139.678°E depth 15 km',
      '1943 Tottori earthquake': '134.09°E 35.47°N',
      '1945 Mikawa earthquake': '34.7°N 137.1°E depth 11 km',
      '1946 Nankai earthquake': '33.00°N 135.60°E depth 30 km',
      '1964 Niigata earthquake': '38.37°N 139.22°E depth 34 km',
      '1968 Hyūga-nada earthquake': '32.45°N 132.27°E depth 30 km',
      '1968 Tokachi earthquake': '40.90°N 143.35°E depth 26 km',
      '1973 Nemuro earthquake': '43.233°N 145.785°E depth 48 km',
      '1974 Izu Peninsula earthquake': '34.6°N 138.8°E depth 9 km',
      '1978 Miyagi earthquake': '38.19°N 142.03°E depth 44 km',
      '1983 Sea of Japan earthquake': '40.462°N 139.102°E depth 24 km',
      '1984 Nagano earthquake': '35.8°N 137.6°E depth 2 km',
      '1987 Chiba earthquake': '35.372°N 140.519°E depth 62.9 km',
      'Great Hanshin earthquake': '34.59°N 135.07°E depth 17.6 km',
      '2004 Chūetsu earthquake': '37.3°N 138.8°E depth 13 km',
      '2005 Fukuoka earthquake': '33°44′18″N 130°10′30″E depth 9 km',
      '2005 Miyagi earthquake': '38.28°N 142.04°E depth 36.0 km',
      '2006 Kuril Islands earthquake': '46.592°N 153.266°E depth 31 km',
      '2007 Noto earthquake': '37.3°N 138.5°E depth 10 km',
      '2007 Chūetsu offshore earthquake': '37.535°N 138.446°E depth 10 km',
      '2008 Iwate–Miyagi Nairiku earthquake': '39°01.7′N 140°52.8′E depth 8 km',
      '2016 Kumamoto earthquakes': '32°46′55.2″N 130°43′33.6″E depth 10 km',
  }
  return location_mapper


_zero_if_none = lambda x: 0 if x is None else float(x)


def _get_lat(search_string):
  """Extracts latitude from Epicenter string."""
  lat_regex = r'([\d+\.]+)(°)?([\d+\.]+)?′?([\d+\.]+)?″?([NS])'
  ns_to_sign = lambda ns: 1 if ns == 'N' else -1

  match_coor = re.search(lat_regex, search_string)
  if match_coor:
    matches = match_coor.groups()
    lat = ns_to_sign(matches[4]) * (
        _zero_if_none(matches[0])
        + _zero_if_none(matches[2]) / 60
        + _zero_if_none(matches[3]) / 3600
    )
  else:
    lat = np.nan
  return lat


def _get_lon(search_string):
  """Extracts longitude from Epicenter string."""
  lon_regex = r'([\d+\.]+)(°)?([\d+\.]+)?′?([\d+\.]+)?″?([EW])'
  ew_to_sign = lambda ew: 1 if ew == 'E' else -1

  match_coor = re.search(lon_regex, search_string)
  if match_coor:
    matches = match_coor.groups()
    lon = ew_to_sign(matches[4]) * (
        _zero_if_none(matches[0])
        + _zero_if_none(matches[2]) / 60
        + _zero_if_none(matches[3]) / 3600
    )
  else:
    lon = np.nan
  return lon


def get_depth(search_string):
  depth_regex = r'depth\s+([\d+\.]+)'
  match_depth = re.search(depth_regex, search_string)
  if match_depth:
    return float(match_depth.groups()[0])
  else:
    return np.nan


def add_longitude_latitude_and_depth(catalog):
  """Adds longitude, latitude and depth columns."""

  # Correct descriptive locations to coordinates
  for r in catalog.iterrows():
    if r[1]['Name of quake'] in _corrected_location_strings():
      catalog.loc[r[0], 'Epicenter'] = _corrected_location_strings()[
          r[1]['Name of quake']
      ]
  # Assign extracted coordinates
  catalog = catalog.assign(
      latitude=catalog['Epicenter'].map(_get_lat),
      longitude=catalog['Epicenter'].map(_get_lon),
      depth=catalog['Epicenter'].map(get_depth),
  )
  return catalog


def _get_mag(search_string):
  mag_regex = r'([\d+\.]+)\s*([a-zA-Z]*)'
  match_coor = re.findall(mag_regex, search_string)
  if match_coor:
    resulting_mag = [t[0] for t in match_coor if t[1] == 'Mw']
    if resulting_mag:
      return float(resulting_mag[0])
    else:
      return match_coor[0][0]
  else:
    return np.nan


def add_magnitude_column(catalog):
  catalog = catalog.assign(
      magnitude=catalog['Magnitude'].map(_get_mag),
  )
  return catalog


def rename_columns(catalog):
  catalog = catalog.rename(
      columns={
          'Rōmaji name': 'region',
          'Name of quake': 'name',
      }
  )
  return catalog


def clean_catalog(catalog):
  """Removes unescessary columns."""
  catalog = catalog.drop(
      columns=[
          'Date and time',
          'Magnitude',
          'Fatalities',
          'Name in Kanji',
          'Epicenter',
          'Description',
          'datetime',
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
  catalog = add_longitude_latitude_and_depth(catalog)
  catalog = add_magnitude_column(catalog)
  catalog = rename_columns(catalog)
  catalog = clean_catalog(catalog)
  with open(_INGESTED_FILE.value, 'wt') as f:
    catalog.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

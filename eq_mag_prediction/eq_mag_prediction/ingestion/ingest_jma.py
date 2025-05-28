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

"""Ingests the raw Japan JMA catalog.

The format appears here:
https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html

_RAW_DIRECTORY is expected to contain only hypocenter files downloaded from the
JMA website.
"""

import collections
import datetime
import enum
import functools
import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import pytz

from tensorflow.io import gfile
from eq_mag_prediction.utilities import data_utils

_RAW_DIRECTORY = flags.DEFINE_string(
    'source_directory',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/raw/jma'
    ),
    'The directory that contains the raw catalog files.',
)
_INGESTED_FILE = flags.DEFINE_string(
    'ingested_path',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/ingested/jma.csv'
    ),
    'The path to the ingested CSV file.',
)


MICRO = 1e6
JAPAN_TZINFO = pytz.timezone('Japan')


################################################################################
#################### JMA string format parse functions #########################
################################################################################


class Agency(enum.Enum):
  J = 'JMA'
  U = 'USGS'
  I = 'OTHER'


def _parse_jma_agency(agency):
  return getattr(Agency, agency).value


def _optional_parser(parser, default=None):
  """Returns parser of a string that may be blank, returns default if blank."""

  def _parser(s, parser, default):
    s = s.replace(' ', '')
    if not s:
      return default
    return parser(s)

  return functools.partial(_parser, parser=parser, default=default)


def _no_decimal_parser(
    total_digits, digits_after_decimal, strip=False, zfill=True
):
  """Returns parser of float encoded as string but without the decimal point."""

  def _parser(s, total_digits, digits_after_decimal):
    if strip:
      s = s.strip()
    # For the depth field the digits after the decimal point are dropped if they
    # are 0. This fixes it. It is possible that this is the fabled I3 mode.
    if zfill:
      s = s.rstrip().ljust(total_digits, '0').lstrip()
    # This is a partial fix. In the Depth field, the format may not be F5.2 but
    # instead I3, 2X whatever that means. We just use the integer and ignore the
    # second part.
    if ' ' in s:
      return float(s.split(' ')[0])

    return float(s.rjust(total_digits, '0')) / (10**digits_after_decimal)

  return functools.partial(
      _parser,
      total_digits=total_digits,
      digits_after_decimal=digits_after_decimal,
  )


def _parse_jma_magnitude(magnitude_str):
  """Parse the Magnitude field in the JMA format.

  This is a decoding of what is described in:
  https://www.data.jma.go.jp/svd/eqev/data/bulletin/hypo_e.html

  Args:
    magnitude_str: A two-char string describing the magnitude.

  Returns:
    A float representing the magnitude.
  """
  magnitude_str = magnitude_str.strip()
  if not magnitude_str:
    return None
  first_char = magnitude_str[0]
  # This should be either one or zero characters.
  rest = magnitude_str[1:]

  if first_char == '-':
    return -float(rest) / 10.0
  if first_char.isalpha():
    integer = ord(first_char) - 64  # A = 1, B = 2, etc.
    return -integer - (float(rest) / 10.0)
  assert magnitude_str.isdigit()
  return _no_decimal_parser(2, 1, strip=True, zfill=False)(magnitude_str)


def _parse_str(s):
  """Parse string fields, stripping them first."""
  return s.strip()


_concat_chars = lambda row: ''.join(row)  # pylint: disable=unnecessary-lambda


_raw_jma_transformations = {
    'Record type identifier': _parse_jma_agency,
    'Year': int,
    'Month': int,
    'Day': int,
    'Hour': int,
    'Minute': int,
    'Second': _no_decimal_parser(4, 2),
    'Second error': _optional_parser(_no_decimal_parser(4, 2)),
    'Latitude (degrees)': _optional_parser(int),
    'Latitude (minutes)': _optional_parser(_no_decimal_parser(4, 2)),
    'Latitude error': _optional_parser(_no_decimal_parser(4, 2)),
    'Longitude (degrees)': _optional_parser(int),
    'Longitude (minutes)': _optional_parser(_no_decimal_parser(4, 2)),
    'Longitude error': _optional_parser(_no_decimal_parser(4, 2)),
    'Depth': _no_decimal_parser(5, 2),
    'Depth error': _optional_parser(_no_decimal_parser(3, 2)),
    'Magnitude 1': _parse_jma_magnitude,
    'Magnitude type 1': _parse_str,
    'Magnitude 2': _parse_jma_magnitude,
    'Magnitude type 2': _parse_str,
    'Travel time table': _optional_parser(int, default=0),
    'Hypocenter location precision': _parse_str,
    'Subsidiary information': _optional_parser(int, default=0),
    'Maximum intensity': _parse_str,
    'Damage class': _parse_str,
    'Tsunami class': _parse_str,
    'District number': _optional_parser(int),
    'Region number': _optional_parser(int),
    'Region name': _parse_str,
    'Number of stations': _optional_parser(int),
    'Hypocenter determination flag': _parse_str,
}  # See format, link in comment at the top of the file.


_data_partition = collections.OrderedDict([
    ('Record type identifier', slice(0, 1)),
    ('Year', slice(1, 5)),
    ('Month', slice(5, 7)),
    ('Day', slice(7, 9)),
    ('Hour', slice(9, 11)),
    ('Minute', slice(11, 13)),
    ('Second', slice(13, 17)),
    ('Second error', slice(17, 21)),
    ('Latitude (degrees)', slice(21, 24)),
    ('Latitude (minutes)', slice(24, 28)),
    ('Latitude error', slice(28, 32)),
    ('Longitude (degrees)', slice(32, 36)),
    ('Longitude (minutes)', slice(36, 40)),
    ('Longitude error', slice(40, 44)),
    ('Depth', slice(44, 49)),
    ('Depth error', slice(49, 52)),
    ('Magnitude 1', slice(52, 54)),
    ('Magnitude type 1', slice(54, 55)),
    ('Magnitude 2', slice(55, 57)),
    ('Magnitude type 2', slice(57, 58)),
    ('Travel time table', slice(58, 59)),
    ('Hypocenter location precision', slice(59, 60)),
    ('Subsidiary information', slice(60, 61)),
    ('Maximum intensity', slice(61, 62)),
    ('Damage class', slice(62, 63)),
    ('Tsunami class', slice(63, 64)),
    ('District number', slice(64, 65)),
    ('Region number', slice(65, 68)),
    ('Region name', slice(68, 92)),
    ('Number of stations', slice(92, 95)),
    ('Hypocenter determination flag', slice(95, 96)),
])  # See format, link in comment at the top of the file.


################################################################################
################## CSV and reading and data organization #######################
################################################################################


def _raw_file_to_raw_df(file_handle):
  all_lines = file_handle.readlines()
  data_array = np.array([[*line] for line in all_lines])
  cols_dict = {}
  for k, v in _data_partition.items():
    cols_dict[k] = np.apply_along_axis(_concat_chars, 1, data_array[:, v])
  return pd.DataFrame(cols_dict)


def _read_catalog_csv(file_path):
  """Reads the raw catalog files."""
  with open(file_path, 'r') as file_handle:
    raw_df = _raw_file_to_raw_df(file_handle)
  return raw_df


def _time_to_epoch_time(catalog):
  """Converts  date format to epoch time."""
  epoch_time = []
  for row in catalog.itertuples():
    microvalue = int(row.Second * MICRO)
    seconds = int(microvalue // MICRO)
    microseconds = int(microvalue % MICRO)
    time = datetime.datetime(
        year=row.Year,
        month=row.Month,
        day=row.Day,
        hour=row.Hour,
        minute=row.Minute,
        second=seconds,
        microsecond=microseconds,
    )
    epoch_time.append(JAPAN_TZINFO.localize(time).timestamp())
  return np.array(epoch_time)


def _get_time_err(catalog):
  """Constructs a time std column."""
  time_std = catalog['Second error'].values
  time_std[np.isnan(time_std)] = 0
  return time_std


def _get_magnitude(catalog):
  """Constructs a magnitude column."""
  return catalog['Magnitude 1'].values


def _get_longitude(catalog):
  """Constructs a longitude column."""
  return (
      catalog['Longitude (degrees)'].values
      + catalog['Longitude (minutes)'].values / 60
  )


def _get_latitude(catalog):
  """Constructs a latitude column."""
  return (
      catalog['Latitude (degrees)'].values
      + catalog['Latitude (minutes)'].values / 60
  )


def _get_longitude_err(catalog):
  """Constructs a longitude std column."""
  longitude_std = catalog['Longitude error'].values / 60
  longitude_std[np.isnan(longitude_std)] = 0
  return longitude_std


def _get_latitude_err(catalog):
  """Constructs a latitude std column."""
  latitude_std = catalog['Latitude error'].values / 60
  latitude_std[np.isnan(latitude_std)] = 0
  return latitude_std


def _columns_mapping():
  columns_name_mapping = {
      'Depth': 'depth',
  }
  return columns_name_mapping


def _clean_catalog(catalog):
  """Adds useful columns, removes unused ones, rename to convention."""
  modified_catalog = catalog.copy()
  modified_catalog = modified_catalog[
      modified_catalog['Record type identifier'] == 'JMA'
  ]
  modified_catalog['time'] = _time_to_epoch_time(modified_catalog)
  modified_catalog['time_std'] = _get_time_err(modified_catalog)
  modified_catalog['magnitude'] = _get_magnitude(modified_catalog)
  modified_catalog['longitude'] = _get_longitude(modified_catalog)
  modified_catalog['latitude'] = _get_latitude(modified_catalog)
  modified_catalog['longitude_std'] = _get_longitude_err(modified_catalog)
  modified_catalog['latitude_std'] = _get_latitude_err(modified_catalog)
  modified_catalog['x_utm'], modified_catalog['y_utm'] = data_utils.PROJECTIONS[
      'japan'
  ](modified_catalog['longitude'].values, modified_catalog['latitude'].values)
  modified_catalog.rename(columns=_columns_mapping(), inplace=True)

  cols_to_drop = list(
      set(_raw_jma_transformations.keys()).intersection(
          modified_catalog.columns
      )
  )
  modified_catalog.drop(
      columns=cols_to_drop,
      inplace=True,
  )
  modified_catalog.dropna(axis=0, inplace=True)
  modified_catalog.reset_index(inplace=True)
  modified_catalog = modified_catalog[[
      'time',
      'time_std',
      'latitude',
      'latitude_std',
      'longitude',
      'longitude_std',
      'depth',
      'magnitude',
      'x_utm',
      'y_utm',
  ]]
  return modified_catalog.sort_values(by='time')


def _read_raw_catalog(directory):
  """Reads the raw catalog files."""
  dfs = []

  raw_files = os.listdir(directory)
  sorted_file_names = sorted(raw_files)
  for f in sorted_file_names:
    raw_df = _read_catalog_csv(os.path.join(directory, f))
    raw_df_transformed = raw_df.transform(_raw_jma_transformations, axis=0)
    dfs.append(_clean_catalog(raw_df_transformed))
  all_catalogs = pd.concat(dfs, ignore_index=True)
  all_catalogs.sort_values(by='time', inplace=True)
  return all_catalogs


def main(_):
  raw_dataframe = _read_raw_catalog(_RAW_DIRECTORY.value)
  with open(_INGESTED_FILE.value, 'wt') as f:
    raw_dataframe.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

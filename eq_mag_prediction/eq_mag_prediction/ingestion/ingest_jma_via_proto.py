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

"""Ingests data from JMA files into an EarthquakeData RecordIO and an h5 file.

For more information about the data set and its format see:
https://www.data.jma.go.jp/svd/eqev/data/bulletin/hypo_e.html


Before ingestion, protobuf compiling is required:
  cd <package root>
  protoc  --proto_path=./eq_mag_prediction/ --python_out=./eq_mag_prediction/
  --pyi_out=./eq_mag_prediction earthquakes.proto

Some scripts may require import of a `***_pb2` file, which is the output of the
compilation.
"""

import collections
import datetime
import functools
import os

from absl import app
from absl import flags
from absl import logging
import pytz
import tensorflow as tf
import xarray as xr

import eq_mag_prediction import earthquakes_pb2
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import file_utils


_SOURCE_DIRECTORY = flags.DEFINE_string(
    'source_directory',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/raw/jma'
    ),
    'A directory of text files you want to ingest.',
)
_INGESTED_DIRECTORY = flags.DEFINE_string(
    'ingested_directory',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/ingested/'
    ),
    'The directory to write the output to.',
)
_OUTPUT_BASENAME = flags.DEFINE_string(
    'output_basename', 'jma', 'The basename of the output .rio and .h5 files'
)


def _parse_jma_agency(agency_str):
  assert len(agency_str) == 1
  if agency_str == 'J':
    return earthquakes_pb2.EarthquakeData.AgencyCode.JMA
  if agency_str == 'U':
    return earthquakes_pb2.EarthquakeData.AgencyCode.USGS
  if agency_str == 'I':
    return earthquakes_pb2.EarthquakeData.AgencyCode.UNSPECIFIED

  raise ValueError('Unrecognized agency code: %s' % agency_str)


def _optional_parser(parser, default=None):
  """Returns parser of a string that may be blank, returns default if blank."""

  def _parser(s, parser, default):
    s = s.strip()
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


# Describes a single field in the JMA format, length in characters.
JMAField = collections.namedtuple('JMAField', ['name', 'length', 'parser'])
# The list of all fields in the JMA format, in order.
JMA_FIELDS = [
    JMAField('Record type identifier', 1, _parse_jma_agency),
    JMAField('Year', 4, int),
    JMAField('Month', 2, int),
    JMAField('Day', 2, int),
    JMAField('Hour', 2, int),
    JMAField('Minute', 2, int),
    JMAField('Second', 4, _no_decimal_parser(4, 2)),
    JMAField('Second error', 4, _optional_parser(_no_decimal_parser(4, 2))),
    JMAField('Latitude (degrees)', 3, _optional_parser(int)),
    JMAField(
        'Latitude (minutes)', 4, _optional_parser(_no_decimal_parser(4, 2))
    ),
    JMAField('Latitude error', 4, _optional_parser(_no_decimal_parser(4, 2))),
    JMAField('Longitude (degrees)', 4, _optional_parser(int)),
    JMAField(
        'Longitude (minutes)', 4, _optional_parser(_no_decimal_parser(4, 2))
    ),
    JMAField('Longitude error', 4, _optional_parser(_no_decimal_parser(4, 2))),
    JMAField('Depth', 5, _no_decimal_parser(5, 2)),
    JMAField('Depth error', 3, _optional_parser(_no_decimal_parser(3, 2))),
    JMAField('Magnitude 1', 2, _parse_jma_magnitude),
    JMAField('Magnitude type 1', 1, _parse_str),
    JMAField('Magnitude 2', 2, _parse_jma_magnitude),
    JMAField('Magnitude type 2', 1, _parse_str),
    JMAField('Travel time table', 1, _optional_parser(int, default=0)),
    JMAField('Hypocenter location precision', 1, _parse_str),
    JMAField('Subsidiary information', 1, _optional_parser(int, default=0)),
    JMAField('Maximum intensity', 1, _parse_str),
    JMAField('Damage class', 1, _parse_str),
    JMAField('Tsunami class', 1, _parse_str),
    JMAField('District number', 1, _optional_parser(int)),
    JMAField('Region number', 3, _optional_parser(int)),
    JMAField('Region name', 24, _parse_str),
    JMAField('Number of stations', 3, _optional_parser(int)),
    JMAField('Hypocenter determination flag', 1, _parse_str),
]


def convert_jma_to_dict(jma_line):
  """Converts a single JMA-format line into a dictionary."""
  res = {}
  index = 0

  for field in JMA_FIELDS:
    try:
      value = jma_line[index : index + field.length]
      res[field.name] = field.parser(value)
    except Exception:
      logging.error(
          'Error: While parsing field %s, encountered value %s.',
          field.name,
          value,
      )
      raise
    index += field.length

  if index != len(jma_line):
    raise RuntimeError(
        'Length of line (%d) does not match total length of JMA fields (%d)'
        % (len(jma_line), index)
    )

  return res


def generate_uncertain_time(
    year, month, day, hour, minute, second, error, timezone_name
):
  """Generates an UncertainValue message describing the epoch time."""
  micro = 1e6
  microvalue = int(second * micro)
  seconds = int(microvalue // micro)
  microseconds = int(microvalue % micro)

  time = datetime.datetime(
      year,
      month,
      day,
      hour,
      minute,
      seconds,
      microseconds,
      tzinfo=pytz.timezone(timezone_name),
  ).timestamp()

  # Error is in seconds.
  utime = earthquakes_pb2.UncertainValue(value=time, std=error)
  return utime


def generate_uncertain_location(degrees, minutes, error):
  """Generates an UncertainValue message describing the location in minutes."""
  loc = degrees + float(minutes) / 60
  if error is not None:
    degree_error = float(error) / 60
  else:
    degree_error = None
  uloc = earthquakes_pb2.UncertainValue(value=loc, std=degree_error)
  return uloc


# A dictionary converting the JMA letter-encoding of the Hypocenter location
# precision field to our integer encoding of the HypocenterLocationPrecision
# enum. See details here:
# https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html
_HYPOCENTER_PRECISION_DICT = {
    '': (  # If empty, assume unknown.
        earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.UNSPECIFIED_PRECISION
    ),
    '0': (
        earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.UNSPECIFIED_PRECISION
    ),
    '1': earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.DEPTH_FREE,
    '2': earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.DEPTH_SLICE,
    '3': earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.FIXED_DEPTH,
    '4': earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.DEPTH_PHASE,
    '5': earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.SP_TIME,
    '7': earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.POOR,
    '8': (
        earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.UNSPECIFIED_PRECISION
    ),
    '9': (
        earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.HYPOCENTER_FIXED
    ),
    'M': (
        earthquakes_pb2.EarthquakeData.HypocenterLocationPrecision.MATCHED_FILTER
    ),
}

# A dictionary converting the JMA letter-encoding of the Hypocenter
# determination flag precision field to our integer encoding of the
# HypocenterDeterminationPrecision enum. See details here:
# https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html
_HYPOCENTER_FLAG_DICT = {
    'N': (
        earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.UNKNOWN_PRECISION
    ),
    '': (  # If empty, assume unknown.
        earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.UNKNOWN_PRECISION
    ),
    'K': earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.HIGH,
    'k': earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.MEDIUM,
    'A': (
        earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.AUTO_HIGHER
    ),
    'a': (
        earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.AUTO_INFERIOR
    ),
    'S': earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.LOW,
    's': (
        earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.INFERIOR
    ),
    'F': (
        earthquakes_pb2.EarthquakeData.HypocenterDeterminationPrecision.FAR_FIELD
    ),
}


def convert_record_to_proto(record):
  """Converts a single record to an EarthquakeData message.

  Args:
    record: A dictionary from field name to value.

  Returns:
    An EarthquakeData message.
  """
  proto = earthquakes_pb2.EarthquakeData()
  proto.dataset = 'jma'
  proto.agency = record['Record type identifier']
  proto.time.CopyFrom(
      generate_uncertain_time(
          record['Year'],
          record['Month'],
          record['Day'],
          record['Hour'],
          record['Minute'],
          record['Second'],
          record['Second error'],
          timezone_name='Japan',
      )
  )
  if (
      record['Latitude (degrees)'] is not None
      and record['Latitude (minutes)'] is not None
  ):
    proto.latitude.CopyFrom(
        generate_uncertain_location(
            record['Latitude (degrees)'],
            record['Latitude (minutes)'],
            record['Latitude error'],
        )
    )
  if (
      record['Longitude (degrees)'] is not None
      and record['Longitude (minutes)'] is not None
  ):
    proto.longitude.CopyFrom(
        generate_uncertain_location(
            record['Longitude (degrees)'],
            record['Longitude (minutes)'],
            record['Longitude error'],
        )
    )
  proto.depth.value = record['Depth']
  if record['Depth error'] is not None:
    proto.depth.std = record['Depth error']
  if record['Magnitude 1'] is not None:
    proto.magnitude.add().value = record['Magnitude 1']
    if record['Magnitude 2'] is not None:
      proto.magnitude.add().value = record['Magnitude 2']

  if record['Hypocenter location precision'] is not None:
    proto.loc_precision = _HYPOCENTER_PRECISION_DICT[
        record['Hypocenter location precision']
    ]

  if record['Hypocenter determination flag'] is not None:
    proto.determination_precision = _HYPOCENTER_FLAG_DICT[
        record['Hypocenter determination flag']
    ]

  return proto


def ingest_jma(source_directory, ingested_directory, output_basename):
  """Parses the raw text files and saves earthquakes as RecordIO."""
  rio_path = os.path.join(ingested_directory, output_basename + '.rio')
  with tf.io.TFRecordWriter(rio_path) as output_file:
    # Here we iterate over lines of text, which can be made much faster if
    # parallelized with Flume. A previous version of this file indeed used
    # Flume, but that broke in the transition to Python 3. We should switch back
    # to Flume if this becomes an issue.
    for line in file_utils.chain_files_iterator(source_directory):
      try:
        record = convert_jma_to_dict(line)
      except ValueError:
        logging.exception('Call to "convert_jma_to_dict" resulted in an error')
      proto = convert_record_to_proto(record)
      output_file.write(proto.SerializeToString())


def convert_protos_to_cdf(ingested_directory, output_basename):
  """Reads the proto buffers from the ingested directory and converts to cdf."""
  cdf_path = os.path.join(ingested_directory, output_basename + '.cdf')
  rio_path = os.path.join(ingested_directory, output_basename + '.rio')

  data = collections.defaultdict(list)

  for proto in tf.data.TFRecordDataset(rio_path):
    proto = earthquakes_pb2.EarthquakeData.FromString(proto.numpy())

    # 98% of the earthquakes are from the JMA agency. We only keep those for
    # consistency in magnitude calculation. Also, some earthquakes have two
    # values of magnitude, presumably because two methods of calculation were
    # used. Again, for consistency we only keep the first one.
    if (
        proto.magnitude
        and proto.magnitude[0].HasField('value')
        and proto.HasField('latitude')
        and proto.HasField('longitude')
        and proto.HasField('depth')
        and proto.HasField('agency')
        and proto.agency == 2
    ):
      data['time'].append(proto.time.value)
      data['time_std'].append(proto.time.std)
      data['latitude'].append(proto.latitude.value)
      data['latitude_std'].append(proto.latitude.std)
      data['longitude'].append(proto.longitude.value)
      data['longitude_std'].append(proto.longitude.std)
      data['depth'].append(proto.depth.value)
      data['magnitude'].append(proto.magnitude[0].value)
      if (len(data['time']) % 100000) == 0:
        logging.info('converted %d records', len(data['time']))

  logging.info('Performing UTM projection.')
  data['x_utm'], data['y_utm'] = data_utils.PROJECTIONS['japan'](
      data['longitude'], data['latitude']
  )

  logging.info('Merging.')
  data = xr.Dataset({k: (['idx'], v) for k, v in data.items()})

  logging.info('Saving to folder.')
  file_utils.save_xr_dataset(cdf_path, data)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Parsing raw files to protos.')
  ingest_jma(
      _SOURCE_DIRECTORY.value, _INGESTED_DIRECTORY.value, _OUTPUT_BASENAME.value
  )

  logging.info('Converting protos to cdf.')
  convert_protos_to_cdf(_INGESTED_DIRECTORY.value, _OUTPUT_BASENAME.value)
  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)

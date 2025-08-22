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

"""Ingests the raw SCSN catalog and serializes it as a CSV file.

Produces three files at FLAGS.ingested_directory:
  FLAGS.ingested_directory/scedc_full.csv - all events.
  FLAGS.ingested_directory/scedc.csv - only earthquakes.
  FLAGS.ingested_directory/scedc_FAILED.txt - lines which failed to parse.

The CSV files are sorted by time and their columns are

- event_id            event ID in the SCEC database
- time                seconds since epoch, produced by time_conversions
- event_type          earthquake (eq)
                      quarry blast (qb)
                      sonic boom (sn)
                      nuclear blast (nt)
                      unknown event (uk)
- geographical_type   local (l)
                      regional (r)
                      teleseism (t)   (distant earthquakes)
- magnitude
- magnitude_type      'e' energy magnitude
                      'w' moment magnitude
                      'b' body-wave magnitude
                      's' surface-wave magnitude
                      'l' local (WOOD-ANDERSON) magnitude
                      'lr' revised local magnitude
                      'c' coda amplitude
                      'h' helicorder magnitude (short-period Benioff)
                      'd' coda duration magnitude
                      'n' no magnitude
- latitude, longitude
- depth               in kilometers
- location_quality    'A'  +- 1 km horizontal distance
                           +- 2 km depth
                      'B'  +- 2 km horizontal distance
                           +- 5 km depth
                      'C'  +- 5 km horizontal distance
                           no depth restriction
                      'D'  >+- 5 km horizontal distance
                      'Z'        no quality listed in database
- picked_phases       number of picked phases in the SCEC database
- seismograms         number of seismograms available in the SCEC database
                      (i.e. # of station traces)
- x_utm, y_utm        UTM coordinates under the projection
                      data_utils.PROJECTIONS['california']

See full format specs of the raw file:
https://scedc.caltech.edu/eq-catalogs/docs/scec_dc.html.
"""

import datetime
import glob
import os
from typing import Union
from absl import app
from absl import flags
from absl import logging
import pandas as pd
from tensorflow.io import gfile
import glob
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import time_conversions

_RAW_FILES_GLOB = flags.DEFINE_string(
    'raw_files_glob',
    os.path.join(
        os.path.dirname(__file__),
        '../..',
        'results/catalogs/raw/SCEDC/*.catalog',
    ),
    'The regex for the paths of the raw catalog files.',
)
_INGESTED_DIRECTORY = flags.DEFINE_string(
    'ingested_directory',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/ingested'
    ),
    'The directory to write the output to.',
)


def _parse_line(line):
  """Parses one line from a catalog file."""
  datetime_utc = datetime.datetime.strptime(line[0:22], '%Y/%m/%d %H:%M:%S.%f')

  # Note that the datetime also contains microsecond data, which is discarded
  time = time_conversions.datetime_utc_to_time(
      datetime_utc.year,
      datetime_utc.month,
      datetime_utc.day,
      datetime_utc.hour,
      datetime_utc.minute,
      datetime_utc.second,
  )

  return {
      'time': time,
      'event_type': line[23:25],
      'geographical_type': line[27:28],
      'magnitude': float(line[29:33]),
      'magnitude_type': line[34:35],
      'latitude': float(line[38:45]),
      'longitude': float(line[46:54]),
      'depth': float(line[55:60]),
      'location_quality': line[61:62],
      'event_id': int(line[63:71]),
      'picked_phases': int(line[72:76]),
      'seismograms': int(line[77:81]),
  }


def _parse_file(path):
  """Parses a single catalog file.

  Args:
    path: The file containing a part of the full catalog.

  Returns:
    A DataFrame with the successfully parsed events and a list of strings with
    the events that failed to parse. Each string is of the format
    '{path} {line_number:05d} {line which failed to parse}'.
  """
  events, failed = [], []
  with open(path, 'rt') as f:
    for i, line in enumerate(f):
      # Lines that start with '#' are comments.
      if line.startswith('#'):
        continue

      if len(line) == 1:
        # If the file is properly formatted, this happens only after the last
        # event, and before the line that declares the number of events.
        last_line = next(f)
        try:
          next(f)
          raise ValueError(
              f'File {path} is badly formatted. An extra '
              'line was received after expected last line. '
          )
        except StopIteration:
          break

      try:
        events.append(_parse_line(line))
      except ValueError as err:
        # It seems like all failed events have the same error, namely that the
        # datetime seconds value is 60 (while the max allowed is 59).
        logging.warning(
            'Error parsing line %i in file %s. Error: %s', i, path, err
        )
        failed.append(f'{path} {i:05d} {line}')

  # Format of last line:
  # ### Number of rows returned:305
  expected_number = int(last_line.split(':')[-1])
  if expected_number != len(events) + len(failed):
    logging.error(
        'Error in parsing %s. Parsed %d events but the expected number is %d.',
        path,
        expected_number,
        len(events) + len(failed),
    )
  else:
    logging.info(
        'Parsed %s with %d events and %d failures.',
        path,
        len(events),
        len(failed),
    )

  return pd.DataFrame(events).sort_values('time'), failed


def ingest_scedc(raw_files_glob, ingested_directory):
  """Ingests all SCEDC files and saves to ingested folder."""
  events = []
  failed_events = []
  for path in glob.glob(raw_files_glob):
    new_events, new_failed = _parse_file(path)
    events.append(new_events)
    failed_events += new_failed

  events = pd.concat(events).set_index('event_id')

  logging.info('Performing UTM projection.')
  events['x_utm'], events['y_utm'] = data_utils.PROJECTIONS['california'](
      events['longitude'].values, events['latitude'].values
  )

  logging.info('Saving to folder.')
  with open(
      os.path.join(ingested_directory, 'scedc_full.csv'), 'wt'
  ) as f:
    events.to_csv(f)

  with open(os.path.join(ingested_directory, 'scedc.csv'), 'wt') as f:
    events.loc[(events.event_type == 'eq')].to_csv(f)

  with open(
      os.path.join(ingested_directory, 'scedc_FAILED.txt'), 'wt'
  ) as f:
    f.writelines(failed_events)


def main(_):
  ingest_scedc(
      raw_files_glob=_RAW_FILES_GLOB.value,
      ingested_directory=_INGESTED_DIRECTORY.value,
  )


if __name__ == '__main__':
  app.run(main)

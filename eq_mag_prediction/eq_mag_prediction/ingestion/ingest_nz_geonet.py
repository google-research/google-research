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

"""Ingests the raw New Zealand GeoNet catalog.

The format appears here:
https://www.geonet.org.nz/data/types/eq_catalogue
"""

import glob
import datetime
import glob
import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from tensorflow.io import gfile
from eq_mag_prediction.utilities import data_utils


_RAW_DIRECTORY = flags.DEFINE_string(
    'raw_directory',
    os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/raw/geonet'
    ),
    'The directory that contains the raw catalog files.',
)
_INGESTED_FILE = flags.DEFINE_string(
    'ingested_directory',
    os.path.join(
        os.path.dirname(__file__),
        '../..',
        'results/catalogs/ingested/nz_geonet.csv',
    ),
    'The path to the ingested CSV file.',
)


def _read_raw_catalog(directory):
  """Reads the raw catalog files."""
  dfs = []

  for f in glob.glob(f'{directory}/*.csv'):
    with open(f, 'rt') as f:
      dfs.append(pd.read_csv(f))

  return pd.concat(dfs)


def _clean_catalog(df):
  """Cleans up unnecessary data from the catalog."""
  clean = df[df.eventtype == 'earthquake'].copy()
  times = np.array([
      datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
      for t in clean.origintime.values
  ])
  clean['time'] = times
  clean['x_utm'], clean['y_utm'] = data_utils.PROJECTIONS['new_zealand'](
      clean['longitude'].values, clean['latitude'].values
  )
  clean = clean.drop(
      columns=[
          'publicid',
          'eventtype',
          'origintime',
          'modificationtime',
          'depthtype',
          'evaluationmethod',
          'evaluationstatus',
          'evaluationmode',
          'earthmodel',
          'usedphasecount',
          'usedstationcount',
          'magnitudestationcount',
          'minimumdistance',
          'azimuthalgap',
          'originerror',
      ]
  )
  clean = clean.sort_values('time').reindex()
  return clean


def main(_):
  raw_dataframe = _read_raw_catalog(_RAW_DIRECTORY.value)
  result = _clean_catalog(raw_dataframe)
  with open(_INGESTED_FILE.value, 'wt') as f:
    result.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)

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

"""Utilities for handling earthquake-related datasets."""

import os
from typing import Sequence

import gin
import numpy as np
import pandas as pd
import pyproj

from eq_mag_prediction.utilities import file_utils

_DAY = 24 * 3600
_YEAR = 365 * _DAY

INGESTED_DIRECTORY = os.path.join(
    os.path.dirname(__file__), '../..', 'results/catalogs/ingested'
)
PROJECTIONS = {
    # A UTM projection centered around Japan. For details: https://epsg.io/3095.
    'japan': pyproj.Proj(
        '+proj=utm +zone=54 +ellps=bessel '
        '+towgs84=-146.414,507.337,680.507,0,0,0,0 '
        '+units=m +no_defs '
    ),
    # A UTM projection centered on south California.
    # For details: https://epsg.io/26911-1750
    'california': pyproj.Proj(
        '+proj=utm +zone=11 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 '
        '+units=m +no_defs'
    ),
    # A UTM projection of New Zealand. For details: https://epsg.io/2134
    'new_zealand': pyproj.Proj(
        '+proj=utm +zone=59 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 '
        '+units=m +no_defs +type=crs'
    ),
    # A UTM projection of Italy. For details: https://epsg.io/7791
    'italy': pyproj.Proj('epsg:7791'),
}
############################
# Utility computation methods
############################


@gin.configurable
def japan_projection():
  """Convenience getter for the Japan projection, useful for Gin."""
  return PROJECTIONS['japan']


@gin.configurable
def california_projection():
  """Convenience getter for the California projection, useful for Gin."""
  return PROJECTIONS['california']


@gin.configurable
def nz_projection():
  """Convenience getter for the New Zealand projection, useful for Gin."""
  return PROJECTIONS['new_zealand']


@gin.configurable
def italy_projection():
  """Convenience getter for the Italy projection, useful for Gin."""
  return PROJECTIONS['italy']


def inside_circle(
    points, center, radius
):
  """Determine of points are in radius from given center.

  Args:
    points: A 2 dimensonal np.array of the coordinates. The shape should be
      (N_points, 2), column 0 would be x coordinates, column 1 y coordinates.
    center: A Series of length 2, containing the (x,y) coordinates of the
      intended circle. Should be in the same units as 'points'.
    radius: A float indicating the radius of the intended circle. Should be in
      the same units as 'points'.

  Returns:
    A boolean array of the same length as points. True values correspond to
    points inside the polygon.
  """
  assert points.shape[1] == 2
  center_array = np.array(center).ravel()
  assert center_array.size == 2
  points_centered = points - center_array[None, :]
  return np.sqrt((points_centered**2).sum(axis=1)) < radius


def normalize_longitude(longitude):
  """Wraps decimal degrees longitude to [-180.0, 180.0]."""
  q, r = divmod(longitude, 360.0)
  if r > 180.0 or (r == 180.0 and q <= -1.0):
    return r - 360.0
  return r


def normalize_lat_lng(latitude, longitude):
  """Check if lat,lng are on a "reasonable" range and normalize them."""
  if latitude > 90 or latitude < -90:
    raise ValueError('Expected latitude in range [-90,90], got %f.' % latitude)
  if longitude > 360 or longitude < -180:
    raise ValueError(
        'Expected longitude in range [-180,360], got %f.' % longitude
    )
  return latitude, normalize_longitude(longitude)


def concatenate_polyline(polyline):
  """Concatenates a list of lines for plotting purposes.

  Concatenates the input to a single line, with nans between subsequent lines,
  for plotting purposes.

  Args:
    polyline: A list of lines, each line of the form [[x1, x2, ...], [y1, y2,
      ...]]

  Returns:
    A single numpy array of the form
      [[x coordinates of line_1, nan, x coordinates of line_2, nan ...],
       [y coordinates of line_1, nan, y coordinates of line_2, nan ...]
      ]
  """
  nan_vector = np.array([[np.nan], [np.nan]])
  return np.hstack([np.hstack([line, nan_vector]) for line in polyline])[:, :-1]


def smear_binned_magnitudes(
    catalog,
    discretization_threshold = 0.01,
):
  """Eliminates magnitude discretization by uniform adding noise to magnitudes.

  smear_binned_magnitudes assumes that in the case of binned magnitudes two
  consecutive values exist in the catalog. E.g. if the magnitude discretization
  is of 0.1, events of magnitude M and of M+0.1 can be found in the catalog
  (without loss of generality). This should hold for any large catalog.

  Args:
    catalog: A dataframe containing a 'magnitude' column
    discretization_threshold: Minimal distance between two unique magnitudes to
      consider data as binned.

  Returns:
    a catalog with data not binned.
  """
  bin_width = np.diff(np.unique(catalog.magnitude.values)).min()
  if bin_width < discretization_threshold:
    return catalog.copy()

  rnd_seed = np.random.RandomState(seed=1905)
  mag_shifts = rnd_seed.uniform(-bin_width / 2, bin_width / 2, len(catalog))
  return_catalog = catalog.copy()
  return_catalog['magnitude'] = catalog['magnitude'].values + mag_shifts
  return return_catalog


def separate_repeating_times_in_catalog(
    catalog, orders_of_magnitude = 4
):
  """Replaces repeats in 'time' column with values tightly spaced.

  Repeating values in the 'time' column will be replaced by:
  np.linspace(repeated_val-time_margin, repeated_val+time_margin, n_repetitions)
  where time_margin is calculated as the smallest time difference (other than 0)
  divided by 10^orders_of_magnitude.

  Args:
    catalog: A dataframe containing a 'time' column
    orders_of_magnitude: int indicating how many orderd of magnitudes below the
      smallest time difference will the new replaced values span over.

  Returns:
      A copy of the original catalog with the replaced time values.
  """
  catalog_cp = catalog.copy()
  # find minimal time difference between event to ensure separation is
  # significantly smaller
  time_diffs = np.diff(np.sort(catalog_cp.time.values))
  min_time_diff = time_diffs[time_diffs != 0].min()
  time_margin = min_time_diff / (10**orders_of_magnitude)

  # find duplicated and iterate over them:
  repeating_logical = catalog_cp.duplicated(['time'], False).values
  repeating_series = catalog_cp.loc[repeating_logical, 'time']
  unique_vals, unique_counts = np.unique(
      repeating_series.values, return_counts=True
  )
  for e, t_val in enumerate(unique_vals):
    n_unique = unique_counts[e]
    new_time_values = np.linspace(
        t_val - time_margin, t_val + time_margin, n_unique
    )
    indexes_2_replace = repeating_series.index[repeating_series == t_val]
    catalog_cp.loc[indexes_2_replace, 'time'] = new_time_values

  return catalog_cp


############################
# Utility data retrieval methods
############################


def look_for_file(filename):
  """Returns path to local file if exists, else to ingested data dir.

  Args:
    filename: str.

  Returns:
    Path to file, if found.

  Raises:
    RuntimeError: if file is not found locally.
  """
  for directory in ['', INGESTED_DIRECTORY]:
    path = os.path.join(directory, filename)
    if os.path.exists(path):
      return path
  raise RuntimeError(
      'File not found.\nHave you ingested the relevant catalog? See README.md.'
  )


@gin.configurable
def scsn_dataframe(
    csv_path = 'scedc.csv', clean_columns = True
):
  """Fetches a pandas DataFrame of the SCSN earthquake catalog.

  Args:
    csv_path: Path to the csv file containing the data. The function expects a
      file created by earthquakes.ingestion.ingest_scsn().
    clean_columns: Remove often unused columns.

  Returns:
    A pandas DataFrame of the SCSN earthquake catalog.
  """
  catalog = pd.read_csv(open(look_for_file(csv_path), 'rt'))
  catalog = catalog[catalog.event_type == 'eq']
  if clean_columns:
    catalog = catalog.drop(
        columns=[
            'event_type',
            'geographical_type',
            'magnitude_type',
            'location_quality',
            'picked_phases',
            'seismograms',
        ]
    )
  return catalog.sort_values('time')


@gin.configurable
def global_cmt_dataframe(csv_path = 'global_cmt.csv'):
  """Fetches a pandas DataFrame of the Global CMT earthquake catalog."""
  df = pd.read_csv(open(look_for_file(csv_path), 'rt'))
  return df.sort_values('time')


@gin.configurable
def hauksson_dataframe(
    csv_path = 'hauksson.csv', clean_columns = True
):
  """Fetches a pandas DataFrame of the Hauksson earthquake catalog.

  Args:
    csv_path: Path to the csv file containing the data. The function expects a
      file created by earthquakes.ingestion.ingest_hauksson.
    clean_columns: Remove often unused columns.

  Returns:
    A pandas DataFrame of the Hauksson earthquake catalog.
  """
  catalog = pd.read_csv(open(look_for_file(csv_path), 'rt'))
  if clean_columns:
    catalog = catalog.drop(
        columns=[
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'picked_phases',
            'quality',
        ]
    )
  return separate_repeating_times_in_catalog(
      catalog.drop_duplicates().sort_values('time')
  )


@gin.configurable
def rsqsim_socal_dataframe(csv_path = 'rsqsim_socal.csv'):
  """Fetches a DataFrame of the RSQSim catalog for southern California."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def amatrice_dataframe(csv_path = 'amatrice_catalog.csv'):
  """Fetches a DataFrame of the Amatrice catalog for central Italy."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def jma_dataframe_from_proto(cdf_path = 'jma.cdf'):
  """Fetches a pandas DataFrame of the JMA earthquake catalog.

  Args:
    cdf_path: Path to the cdf file containing the data. The function expects a
      file created by earthquakes.ingestion.ingest_jma.convert_protos_to_cdf().

  Returns:
    A pandas DataFrame of the JMA earthquake catalog.
  """
  return (
      file_utils.load_xr_dataset(look_for_file(cdf_path))
      .to_dataframe()
      .drop_duplicates()
  )


@gin.configurable
def jma_dataframe(csv_path = 'jma.csv'):
  """Fetches a pandas DataFrame of the JMA earthquake catalog."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def nz_geonet_dataframe(csv_path = 'nz_geonet.csv'):
  """Fetches a pandas DataFrame of the New Zealand GeoNeT earthquake catalog."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def nz_major_earthquakes_dataframe(
    csv_path = 'major_earthquakes_nz.csv',
):
  """Fetches a pandas DataFrame of New Zealand's major earthquakes."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def japan_major_earthquakes_dataframe(
    csv_path = 'major_earthquakes_japan.csv',
):
  """Fetches a pandas DataFrame of Japan's major earthquakes."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def california_major_earthquakes_dataframe(
    csv_path = 'major_earthquakes_california.csv',
):
  """Fetches a pandas DataFrame of California's major earthquakes."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def sample_catalog_dataframe(
    csv_path = 'sample_catalog.csv',
):
  """Fetches a pandas DataFrame of a sample earthquake catalog."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))


@gin.configurable
def mock_catalog_dataframe(
    csv_path = 'mock.csv',
):
  """Fetches a pandas DataFrame of a mock earthquake catalog."""
  return pd.read_csv(open(look_for_file(csv_path), 'rt'))

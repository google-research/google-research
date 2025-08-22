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

"""Encoders of different data sources.

Every class here is a different encoder for some data source. Each data source
(e.g. the catalog, seismic waves, tides) can have multiple encoders.

Every encoder is responsible for the feature engineering, and constructing an
encoding model. Additionally, the encoders are responsible for normalizing their
features. In order to reproduce the same features during evaluation, the
encoders also store themselves in defined cache folder, and can be loaded (thus
fetching the same normalizer, and hyperparameters that were used during
training).

Typical usage example:

  encoder = Encoder(...)

  train_features = encoder.build_features(train_examples)
  test_features = encoder.build_features(test_examples)

  # For some encoders we may want to normalize the features.
  scaler = ml_utils.StandardScaler(encoder.scaling_axes)
  train_features = scaler.fit_transform(train_features)
  test_features = scaler.transform(test_features)

  ...

In order to reuse the same hyperparameters and scaler that were used during
training, use the `store` and `load` method for every encoder.
"""

import abc
import hashlib
import json
import os
import pickle
import tempfile
from typing import Any, Callable, Optional, Sequence, Union

import gin
import joblib
import numpy as np
import pandas as pd
import pyproj
import tensorflow as tf
import tqdm

from tensorflow.io import gfile
from eq_mag_prediction.forecasting import external_configurations
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.seismology import earthquake_properties
from eq_mag_prediction.utilities import architectures
from eq_mag_prediction.utilities import catalog_analysis
from eq_mag_prediction.utilities import catalog_processing
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.utilities import time_conversions

CACHE_FEATURES_DIR = os.path.join(
    os.path.dirname(__file__), '../..', 'results/cached_features'
)

# A type for functions that extract features from the catalog.
# The inputs are (1) the catalog, with n earthquakes, (2) a timestamp, (3) and
# (4) arrays of the x and y coordinates (respectively) of a list of m locations.
# The output is a 2d array of shape (m, n), holding the feature that corresponds
# to each location and earthquake.
CatalogFeature = Callable[
    [pd.DataFrame, int, np.ndarray, np.ndarray], np.ndarray
]
# Features for a single encoder - for train, validation, and test.
FeaturesForEncoder = tuple[np.ndarray, np.ndarray, np.ndarray]
FeaturesForAllEncoders = dict[str, FeaturesForEncoder]
# Features for all encoders, for a single slice (train, validation or test).
# The features can be either an array or a pair of arrays (for non-spatial
# encoders).
AllFeaturesForSlice = Sequence[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]

_EVENT_TIME_TOLERANCE = 1e-4  # Well under typical catalog's accuracy, [s]
# Distance filler to avoid zero division, in UTM utins:
# Set below the minimal distance in the Hauksson catalog with a m=1.8 cutoff.
_DISTANCE_EPSILON = 10
_ELASTIC_MODULUS_SCALE = 1e10  # Usual scale for elastic moduli.
_STRAIN_DEPTH_KM = 8
_STRAIN_CAP = 2e-18  # Roughly the maximal strain at 1km distance from a M=5.


class BaseEncoder(abc.ABC):
  """Base class for encoders.

  Attributes:
    name: A name for the encoder, can be used by the head model.
    scaling_axes: The axes along which a Scaler class should normalize features.
      For instance, if we want the distribution of a feature over all timestamps
      be N(0, 1), the axes would be (0, ), and we would use StandardScaler. Can
      be set to None to indicate that no normalization should be applied.
    is_location_dependent: Whether the encoder is location dependent. Location
      independent encoders have separate features per timestamp and per location
      (grid cell), while location dependent encoders have a single tensor of
      features per timestamp per location.
    n_features: The number of features on the last axis.
    build_features_kwargs: Keyword arguments that were used when the features
      were build. By adding this here, they will be stored with the encoder.
      Then, when we load the encoder and want to calculate features on new times
      or catalogs, we can use the same arguments.
  """

  def __init__(self, name, is_location_dependent):
    self.name = name
    self.is_location_dependent = is_location_dependent
    self.scaling_axes = None
    self.build_features_kwargs = {}

  def __eq__(self, other):
    self_fields = self.__dict__
    other_fields = other.__dict__
    if list(self_fields.keys()) != list(other_fields.keys()):
      return False
    compare_fields = []
    for key in self_fields:
      if isinstance(self_fields[key], (pd.DataFrame)):
        compare_fields.append(self_fields[key].equals(other_fields[key]))
      else:
        compare_fields.append(self_fields[key] == other_fields[key])
    return all(compare_fields)

  def uuid(self):
    """Creates a unique str id for an encoder instance."""
    encoder_fields = self.__dict__.copy()
    encoder_fields['_catalog'] = catalog_processing.hash_pandas_object(
        encoder_fields.pop('_catalog')
    )
    if 'feature_functions' in encoder_fields:
      encoder_fields['feature_functions'] = [
          f.__name__ for f in encoder_fields.pop('feature_functions')
      ]
    _ = encoder_fields.pop('build_features_kwargs', None)
    _ = encoder_fields.pop('n_features', None)
    encoder_identifier = json.dumps(encoder_fields, sort_keys=True)
    encoder_identifier = hashlib.sha1(
        encoder_identifier.encode('utf-8')
    ).hexdigest()
    return encoder_identifier

  def build_features_uuid(self):
    """Create a uuid str for encoder's build_features method, considering the specific argument values used."""
    build_features_identifier = json.dumps(
        gin.get_bindings(f'{self.__class__.__name__}.prepare_features'),
        sort_keys=True,
    ).encode('utf-8')
    build_features_identifier = hashlib.sha1(
        build_features_identifier
    ).hexdigest()
    return build_features_identifier

  @abc.abstractmethod
  def build_features(
      self, examples, train = False
  ):
    """Builds features for a set of examples."""

  def cache_file_location(self, name):
    """The location of a cache file of features."""
    return os.path.join(CACHE_FEATURES_DIR, f'{self.name}_{name}')

  def load_or_build_features(
      self, name, examples, train = False
  ):
    """Loads features for a region, and if they do not exists - builds them.

    This method assumes that the features are stored in cache folder with a
    consistent name per region. It is the responsibility of the user to delete
    this cache file either if the region configuration changes, or if the
    `build_features` method changes.

    Args:
      name: The name for a set of features. If the name is one that already has
        cached features, they will be loaded and not calculated.
      examples: A set of examples to calculate features.
      train: If this is the training phase, may fit some internal values.

    Returns:
      An array of features.
    """
    path = self.cache_file_location(name)
    if os.path.exists(path):
      with open(path, 'rb') as f:
        result = np.load(f)
        self.n_features = result.shape[-1]
        return result

    features = self.build_features(examples, train=train)
    if not os.path.exists(path):
      with open(path, 'wb') as f:
        np.save(f, features)
    return features

  def flatten_features(self, features):
    """Reshapes the features tensor so that the examples are per pixel.

    In general, the features are shaped as (time, x, y, feature). This method
    reshapes the features so that it is (time * x * y, feature), that is - per
    pixel.

    Args:
      features: Array of features, calculated by an encoder.

    Returns:
      The reshaped array of features.
    """
    shape = features.shape
    return features.reshape((shape[0] * shape[1] * shape[2], *shape[3:]))

  def flatten_features_per_time(self, features):
    """Reshapes the features tensor so that the examples are per timestamp.

    In general, the features are shaped as (time, x, y, feature). This method
    reshapes the features so that it is (time, feature * x * y), that is - per
    timestamp.

    Args:
      features: Array of features, calculated by an encoder.

    Returns:
      The reshaped array of features.
    """
    return np.reshape(features, (features.shape[0], -1))

  @abc.abstractmethod
  def build_model(self):
    """Builds a keras model to encode the features.

    Inheriting classes may add arguments to this method. Alternatively, they may
    call a constructor from the `architectures` module, and control the model
    with Gin.
    """

  def store(self, path):
    """Stores the encoder in a given path in cache folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
      f.truncate(0)
      pickle.dump(self, f)

  @classmethod
  def load(cls, path):
    """Loads the encoder from a file in cache folder."""
    result = joblib.load(path)
    return result


class CatalogEncoder(BaseEncoder, abc.ABC):
  """An abstract encoder that uses the earthquakes catalog."""

  def __init__(
      self, name, is_location_dependent, catalog
  ):
    super().__init__(name=name, is_location_dependent=is_location_dependent)
    self.scaling_axes = None

    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"
    self._catalog = catalog.copy()

  @abc.abstractmethod
  def build_features(
      self,
      examples,
      train = False,
      custom_catalog = None,
  ):
    """Builds features for a set of examples."""


@gin.configurable
class CatalogColumnsEncoder(CatalogEncoder):
  """An encoder for fetching specific features from the catalog.

  CatalogColumnsEncoder extracts the location and time to last event from
  the catalog to use them, raw, as features. I.e. given an example, its location
  and the time to last event will be returned. (if example's time is an event's
  time, time to previous event will be returned).
  Additional event properties may be returned by demand (e.g. depth) in this
  case the property of the closest event in time-space will be returned.
  """

  def __init__(
      self,
      catalog,
  ):
    super().__init__(
        name='catalog_earthquakes', is_location_dependent=True, catalog=catalog
    )

    self.n_features = 3
    self.scaling_axes = (0,)

  @gin.configurable('CatalogColumnsEncoder.prepare_features')
  def build_features(
      self,
      examples,
      train = False,
      custom_catalog = None,
      *,
      additional_columns = (),
  ):
    """Returns a 2d array, indices [i,j], for the i-th timestamp, j-th feature.

    Features retuned by default are the time to previous event, longitude and
    latitude.
    This function can currently receive only examples containing a single
    location per timestamp.
    Additional features can be returned on demand in which case the feature of
    the most proximate space-(past)time location according to an all-positive
    metric diag(dt,d(lon),d(lat)) is taken. Both time and distance are
    normalized by the typical scales to reduce bias of one component over the
    other.

    Args:
      examples: A mapping from times to locations. (Currently takes single
        location per time)
      train: If this is the training phase, may fit some internal values. Unused
        in this case.
      custom_catalog: If present, will use this catalog instead of the stored
        one.
      additional_columns: Additional columns from the catalog to add as
        features. Specific values will be chosen by space-time proximity,
        considering past events only.

    Returns:
      An ndarray of size [len(examples), 3+len(additional_columns)] conatining
      the requested data from the examples and catalog.
    """
    self.build_features_kwargs['additional_columns'] = additional_columns
    catalog = self._catalog if custom_catalog is None else custom_catalog
    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"

    additional_columns = list(additional_columns)
    self.n_features = 3 + len(additional_columns)

    first_locations = next(iter(examples.values()))
    assert_message = (
        'TODO: generalize this feature to handle more than'
        ' a single location per timestamp'
    )
    single_location_boolean = (len(first_locations) == 1) & (
        len(first_locations[0]) == 1
    )
    assert single_location_boolean, assert_message

    features = np.zeros((len(examples), self.n_features))

    timestamps = np.array(sorted(examples.keys()))
    for i, timestamp in iter(enumerate(timestamps)):
      assert_message = 'timestamp smaller than minimal time in catalog'
      assert timestamp >= catalog['time'].min(), assert_message
      location = examples[timestamp][0][0]
      space_time_distance = self._space_time_distance(
          timestamp, location, catalog
      )

      space_time_distance[catalog['time'].values > timestamp] = np.inf

      # find the indexes of two closest events. If the closest event is the
      # current event, time diff from previous event will require the SECOND
      # closest.
      smallest_indexes = np.argpartition(space_time_distance, 2)
      time_diff = timestamp - catalog.iloc[smallest_indexes[0]]['time']
      # time_diff should be >0 unless it is the first event in the catalog:
      if (time_diff <= _EVENT_TIME_TOLERANCE) and (smallest_indexes[0] != 0):
        time_diff = timestamp - catalog.iloc[smallest_indexes[1]]['time']
      features[i, 0] = time_diff
      features[i, 1] = location.lng
      features[i, 2] = location.lat
      features[i, 3:] = catalog.iloc[smallest_indexes[0]][
          additional_columns
      ].values
    return features

  def build_model(self):
    return architectures.identity_model(
        model_name='columns_identity', input_size=self.n_features
    )

  def flatten_features(self, features):
    return features

  def _space_time_distance(self, timestamp, point, catalog):
    space_distance = np.sqrt(
        (catalog['longitude'].values - point.lng) ** 2
        + (catalog['latitude'].values - point.lat) ** 2
    )
    space_distance = space_distance / _DISTANCE_EPSILON
    time_distance = (catalog['time'].values - timestamp) / _EVENT_TIME_TOLERANCE
    return np.sqrt(space_distance**2 + time_distance**2)


@gin.configurable(allowlist=['projection'])
def _project_utm(
    longitudes,
    latitudes,
    projection = gin.REQUIRED,
):
  """Projects longitude and latitude coordinates to UTM."""
  return projection(np.array(longitudes), np.array(latitudes))


class LocationIndependentEncoder(BaseEncoder, abc.ABC):
  """An abstract encoder that produces spatially-independent features.

  For such encoders, features are independent of the location of the example.
  Instead, this class provides a method to encode the locations of examples. The
  (spatially-independent) features will be repeated per location by the model.
  In other words, the input to the model is a tuple of two tensors - one of
  features, shaped (t, f1), and one of location encodings, shaped (t, x, y, f2).
  The features are repeated and stacked inside the model, so that the total
  representation has shape (t, x, y, f1+f2). Using this method, instead of
  repeating the features per location, can save a signifcant amount of memory.

  Attributes:
    n_location_features: The number of features on the last axis of the location
      encoding.
    location_scaling_axes: Similar to `scaling_axes`, the axes along which a
      Scaler class should normalize the location features.
  """

  def __init__(self, name, is_location_dependent = False):
    # This is silly, but required by the multiple inheritance to pass the
    # argument.
    assert not is_location_dependent
    super().__init__(name=name, is_location_dependent=False)
    self.scaling_axes = None
    self.location_scaling_axes = None
    self.n_location_features = 2

  def spatial_encoding_cache_file_location(self, name):
    """The location of a cache file of features."""
    return os.path.join(CACHE_FEATURES_DIR, f'{self.name}_{name}_spatial')

  def load_or_build_location_features(
      self,
      name,
      examples,
      *,
      total_pixels,
      first_pixel_index,
      total_regions,
      region_index,
  ):
    """Loads features for a region, and if they do not exists - builds them.

    This method assumes that the features are stored in cache folder with a
    consistent name per region. It is the responsibility of the user to delete
    this cache file either if the region configuration changes, or if the
    `build_features` method changes.

    Args:
      name: The name for a set of features. If the name is one that already has
        cached features, they will be loaded and not calculated.
      examples: A set of examples to calculate features.
      total_pixels: The total number of pixels that participate in the
        experiment, across the different regions.
      first_pixel_index: The index of the first pixel for the current encoder.
      total_regions: The total number of regions that participate in the
        experiment.
      region_index: The index of the region for the current encoder.

    Returns:
      An array of features.
    """
    path = self.spatial_encoding_cache_file_location(name)
    if os.path.exists(path):
      with open(path, 'rb') as f:
        result = np.load(f)
        self.n_location_features = result.shape[-1]
        return result

    features = self.build_location_features(
        examples,
        total_pixels=total_pixels,
        first_pixel_index=first_pixel_index,
        total_regions=total_regions,
        region_index=region_index,
    )
    if not os.path.exists(path):
      with open(path, 'wb') as f:
        np.save(f, features)
    return features

  @abc.abstractmethod
  def build_location_features(
      self, examples, **kwargs
  ):
    """Returns an encoding of locations per timestamp."""

  def flatten_features(self, features):
    """Overrides the reshaping, because here the examples are per timestamp."""
    return features

  def expand_features(
      self, features, shape
  ):
    """Repeats the spatially independent features per location.

    Note that this happens in memory, so it should be avoided, unless done in
    batches.

    Args:
      features: Array of features, calculated by an encoder.
      shape: The grid shape that we want to expand the features to.

    Returns:
      An array with shape (time, x, y, ...).
    """
    old_shape = features.shape
    features = np.expand_dims(features, axis=(1, 2))
    features = np.repeat(features, shape[0], axis=1)
    features = np.repeat(features, shape[1], axis=2)
    return features.reshape(
        (old_shape[0] * shape[0] * shape[1], *old_shape[1:])
    )

  def flatten_location_features(self, features):
    """Reshapes the spatial encoding so that the examples are per pixel."""
    shape = features.shape
    return features.reshape((shape[0] * shape[1] * shape[2], *shape[3:]))


@gin.configurable
def _mock_earthquake(
    catalog, add_angles = False
):
  """Creates a tiny earthquake in the mean location of the catalog."""
  if len(catalog) == 0:  # pylint: disable=g-explicit-length-test
    dict_for_df = {
        'time': [time_conversions.datetime_japan_to_time(1900)],
        'longitude': [140],
        'latitude': [38],
        'depth': [40],
        'magnitude': [-3],
        'event_id': 0,
    }
    if add_angles:
      dict_for_df['strike'] = 0
      dict_for_df['rake'] = 0
      dict_for_df['dip'] = np.pi / 2
    return pd.DataFrame(dict_for_df)
  dict_for_df = {
      'time': [catalog.time.min()],
      'longitude': [catalog.longitude.mean()],
      'latitude': [catalog.latitude.mean()],
      'depth': [catalog.depth.mean()],
      'magnitude': [-3],
      'event_id': 0,
  }
  if add_angles:
    dict_for_df['strike'] = [catalog.strike.mean()]
    dict_for_df['rake'] = [catalog.rake.mean()]
    dict_for_df['dip'] = [catalog.dip.mean()]
  return pd.DataFrame(dict_for_df)


def _repeat_grid(
    array, grid
):
  """Repeats an input array along 2 dimensions, matching the grid size."""
  result = array.reshape(1, *array.shape).repeat(len(grid[0]), axis=0)
  return result.reshape(1, *result.shape).repeat(len(grid), axis=0)


def _distances(
    catalog,
    longitudes,
    latitudes,
):
  """Calculates the distance between every earthquake and every coordinate."""
  xs, ys = _project_utm(longitudes, latitudes)
  catalog_xs, catalog_ys = _project_utm(
      catalog.longitude.values, catalog.latitude.values
  )
  return (
      np.subtract.outer(xs, catalog_xs) ** 2
      + np.subtract.outer(ys, catalog_ys) ** 2
  ) ** 0.5


def _sines(
    catalog, longitudes, latitudes
):
  """Finds the sines of angles between every earthquake and coordinate."""
  _, ys = _project_utm(longitudes, latitudes)
  _, catalog_ys = _project_utm(
      catalog.longitude.values, catalog.latitude.values
  )
  distances = _distances(catalog, longitudes, latitudes)
  # For points that coincide with earthquake locations, the distance is 0, so
  # the angle is arbitrary (and dividing by 0 gives NaN valules).
  distances[np.where(distances == 0)] = 1
  return -np.subtract.outer(ys, catalog_ys) / distances


def _cosines(
    catalog, longitudes, latitudes
):
  """Finds the cosines of angles between every earthquake and coordinate."""
  xs, _ = _project_utm(longitudes, latitudes)
  catalog_xs, _ = _project_utm(
      catalog.longitude.values, catalog.latitude.values
  )
  distances = _distances(catalog, longitudes, latitudes)
  # For points that coincide with earthquake locations, the distance is 0, so
  # the angle is arbitrary (and dividing by 0 gives NaN valules).
  distances[np.where(distances == 0)] = 1
  return -np.subtract.outer(xs, catalog_xs) / distances


def _catalog_features_limited_earthquakes(
    catalog,
    sort_by,
    examples,
    lookback_seconds,
    max_earthquakes,
    feature_functions,
    display_progress = False,
):
  """Generates a raw feature numpy array for a limited number of earthquakes.

  Selects at most `max_earthquakes` in the past `lookback_seconds`, sorted by
  `sort_by`, and then calculates feature functions on the resulting entries.

  Args:
    catalog: Catalog of earthquakes.
    sort_by: Column name to sort earthquakes by, before choosing the top few.
      The sort is ascending, and we take the last `max_earthquakes` (so we take
      the most recent earthquakes if we sort by time, or the greatest magnitude
      earthquakes if we sort by magnitude).
    examples: Times and locations at which to calculate the features.
    lookback_seconds: Number of seconds in the past to search for earthquakes.
    max_earthquakes: Limit for the number of earthquakes to keep. If there are
      less than `max_earthquakes` in the time range, mock earthquakes with tiny
      magnitudes will be added.
    feature_functions: Functions for extracting features from the catalog.
    display_progress: Display the progress along timestamps for feature
      creation.

  Returns:
    Numpy array of earthquake features. The first axis corresponds to
    timestamps, the second to locations, the third to an earthquake (sorted by
    `sort_by`, a total of `max_earthquakes`) and the fourth to
    `feature_functions`. A special feature is added to indicate whether the
    earthquake was a mock earthquake or a real one.
  """
  timestamps = np.array(sorted(examples.keys()))
  # For each evaluation time, find the index of the first earthquake after it,
  # and the index of the last earthquake that fits in the time range.
  first_earthquakes_indices = np.searchsorted(
      catalog.time, timestamps, side='left'
  )
  last_earthquake_indices = np.searchsorted(
      catalog.time, timestamps - lookback_seconds, side='left'
  )
  first_locations = next(iter(examples.values()))
  features = np.zeros((
      len(examples),
      len(first_locations),
      len(first_locations[0]),
      max_earthquakes,
      len(feature_functions) + 1,
  ))
  if display_progress:
    iter_timesstamps = tqdm.tqdm(enumerate(timestamps))
  else:
    iter_timesstamps = iter(enumerate(timestamps))
  for i, timestamp in iter_timesstamps:
    xs = np.array([[point.lng for point in row] for row in examples[timestamp]])
    ys = np.array([[point.lat for point in row] for row in examples[timestamp]])
    first_earthquake_index = first_earthquakes_indices[i]
    last_earthquake_index = last_earthquake_indices[i]

    # We used side='left' when searching, so `first_earthquake_index - 1` is
    # always before the current time.
    earthquake_indices = slice(last_earthquake_index, first_earthquake_index)
    subcatalog = (
        catalog.iloc[earthquake_indices]
        .sort_values(sort_by, ascending=True)
        .tail(max_earthquakes)
        .copy()
    )
    # Fill with default values, so that the length will always be the same.
    to_append = _mock_earthquake(subcatalog)
    is_mock_feature_array = np.zeros(features.shape[1:-1])
    for index in range(len(subcatalog), max_earthquakes):
      subcatalog = pd.concat([subcatalog, to_append])
      is_mock_feature_array[:, :, index] = 1

    features[i, :, :, :, -1] = is_mock_feature_array
    for k, function in enumerate(feature_functions):
      features[i, :, :, :, k] = function(subcatalog, timestamp, xs, ys)

  return features


@tf.function
def _repeat(tensor, times):
  result = tf.expand_dims(tensor, 1)
  return tf.repeat(result, times, axis=1)


def _strain_tensor(
    catalog, longitudes, latitudes
):
  """Returns the strain tensor for every earthquake and every location."""
  xs, ys = _project_utm(longitudes, latitudes)
  result = np.zeros((3, 3, len(catalog), *xs.shape))
  catalog = catalog.reset_index()
  for i, eq in catalog.iterrows():
    strike, rake, dip = (
        np.deg2rad(eq.strike),
        np.deg2rad(eq.rake),
        np.deg2rad(eq.dip),
    )
    relative_x, relative_y = (xs - eq.x_utm), (ys - eq.y_utm)
    # Converting to meters.
    relative_depth = np.ones(xs.shape) * (eq.depth - _STRAIN_DEPTH_KM) * 1000
    relative_coordinates = np.stack([relative_x, relative_y, relative_depth])
    result[:, :, i, :] = (
        earthquake_properties.double_couple_strain_tensor_in_ned(
            strike, rake, dip, eq.magnitude, relative_coordinates
        )
    )
  result = np.clip(result, -_STRAIN_CAP, _STRAIN_CAP)
  result *= _ELASTIC_MODULUS_SCALE
  return np.stack(
      [
          result[0, 0],
          result[1, 1],
          result[2, 2],
          result[0, 1],
          result[0, 2],
          result[1, 2],
      ],
      axis=-1,
  )


def _regular_grid_shifts(side_deg, n_points_side):
  sub_square_side = side_deg / n_points_side
  from_lng = from_lat = -side_deg / 2 + sub_square_side / 2
  res = np.zeros((n_points_side, n_points_side, 2))
  for i in range(n_points_side):
    res[i, :, 0] = from_lng + i * sub_square_side
    res[:, i, 1] = from_lat + i * sub_square_side
  return res


# pylint: disable=unused-argument
# The time that elapsed since the last earthquake (and derived features).
def _time_diff(df, t):
  return t - df.time.values


def _inverse_time_diff(df, t):
  return 1 / (t + 1e-3 - df.time.values)


def _log_time_diff(df, t):
  return np.log(t + 1e-3 - df.time.values)


# The exponent of the magnitude of the last earthquake.
def _magnitude(df, t):
  return np.exp(df.magnitude.values)


def _inverse_exp_magnitude(df, t):
  return 1 / np.exp(df.magnitude.values)


# The location of the last earthquake.
def _utm_x(df, t):
  return _project_utm(df.longitude.values, df.latitude.values)[0]


def _utm_y(df, t):
  return _project_utm(df.longitude.values, df.latitude.values)[1]


def _depth(df, t):
  return df.depth.values


def _log_depth(df, t):
  depth_vals = df.depth.values
  depth_epsilon = 1e-3
  depth_log = np.where(
      depth_vals >= 0,
      np.log(depth_epsilon + depth_vals),
      -np.log(depth_epsilon - depth_vals),
  )
  return depth_log


# pylint: enable=unused-argument


def _stations_to_locations(
    stations, station_codes
):
  station_locations = []
  for station_code in station_codes:
    station = stations[stations.code == station_code]
    station_locations.append(
        (station.longitude.values[0], station.latitude.values[0])
    )
  return station_locations


@gin.configurable
class RecentEarthquakesEncoder(CatalogEncoder, LocationIndependentEncoder):
  """An encoder for learning a function over some recent earthquakes.

  Constants:
    FEATURE_FUNCTIONS: The functions that are calculated for every past
      earthquake. Every functions takes as input the subcatalog of past
      earthquakes, and the time of the example.
  """

  FEATURE_FUNCTIONS = (
      _time_diff,
      _inverse_time_diff,
      _log_time_diff,
      _magnitude,
      _inverse_exp_magnitude,
      _utm_x,
      _utm_y,
  )

  def __init__(
      self,
      catalog,
      magnitude_threshold,
      use_depth_as_feature = True,
  ):
    super().__init__(
        name='recent_earthquakes', is_location_dependent=False, catalog=catalog
    )

    self.feature_functions = self.FEATURE_FUNCTIONS
    if use_depth_as_feature:
      self._add_depth_as_feature()

    self.scaling_axes = (0, 1)
    self.location_scaling_axes = (0, 1, 2)
    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"
    self._magnitude_threshold = magnitude_threshold
    self._catalog = catalog[catalog.magnitude >= magnitude_threshold].copy()
    # There is 1 extra feature, to indicate whether the earthquake is fake.
    self.n_features = len(self.feature_functions) + 1

  @gin.configurable('RecentEarthquakesEncoder.build_location_features')
  def build_location_features(
      self,
      examples,
      *,
      total_pixels = 0,
      first_pixel_index = 0,
      total_regions = 0,
      region_index = 0,
      **kwargs,
  ):
    """Returns an encoding of locations per timestamp."""
    self.n_location_features = total_pixels + total_regions + 2

    timestamps = sorted(examples.keys())
    first_locations = next(iter(examples.values()))
    features = np.zeros((
        len(examples),
        len(first_locations),
        len(first_locations[0]),
        self.n_location_features,
    ))

    for i, timestamp in enumerate(timestamps):
      lngs = np.array(
          [[point.lng for point in row] for row in examples[timestamp]]
      )
      lats = np.array(
          [[point.lat for point in row] for row in examples[timestamp]]
      )
      xs, ys = _project_utm(lngs, lats)
      features[i, :, :, -2] = xs
      features[i, :, :, -1] = ys

    if (total_pixels == 0) & (total_regions == 0):
      return features

    features[:, :, :, region_index] = 1
    pixel = 0
    for row_index, row in enumerate(first_locations):
      for col_index in range(len(row)):
        total_pixel_index = total_regions + first_pixel_index + pixel
        features[:, row_index, col_index, total_pixel_index] = 1
        pixel += 1

    return features

  @gin.configurable('RecentEarthquakesEncoder.prepare_features')
  def build_features(
      self,
      examples,
      train = False,
      custom_catalog = None,
      *,
      limit_lookback_seconds = gin.REQUIRED,
      max_earthquakes = gin.REQUIRED,
  ):
    """Builds features for a set of examples.

    Args:
      examples: A mapping from times to locations.
      train: If this is the training phase, may fit some internal values. Unused
        in this case.
      custom_catalog: If present, will use this catalog instead of the stored
        one.
      limit_lookback_seconds: Number of seconds in the past to search for the
        recent earthquakes.
      max_earthquakes: Limit for the number of earthquakes to keep. If there are
        less than `max_earthquakes` in the time range, mock earthquakes with
        tiny magnitudes will be added.

    Returns:
      A 3-dimensional array, that holds at index [i,j,k] the application of th
      k-th feature function on the j-th (feature) earthquake before the i-th
      example.
    """
    self.build_features_kwargs['limit_lookback_seconds'] = (
        limit_lookback_seconds
    )
    self.build_features_kwargs['max_earthquakes'] = max_earthquakes
    catalog = self._catalog if custom_catalog is None else custom_catalog
    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"

    timestamps = np.array(sorted(examples.keys()))
    # For each evaluation time, find the index of the first earthquake after it,
    # and the index of the last earthquake that fits in the time range.
    first_earthquakes_indices = np.searchsorted(
        catalog.time, timestamps, side='left'
    )
    last_earthquake_indices = np.searchsorted(
        catalog.time, timestamps - limit_lookback_seconds, side='left'
    )

    features = np.zeros((len(examples), max_earthquakes, self.n_features))
    for i, timestamp in enumerate(timestamps):
      first_earthquake_index = first_earthquakes_indices[i]
      last_earthquake_index = last_earthquake_indices[i]

      # We used side='left' when searching, so `first_earthquake_index - 1` is
      # always before the current time.
      earthquake_indices = slice(last_earthquake_index, first_earthquake_index)
      subcatalog = (
          catalog.iloc[earthquake_indices]
          .sort_values('time', ascending=True)
          .tail(max_earthquakes)
          .copy()
      )
      # Fill with default values, so that the length will always be the same.
      to_append = _mock_earthquake(subcatalog)
      is_mock_feature_array = np.zeros((max_earthquakes,))
      for index in range(len(subcatalog), max_earthquakes):
        subcatalog = pd.concat([subcatalog, to_append])
        is_mock_feature_array[index] = 1

      features[i, :, len(self.feature_functions)] = is_mock_feature_array
      for k, function in enumerate(self.feature_functions):
        features[i, :, k] = function(subcatalog, timestamp)

    return features

  @gin.configurable('RecentEarthquakesEncoder.build_model')
  def build_model(
      self,
      kernel_regularization = None,
      **kwargs,
  ):
    """Builds a keras model to encode the features.

    This architecture combines the spatially-independent encoding with the
    spatially-dependent encoding.

    The architecture first shifts some features of the spatially-independent
    encoding (currently, the -2nd and -3rd features) by the values of the
    spatially-dependent encoding. Intuitively, this should cause the model to
    learn more spatially-independent rules (instead of 'remembering' which
    (x_i, y_i) correlate to which labels in the training set, hopefully making
    everything equivariant).
    The model also applies some feature functions on the shifted coordinates.

    Args:
      kernel_regularization: Regularization to be used by all components of the
        model.
      **kwargs: Keyword arguments to be passed to the underlying architecture
        (such as layer sizes, regularization, activation).

    Returns:
      A Keras model that combines the spatially-independent and the
      spatially-dependent encoding, and then applies a recurrent model on the
      encoded earthquakes.
    """
    input_layer = tf.keras.layers.Input(
        shape=(None, self.n_features), name=self.name, dtype='float64'
    )
    input_location_layer = tf.keras.layers.Input(
        shape=(self.n_location_features,),
        name=f'{self.name}_location',
        dtype='float64',
    )
    n_past_examples = tf.shape(input_layer)[1]

    cell_location = tf.keras.layers.Lambda(lambda inp: inp[:, -2:])(
        input_location_layer
    )
    cell_location = tf.keras.layers.Lambda(lambda inp: _repeat(inp[0], inp[1]))(
        (cell_location, n_past_examples)
    )

    output_layer = input_layer
    output_layer = tf.keras.layers.Concatenate(axis=2)(
        [output_layer, cell_location]
    )

    def translation(tensor):
      """Translate features at specific coordinates, as described above."""
      x, y = tensor[:, :, -2], tensor[:, :, -1]
      x_event, y_event = tensor[:, :, -5], tensor[:, :, -4]
      is_mock_earthquake = tensor[:, :, -3]
      other_features = tensor[:, :, :-5]
      shifted_x = tf.subtract(x_event, x)
      shifted_y = tf.subtract(y_event, y)
      square_dist = tf.square(shifted_x) + tf.square(shifted_y)
      log_dist = tf.math.log(square_dist + 1e-8)
      pixel_adjusted_features = tf.stack(
          [shifted_x, shifted_y, square_dist, log_dist, is_mock_earthquake],
          axis=-1,
      )
      return tf.concat([other_features, pixel_adjusted_features], axis=-1)

    output_layer = tf.keras.layers.Lambda(translation)(output_layer)
    layer_size = self.n_features
    if self.n_location_features > 2:
      location_features = tf.keras.layers.Lambda(lambda inp: inp[:, :-2])(
          input_location_layer
      )
      spatial_embedding_model = architectures.fully_connected_model(
          f'{self.name}_location', self.n_location_features - 2
      )
      n_encoded_location_features = spatial_embedding_model.output_shape[-1]
      encoded_location = spatial_embedding_model(location_features)
      encoded_location = tf.keras.layers.Lambda(
          lambda inp: _repeat(inp[0], inp[1])
      )((encoded_location, n_past_examples))

      n_combined_features = layer_size + n_encoded_location_features + 2
      output_layer = tf.keras.layers.Concatenate(axis=2)(
          [output_layer, encoded_location]
      )
    else:
      n_combined_features = layer_size + 2
    output_layer = architectures.rnn_model(
        self.name,
        n_combined_features,
        kernel_regularization=kernel_regularization,
        **kwargs,
    )(output_layer)

    return tf.keras.models.Model(
        inputs=[input_layer, input_location_layer], outputs=output_layer
    )

  def _add_depth_as_feature(self):
    # UTM needs to be last in the features to fit the model construction.
    self.feature_functions = (_depth, _log_depth) + self.feature_functions


@gin.configurable
class BiggestEarthquakesEncoder(CatalogEncoder):
  """An encoder for learning a function over the biggest earthquakes in history.

  Constants:
    FEATURE_FUNCTIONS: The functions that are calculated for every past
      earthquake. Every functions takes as input the subcatalog of past
      earthquakes, and the time of the example.
      the cells.
  """

  FEATURE_FUNCTIONS = (
      # pylint: disable=g-long-lambda
      # The time that elapsed since the last earthquake (and derived features).
      lambda df, t, xs, ys: _repeat_grid(t - df.time.values, xs),
      lambda df, t, xs, ys: _repeat_grid(1 / (t + 1e-3 - df.time.values), xs),
      lambda df, t, xs, ys: _repeat_grid(np.log(t + 1e-3 - df.time.values), xs),
      # The exponent of the magnitude of the last earthquake.
      lambda df, t, xs, ys: _repeat_grid(np.exp(df.magnitude.values), xs),
      lambda df, t, xs, ys: _repeat_grid(1 / np.exp(df.magnitude.values), xs),
      # The depth of the last earthquake.
      lambda df, t, xs, ys: _repeat_grid(df.depth.values, xs),
      lambda df, t, xs, ys: _repeat_grid(np.log(1e-3 + df.depth.values), xs),
      # The location of the last earthquake, with respect to the cell.
      lambda df, t, xs, ys: 1 / (_distances(df, xs, ys) + _DISTANCE_EPSILON),
      lambda df, t, xs, ys: (
          1 / ((_distances(df, xs, ys) + _DISTANCE_EPSILON) ** 2)
      ),
      lambda df, t, xs, ys: _distances(df, xs, ys),
      lambda df, t, xs, ys: (
          np.log((_distances(df, xs, ys) + _DISTANCE_EPSILON) ** 2)
      ),
      lambda df, t, xs, ys: np.subtract.outer(xs, df.longitude.values),
      lambda df, t, xs, ys: np.subtract.outer(ys, df.latitude.values),
      lambda df, t, xs, ys: _sines(df, xs, ys),
      lambda df, t, xs, ys: _cosines(df, xs, ys),
      # pylint: enable=g-long-lambda
  )

  def __init__(self, catalog):
    super().__init__(
        name='biggest_earthquakes', is_location_dependent=True, catalog=catalog
    )
    self.scaling_axes = (0, 1, 2, 3)
    # There is 1 extra feature, to indicate whether the earthquake is fake.
    self.n_features = len(BiggestEarthquakesEncoder.FEATURE_FUNCTIONS) + 1

  @gin.configurable('BiggestEarthquakesEncoder.prepare_features')
  def build_features(
      self,
      examples,
      train = False,
      custom_catalog = None,
      *,
      limit_lookback_seconds = gin.REQUIRED,
      max_earthquakes = gin.REQUIRED,
      display_progress = False,
  ):
    """Builds features for a set of examples.

    Args:
      examples: A mapping from times to locations.
      train: If this is the training phase, may fit some internal values. Unused
        in this case.
      custom_catalog: If present, will use this catalog instead of the stored
        one.
      limit_lookback_seconds: Number of seconds in the past to search for the
        recent earthquakes.
      max_earthquakes: Limit for the number of earthquakes to keep. If there are
        less than `max_earthquakes` in the history, mock earthquakes with tiny
        magnitudes will be added.
      display_progress: Display the progress along timestamps for feature
        creation.

    Returns:
      A 5-dimensional array, that holds at index [i,j,k,l,m] the application of
      the m-th feature function on the l-th biggest (feature) earthquake before
      the i-th example, at location j,k in the grid.
    """
    self.build_features_kwargs['limit_lookback_seconds'] = (
        limit_lookback_seconds
    )
    self.build_features_kwargs['max_earthquakes'] = max_earthquakes
    catalog = self._catalog if custom_catalog is None else custom_catalog
    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"

    return _catalog_features_limited_earthquakes(
        catalog,
        'magnitude',
        examples,
        limit_lookback_seconds,
        max_earthquakes,
        self.FEATURE_FUNCTIONS,
        display_progress,
    )

  @gin.configurable(
      'BiggestEarthquakesEncoder.build_model', denylist=['input_shape']
  )
  def build_model(self, **kwargs):
    """Builds a keras model to encode the features."""
    return architectures.rnn_model(self.name, self.n_features, **kwargs)


@gin.configurable
class SeismicityGridEncoder(CatalogEncoder):
  """An encoder for specifying the rate of earthquake over past periods."""

  def __init__(self, catalog):
    super().__init__(
        name='seismicity_grid', is_location_dependent=True, catalog=catalog
    )
    self.scaling_axes = None
    self.n_features = None
    self.grid_size = None

  @gin.configurable('SeismicityGridEncoder.prepare_features')
  def build_features(
      self,
      examples,
      train = False,
      custom_catalog = None,
      *,
      grid_n_side_points = gin.REQUIRED,
      grid_side_deg = gin.REQUIRED,
      lookback_seconds = gin.REQUIRED,
      magnitudes = gin.REQUIRED,
  ):
    """Builds features for a set of examples.

    Args:
      examples: A mapping from times to locations.
      train: If this is the training phase, may fit some internal values. Unused
        in this case.
      custom_catalog: If present, will use this catalog instead of the stored
        one.
      grid_n_side_points: The features will be calculated on a NxN grid around
        the center, this value specifies the N.
      grid_side_deg: The features will be calculated on a NxN grid around the
        center, this value specifies the size (in degrees) of every cell.
      lookback_seconds: Time intervals (in the past from the evaluation times)
        for calculation of seismicity rates.
      magnitudes: Magnitude thresholds to calculate seismicity.

    Returns:
      A 6-dimensional array. The 1st dimension is time (of the example), the 2nd
      and 3rd are equal to 1 (we currently only support 1D examples). The 4th
      and 5th dimensions correspond to a pixel in the region (sized
      `grid_side_deg`). The 6th dimension is the feature dimension. Features are
      representation of a 2D histogram of earthquakes above certain magnitude
      thresholds and at given lookback times before the example.
    """
    catalog = self._catalog if custom_catalog is None else custom_catalog
    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"
    self.n_features = len(magnitudes) * len(lookback_seconds)
    self.grid_size = (grid_n_side_points, grid_n_side_points)
    self.build_features_kwargs['grid_n_side_points'] = grid_n_side_points
    self.build_features_kwargs['grid_side_deg'] = grid_side_deg
    self.build_features_kwargs['lookback_seconds'] = lookback_seconds
    self.build_features_kwargs['magnitudes'] = magnitudes

    timestamps = sorted(examples.keys())
    first_locations = next(iter(examples.values()))
    assert (len(first_locations) == 1) & (
        len(first_locations[0]) == 1
    ), 'Currently only supporting 1D examples.'
    center = first_locations[0][0]

    magnitudes = sorted(magnitudes, reverse=True)
    lookback_seconds = np.array(sorted(lookback_seconds, reverse=True)).astype(
        'float64'
    )
    n_grid, side = grid_n_side_points, grid_side_deg
    lng_bins = np.arange(n_grid + 1) * side + center.lng - n_grid * side / 2
    lat_bins = np.arange(n_grid + 1) * side + center.lat - n_grid * side / 2
    features = np.zeros((len(examples), 1, 1, *self.grid_size, self.n_features))

    index = 0
    for magnitude in magnitudes:
      subcatalog = catalog[catalog.magnitude >= magnitude]
      times_array = subcatalog.time.values
      for lookback in lookback_seconds:
        for i, t in enumerate(timestamps):
          last_index = np.searchsorted(times_array, t, side='left')
          first_index = np.searchsorted(times_array, t - lookback, side='left')
          time_slice = slice(first_index, last_index)
          longitudes = subcatalog.longitude.values[time_slice]
          latitudes = subcatalog.latitude.values[time_slice]
          features[i, 0, 0, :, :, index] = np.histogram2d(
              longitudes, latitudes, bins=(lng_bins, lat_bins)
          )[0]
        index += 1

    return features

  @gin.configurable('SeismicityGridEncoder.build_model')
  def build_model(
      self,
      filters,
      kernels,
      max_pool_kernels,
      activation = 'relu',
      kernel_regularization = None,
  ):
    """Builds a keras model to encode the features."""
    input_layer = tf.keras.layers.Input(
        shape=(*self.grid_size, self.n_features),
        name=self.name,
        dtype='float64',
    )

    output = input_layer
    for i, units in enumerate(filters):
      kernel, max_pool = kernels[i], max_pool_kernels[i]
      output = tf.keras.layers.Conv2D(
          units,
          kernel,
          kernel_regularizer=kernel_regularization,
          name=f'{self.name}_{i}_cnn',
          activation=activation,
      )(output)
      output = tf.keras.layers.MaxPool2D(max_pool)(output)

    output = tf.keras.layers.Flatten()(output)

    return tf.keras.models.Model(inputs=input_layer, outputs=output)


@gin.configurable
class SeismicityRateEncoder(CatalogEncoder):
  """An encoder for specifying the rate of earthquake over past periods."""

  def __init__(self, catalog):
    super().__init__(
        name='seismicity_rate', is_location_dependent=True, catalog=catalog
    )
    self.scaling_axes = (0, 1, 2)
    self.n_features = None

  @gin.configurable('SeismicityRateEncoder.prepare_features')
  def build_features(
      self,
      examples,
      train = False,
      custom_catalog = None,
      *,
      grid_side_deg = gin.REQUIRED,
      lookback_seconds = gin.REQUIRED,
      magnitudes = gin.REQUIRED,
  ):
    """Builds features for a set of examples.

    Args:
      examples: A mapping from times to locations.
      train: If this is the training phase, may fit some internal values. Unused
        in this case.
      custom_catalog: If present, will use this catalog instead of the stored
        one.
      grid_side_deg: The size (in degrees) of every grid cell.
      lookback_seconds: Time intervals (in the past from the evaluation times)
        for calculation of seismicity rates.
      magnitudes: Magnitude thresholds to calculate seismicity.

    Returns:
      A 5-dimensional array, that holds at index [i,j,k,l,m] the average rate of
      earthquakes (per day) in the (j, k)-th cell, above magnitude
      `magnitudes[l]`, in the `lookback_seconds[m]` seconds before the i-th
      example time.
    """
    catalog = self._catalog if custom_catalog is None else custom_catalog
    time = catalog.time.values
    assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"
    self.n_features = len(magnitudes)
    self.build_features_kwargs['grid_side_deg'] = grid_side_deg
    self.build_features_kwargs['lookback_seconds'] = lookback_seconds
    self.build_features_kwargs['magnitudes'] = magnitudes

    magnitudes = sorted(magnitudes, reverse=True)
    lookback_seconds = np.array(sorted(lookback_seconds, reverse=True)).astype(
        'float64'
    )
    lookback_deltas = np.concatenate(
        [lookback_seconds[:-1] - lookback_seconds[1:], [lookback_seconds[-1]]]
    )
    calculated_feature = catalog_analysis.compute_property_in_time_and_space(
        catalog,
        catalog_analysis.energy_in_square,
        examples,
        grid_side_deg,
        lookback_seconds,
        magnitudes,
    )
    # energy_in_square returns one scalar so the last dimension is redundent.
    features = calculated_feature[:, :, :, :, :, 0]
    features[:, :, :, :-1, :] -= features[:, :, :, 1:, :]
    features[:, :, :, :, 1:] -= features[:, :, :, :, :-1]
    for lookback_i, lookback_delta in enumerate(lookback_deltas):
      features[:, :, :, lookback_i, :] /= lookback_delta
    return features

  @gin.configurable(
      'SeismicityRateEncoder.build_model', denylist=['input_shape']
  )
  def build_model(self, **kwargs):
    """Builds a keras model to encode the features."""
    return architectures.rnn_model(self.name, self.n_features, **kwargs)
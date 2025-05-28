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

"""Training examples for different forecasting models.

The unifying logic for these models is that we have a set of interesting times,
and for every time we have a list of evaluation locations (may be on a regular
grid, or a single location per time).
For example, when forecasting the magnitude of the next earthquake, the examples
are the times immediately before target earthquakes, and the locations are the
locations of the target earthquakes. When forecasting the expected number of
earthquakes per grid cell for tomorrow, the times are the start of each day, and
the locations are the centers of the grid cells.

This module also contains methods to construct labels for the different models.
"""

import abc
import dataclasses
import functools
import hashlib
import json
from typing import Any, List, Mapping, Optional, Sequence

import gin
import numpy as np
import pandas as pd

from eq_mag_prediction.forecasting import data_sources
from eq_mag_prediction.utilities import catalog_analysis
from eq_mag_prediction.utilities import catalog_filters
from eq_mag_prediction.utilities import geometry


# A mapping between times (in seconds since Epoch) to a grid of locations (in
# longitude-latitude coordinates). Normally, features are calculated up to the
# timestamp (excluding it), and labels are calculated at the timestamp (or for a
# window that starts with the timestamp).
Examples = Mapping[float, List[List[geometry.Point]]]


@dataclasses.dataclass
class Domain(abc.ABC):
  """A wrapper for training, validation and test examples."""

  @property
  @abc.abstractmethod
  def train_examples(self):
    """Examples in the training set."""

  @property
  @abc.abstractmethod
  def validation_examples(self):
    """Examples in the validation set."""

  @property
  @abc.abstractmethod
  def test_examples(self):
    """Examples in the test set."""

  def domain_examples_uuid(self):
    """Create a unique id str for examples in a domain."""
    domain_examples_id = ''
    for set_name in ['train', 'validation', 'test']:
      examples_with_str = {
          k: self._list_of_points_to_str(v)
          for k, v in getattr(self, f'{set_name}_examples').items()
      }
      encoded_examples_json = json.dumps(
          examples_with_str, sort_keys=True
      ).encode('utf-8')
      examples_identifier = (
          f'{set_name}_{hashlib.sha1(encoded_examples_json).hexdigest()}'
      )
      domain_examples_id += examples_identifier + '_'
    return hashlib.sha1(domain_examples_id[:-1].encode('utf-8')).hexdigest()

  def _list_of_points_to_str(self, examples_value):
    """Recursively converts a nested list of geometry.Point objects to a nested list with the same structure of their str values."""
    new_list = []
    for ex in examples_value:
      if isinstance(ex, geometry.Point):
        new_list.append(str(ex))
      elif isinstance(ex, list):
        new_list.append(self._list_of_points_to_str(ex))
      else:
        new_list.append(ex)
    return new_list


@gin.configurable
@dataclasses.dataclass
class GriddedDomain(Domain, abc.ABC):
  """A domain with a regular grid of times and locations."""

  rectangle: geometry.Rectangle
  grid_side_degrees: float
  time_window_size_seconds: float

  @functools.cached_property
  def cell_centers(self):
    """Evaluation locations on a regular grid."""
    return geometry.Rectangles.init_grid(
        (self.rectangle.min_lng, self.rectangle.max_lng),
        (self.rectangle.min_lat, self.rectangle.max_lat),
        self.grid_side_degrees,
    ).to_centers()

  @functools.cached_property
  def shape(self):
    """Returns the shape of the grid, as a numpy array."""
    return np.array(self.cell_centers).shape

  @functools.cached_property
  @abc.abstractmethod
  def all_times(self):
    """Returns all of the times of the examples - train, validation and test."""


@gin.configurable
@dataclasses.dataclass
class RegularDomain(GriddedDomain):
  """A gridded domain domain with continuous times chunks."""

  train_start_time: float
  validation_start_time: float
  test_start_time: float
  test_end_time: float

  @functools.cached_property
  def all_times(self):
    """Returns all of the times of the examples - train, validation and test."""
    return np.arange(
        self.train_start_time, self.test_end_time, self.time_window_size_seconds
    )

  @functools.cached_property
  def train_examples(self):
    """Examples in the training set."""
    times = np.arange(
        self.train_start_time,
        self.validation_start_time,
        self.time_window_size_seconds,
    )
    return {t: self.cell_centers for t in times}

  @functools.cached_property
  def validation_examples(self):
    """Examples in the validation set."""
    times = np.arange(
        self.validation_start_time,
        self.test_start_time,
        self.time_window_size_seconds,
    )
    return {t: self.cell_centers for t in times}

  @functools.cached_property
  def test_examples(self):
    """Examples in the test set."""
    times = np.arange(
        self.test_start_time, self.test_end_time, self.time_window_size_seconds
    )
    return {t: self.cell_centers for t in times}


@gin.configurable
class CatalogDomain(Domain):
  """A domain containing relevant time-space locations."""

  def __init__(
      self,
      train_start_time,
      validation_start_time,
      test_start_time,
      test_end_time,
      test_times = None,
      test_locations = None,
      earthquakes_catalog = None,
      user_magnitude_threshold = None,
  ):
    """Initiates a CatalogDomain instance.

    Args:
      train_start_time: First timestamp included in the train set.
      validation_start_time:  First timestamp included in the validation set.
        Defines the first time to be excluded from the train set.
      test_start_time: First timestamp included in the test set. Defines the
        first time to be excluded from the validation set.
      test_end_time: The first time to be excluded from the test set.
      test_times: Explicit definition of test times. If given overrides
        test_start_time and test_end_time.
      test_locations: Explicit definition of test locations. If given overrides
        extraction of test-locations from catalog. Should be sequence of pairs
        of [lon, lat] (e.g. 2d array of coordinates).
      earthquakes_catalog: A pd.DataFrame of an earthquake catalog. If not given
        will use data_sources.target_catalog with no lines excluded.
      user_magnitude_threshold: If given, overrides the completeness magnitude
        that will otherwise be calculated on the train set.
    """
    super().__init__()
    self.train_start_time = train_start_time
    self.validation_start_time = validation_start_time
    self.test_start_time = test_start_time
    self.test_end_time = test_end_time
    self.test_times = test_times
    if test_locations is not None:
      test_locations_array = np.array(test_locations).squeeze()
      correct_dims = (test_locations_array.ndim == 2) & (
          test_locations_array.shape[1] == 2
      )
      assert correct_dims, 'test_locations should be shaped Nx2'
      self.test_locations = test_locations_array
    else:
      self.test_locations = None
    self.user_magnitude_threshold = user_magnitude_threshold
    if earthquakes_catalog is None:
      target_earthquakes = data_sources.target_catalog()
    else:
      target_earthquakes = earthquakes_catalog
    self.earthquakes_catalog = target_earthquakes

  @functools.cached_property
  def train_examples(self):
    """Examples in the training set."""
    return self._crop_and_create_examples(
        self.train_start_time,
        self.validation_start_time,
        self.event_times,
        self._event_locations,
    )

  @functools.cached_property
  def validation_examples(self):
    """Examples in the validation set."""
    return self._crop_and_create_examples(
        self.validation_start_time,
        self.test_start_time,
        self.event_times,
        self._event_locations,
    )

  @functools.cached_property
  def test_examples(self):
    """Examples in the test set.

    If entered explicitly will use the input, otherwise will use the relevant
    times from the event_times/_locations
    """
    if self.test_times is None:
      test_times = self.event_times
    else:
      test_times = self.test_times
    if self.test_locations is None:
      test_locations = self._event_locations
    else:
      test_locations = self.test_locations

    return self._crop_and_create_examples(
        self.test_start_time, self.test_end_time, test_times, test_locations
    )

  @functools.cached_property
  def event_times(self):
    """Returns the timestamps of all events above the magnitude threshold."""
    all_catalog_timestamps = self.earthquakes_catalog['time'].values
    return all_catalog_timestamps[self._magnitude_logical]

  @functools.cached_property
  def _event_locations(self):
    all_catalog_locations = self.earthquakes_catalog[
        ['longitude', 'latitude']
    ].values
    return all_catalog_locations[self._magnitude_logical]

  @functools.cached_property
  def event_magnitudes(self):
    """Returns the magnitudes of all events above the magnitude threshold."""
    all_magnitudes = self.earthquakes_catalog['magnitude'].values
    return all_magnitudes[self._magnitude_logical]

  @functools.cached_property
  def _magnitude_logical(self):
    all_magnitudes = self.earthquakes_catalog['magnitude'].values
    return all_magnitudes >= self.magnitude_threshold

  @functools.cached_property
  def magnitude_threshold(self):
    """Returns a magnitude threshold to be used as a cutoff for the catalog.

    If given, the user input self.user_magnitude_threshold will be used,
    otherwise the completeness magnitude of the train set will be used.
    """
    if self.user_magnitude_threshold is not None:
      return self.user_magnitude_threshold
    else:
      all_times_vector = self.earthquakes_catalog['time'].values
      train_times_logical = (all_times_vector >= self.train_start_time) & (
          all_times_vector < self.validation_start_time
      )
      train_time_magnitudes = self.earthquakes_catalog['magnitude'].values[
          train_times_logical
      ]
      return catalog_analysis.estimate_completeness(train_time_magnitudes)

  def _crop_and_create_examples(
      self, start_time, end_time, times, locations
  ):
    """Creates examples from times and locations appearing in the catalog."""
    event_times_array = np.array(times)
    times_logical = (event_times_array >= start_time) & (
        event_times_array < end_time
    )
    times_filtered = event_times_array[times_logical]
    locations_filtered = np.array(locations)[times_logical]
    to_points = lambda lon, lat: [[geometry.Point(lon, lat)]]
    points = list(
        map(to_points, locations_filtered[:, 0], locations_filtered[:, 1])
    )
    return {t: p for t, p in zip(times_filtered, points)}


@dataclasses.dataclass(frozen=True)
class Labels:
  """A container of labels. The expected shape is (n_times, *locations_grid)."""

  train_labels: np.ndarray
  validation_labels: np.ndarray
  test_labels: np.ndarray

  @property
  def flat_train_labels(self):
    """Train labels per pixel instead of per grid."""
    return self.train_labels.reshape((-1, 1))

  @property
  def flat_validation_labels(self):
    """Train labels per pixel instead of per grid."""
    return self.validation_labels.reshape((-1, 1))

  @property
  def flat_test_labels(self):
    """Train labels per pixel instead of per grid."""
    return self.test_labels.reshape((-1, 1))


@gin.configurable
def magnitude_prediction_labels(catalog_domain):
  """Returns the magnitude labels for a magnitude prediction task."""

  def _crop_mags_by_times(start_time, end_time):
    """Returns magnitudes of the events in the relevant time window."""
    times_logical = (catalog_domain.event_times >= start_time) & (
        catalog_domain.event_times < end_time
    )
    return catalog_domain.event_magnitudes[times_logical]

  return Labels(
      _crop_mags_by_times(
          catalog_domain.train_start_time, catalog_domain.validation_start_time
      ),
      _crop_mags_by_times(
          catalog_domain.validation_start_time, catalog_domain.test_start_time
      ),
      _crop_mags_by_times(
          catalog_domain.test_start_time, catalog_domain.test_end_time
      ),
  )

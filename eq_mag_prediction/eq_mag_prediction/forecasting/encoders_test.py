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

"""Tests for encoders."""

import tempfile

import gin
import numpy as np
import pandas as pd

from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import catalog_filters
from eq_mag_prediction.utilities import data_utils
# pylint: disable=unused-import - Required to parse the Gin config properly
from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.utilities import test_utils
# pylint: enable=unused-import
import os
from absl.testing import absltest
from absl.testing import parameterized

_SECONDS_IN_DAY = 86400


class RecentEarthquakesEncoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    with open(os.path.join(os.path.dirname(__file__), 'configs/test.gin')) as f:
      with gin.unlock_config():
        gin.parse_config(f.read(), skip_unknown=True)

    self.domain = training_examples.RegularDomain()
    self.magnitude_threshold = 4
    self.encoder = encoders.RecentEarthquakesEncoder(
        magnitude_threshold=self.magnitude_threshold
    )
    self.limit_lookback_seconds = _SECONDS_IN_DAY * 10000
    self.max_earthquakes = 2
    self.catalog = self.encoder._catalog

  def test_build_features(self):
    features = self.encoder.build_features(
        self.domain.train_examples,
        limit_lookback_seconds=self.limit_lookback_seconds,
        max_earthquakes=self.max_earthquakes,
    )

    self.assertEqual(
        features.shape,
        (
            len(self.domain.train_examples),
            self.max_earthquakes,
            len(self.encoder.feature_functions) + 1,
        ),
    )

    for i, timestamp in enumerate(sorted(self.domain.train_examples)):
      # There are 2922 training times, we check every 100th to save time.
      if i % 100 != 0:
        continue
      past_earthquakes = self.encoder._catalog[
          self.encoder._catalog.time < timestamp
      ]
      past_earthquakes = past_earthquakes[
          past_earthquakes.magnitude >= self.magnitude_threshold
      ]
      past_earthquakes = past_earthquakes.sort_values('time', ascending=True)
      past_earthquakes = past_earthquakes.iloc[-self.max_earthquakes :]
      past_xs, past_ys = data_utils.PROJECTIONS['japan'](
          past_earthquakes.longitude.values, past_earthquakes.latitude.values
      )

      # Test that we do not include the current event.
      self.assertTrue(np.all(features[i, :, 0] > 0))

      np.testing.assert_allclose(
          features[i, :, 2], timestamp - past_earthquakes.time.values
      )
      np.testing.assert_allclose(
          features[i, :, 5], np.exp(past_earthquakes.magnitude.values)
      )
      np.testing.assert_allclose(
          features[i, :, 1], np.log(1e-3 + past_earthquakes.depth.values)
      )
      np.testing.assert_allclose(features[i, :, 7], past_xs)
      np.testing.assert_allclose(features[i, :, 8], past_ys)

      # Assert that all feature earthquakes are before or at the evaluation
      # timestamp.
      self.assertTrue(np.all(features[i, :, 2] > 0))

  def test_excludes_current_time(self):
    for _ in range(100):
      i = np.random.randint(len(self.catalog))

      # Duplicate the row, not changing the time.
      catalog = self.catalog.copy()
      row = catalog.iloc[i]
      new_row = row.copy()
      new_row['longitude'] += np.random.randn()
      new_row['latitude'] += np.random.randn()
      new_row['depth'] += np.random.randn()
      new_row['magnitude'] += np.random.randn()
      duplicate_time = new_row.time
      catalog = pd.concat([catalog, new_row.to_frame().T])
      catalog = catalog.sort_values('time', ascending=True)
      new_encoder = encoders.RecentEarthquakesEncoder(
          catalog, magnitude_threshold=self.magnitude_threshold
      )
      examples = {
          duplicate_time: next(iter(self.domain.train_examples.values()))
      }

      old_features = self.encoder.build_features(
          examples,
          limit_lookback_seconds=self.limit_lookback_seconds,
          max_earthquakes=self.max_earthquakes,
      )
      new_features = new_encoder.build_features(
          examples,
          limit_lookback_seconds=self.limit_lookback_seconds,
          max_earthquakes=self.max_earthquakes,
      )

      np.testing.assert_allclose(old_features, new_features)

  @parameterized.named_parameters(
      dict(testcase_name='no_gridding', total_pixels=0, total_regions=0),
      dict(testcase_name='with_gridding', total_pixels=None, total_regions=1),
  )
  def test_and_build_model(self, total_pixels, total_regions):
    with gin.unlock_config():
      gin.bind_parameter('fully_connected_model.layer_sizes', (5,))

    features = self.encoder.build_features(
        self.domain.train_examples,
        limit_lookback_seconds=self.limit_lookback_seconds,
        max_earthquakes=self.max_earthquakes,
    )
    features = self.encoder.flatten_features(features)
    features = self.encoder.expand_features(features, self.domain.shape)
    if total_pixels is None:
      total_pixels = self.domain.shape[0] * self.domain.shape[1]
    grid_features = self.encoder.build_location_features(
        self.domain.train_examples,
        total_pixels=total_pixels,
        first_pixel_index=0,
        total_regions=total_regions,
        region_index=0,
    )
    grid_features = self.encoder.flatten_location_features(grid_features)
    units = (16, 8, 4)

    expected_shape = (
        len(self.domain.train_examples)
        * self.domain.shape[0]
        * self.domain.shape[1],
        units[-1],
    )
    model = self.encoder.build_model(units=units)

    self.assertEqual(
        model.predict([features, grid_features]).shape, expected_shape
    )


class BiggestEarthquakesEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    with open(os.path.join(os.path.dirname(__file__), 'configs/test.gin')) as f:
      with gin.unlock_config():
        gin.parse_config(f.read(), skip_unknown=True)

    self.domain = training_examples.RegularDomain()
    self.encoder = encoders.BiggestEarthquakesEncoder()
    # Remove the focal mechanism feature functions - they don't appear in the
    # sample catalog.
    self.encoder.FEATURE_FUNCTIONS = self.encoder.FEATURE_FUNCTIONS[:-5]
    self.encoder.n_features -= 5
    self.limit_lookback_seconds = _SECONDS_IN_DAY * 10000
    self.max_earthquakes = 2
    self.catalog = self.encoder._catalog

  def test_build_features(self):
    features = self.encoder.build_features(
        self.domain.train_examples,
        limit_lookback_seconds=self.limit_lookback_seconds,
        max_earthquakes=self.max_earthquakes,
    )

    self.assertEqual(
        features.shape,
        (
            len(self.domain.train_examples),
            *self.domain.shape,
            self.max_earthquakes,
            len(self.encoder.FEATURE_FUNCTIONS) + 1,
        ),
    )

    for i, timestamp in enumerate(sorted(self.domain.train_examples)):
      # There are 2922 training times, we check every 100th to save time.
      if i % 100 != 0:
        continue
      past_earthquakes = self.encoder._catalog[
          self.encoder._catalog.time < timestamp
      ]
      past_earthquakes = past_earthquakes.sort_values(
          'magnitude', ascending=True
      )
      past_earthquakes = past_earthquakes.iloc[-self.max_earthquakes :]
      past_xs, past_ys = data_utils.PROJECTIONS['japan'](
          past_earthquakes.longitude.values, past_earthquakes.latitude.values
      )

      for row in range(self.domain.shape[0]):
        for col in range(self.domain.shape[1]):
          point = self.domain.train_examples[timestamp][row][col]
          np.testing.assert_allclose(
              features[i, row, col, :, 0],
              timestamp - past_earthquakes.time.values,
          )
          np.testing.assert_allclose(
              features[i, row, col, :, 3],
              np.exp(past_earthquakes.magnitude.values),
          )
          np.testing.assert_allclose(
              features[i, row, col, :, 6],
              np.log(1e-3 + past_earthquakes.depth.values),
          )
          x, y = data_utils.PROJECTIONS['japan'](point.lng, point.lat)
          np.testing.assert_allclose(
              features[i, row, col, :, 9],
              np.sqrt((past_xs - x) ** 2 + (past_ys - y) ** 2),
          )

      # Assert that all feature earthquakes are before or at the evaluation
      # timestamp.
      self.assertTrue(np.all(features[i, :, :, :, 0] > 0))

  def test_and_build_model(self):
    features = self.encoder.build_features(
        self.domain.train_examples,
        limit_lookback_seconds=self.limit_lookback_seconds,
        max_earthquakes=self.max_earthquakes,
    )
    features = self.encoder.flatten_features(features)
    units = (16, 8, 4)

    expected_shape = (
        len(self.domain.train_examples)
        * self.domain.shape[0]
        * self.domain.shape[1],
        units[-1],
    )
    model = self.encoder.build_model(units=units)

    self.assertEqual(model.predict(features).shape, expected_shape)


class SeismicityRateEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    with open(os.path.join(os.path.dirname(__file__), 'configs/test.gin')) as f:
      with gin.unlock_config():
        gin.parse_config(f.read(), skip_unknown=True)

    self.domain = training_examples.RegularDomain()
    self.encoder = encoders.SeismicityRateEncoder()
    self.magnitudes = (6, 5.5, 5)
    self.lookback_seconds = (_SECONDS_IN_DAY * 365 * 2, _SECONDS_IN_DAY * 365)
    self.catalog = self.encoder._catalog

  def test_build_features(self):
    features = self.encoder.build_features(
        self.domain.train_examples,
        grid_side_deg=self.domain.grid_side_degrees,
        lookback_seconds=self.lookback_seconds,
        magnitudes=self.magnitudes,
    )

    self.assertEqual(
        features.shape,
        (
            len(self.domain.train_examples),
            *self.domain.shape,
            len(self.lookback_seconds),
            len(self.magnitudes),
        ),
    )

    for i, timestamp in enumerate(sorted(self.domain.train_examples)):
      # There are 2922 training times, we check every 100th to save time.
      if i % 100 != 0:
        continue
      for j, row in enumerate(self.domain.train_examples[timestamp]):
        for k, center in enumerate(row):
          subcatalog = catalog_filters.limit_catalog_to_square(
              self.catalog, center, self.domain.grid_side_degrees
          )

          for l, lookback in enumerate(self.lookback_seconds):
            lookback_range = [lookback]
            if l == len(self.lookback_seconds) - 1:
              lookback_range.append(0)
            else:
              lookback_range.append(self.lookback_seconds[l + 1])

            for m, magnitude in enumerate(self.magnitudes):
              magnitude_range = [magnitude]
              if m == 0:
                magnitude_range.append(100)
              else:
                magnitude_range.append(self.magnitudes[m - 1])

              in_magnitude_time_bin = subcatalog[
                  (subcatalog.magnitude >= magnitude_range[0])
                  & (subcatalog.magnitude < magnitude_range[1])
                  & (subcatalog.time >= timestamp - lookback_range[0])
                  & (subcatalog.time < timestamp - lookback_range[1])
              ]

              self.assertAlmostEqual(
                  features[i, j, k, l, m]
                  * (lookback_range[0] - lookback_range[1]),
                  np.sum(np.exp(in_magnitude_time_bin.magnitude)),
              )

  def test_and_build_model(self):
    features = self.encoder.build_features(
        self.domain.train_examples,
        grid_side_deg=self.domain.grid_side_degrees,
        lookback_seconds=self.lookback_seconds,
        magnitudes=self.magnitudes,
    )
    features = self.encoder.flatten_features(features)
    units = (20, 10, 90)

    expected_shape = (
        len(self.domain.train_examples)
        * self.domain.shape[0]
        * self.domain.shape[1],
        units[-1],
    )
    model = self.encoder.build_model(units=units)

    self.assertEqual(model.predict(features).shape, expected_shape)


class CatalogColumnsEncoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    n_events_in_catalog = int(1e3)
    self.train_start_time = 0
    self.validation_start_time = 700
    self.test_start_time = 800
    self.test_end_time = n_events_in_catalog
    spatial_vector = np.linspace(-50, 50, n_events_in_catalog)
    self.input_catalog, _ = _random_catalog(
        dataframe_len=n_events_in_catalog, spatial_vector=spatial_vector
    )

  def test_build_features_examples_from_catalog(self):
    domain = training_examples.CatalogDomain(
        self.train_start_time,
        self.validation_start_time,
        self.test_start_time,
        self.test_end_time,
        earthquakes_catalog=self.input_catalog,
        user_magnitude_threshold=5,
    )
    additional_columns = ['depth']
    encoder = encoders.CatalogColumnsEncoder(domain.earthquakes_catalog)

    max_time = 10
    time_logical = self.input_catalog['time'].values < max_time
    example_times = self.input_catalog['time'].values[time_logical]
    cat = self.input_catalog
    lon = lambda t: cat.longitude[cat.time == t].values[0]
    lat = lambda t: cat.latitude[cat.time == t].values[0]
    test_examples = {
        t: [[geometry.Point(lon(t), lat(t))]] for t in example_times
    }
    resulting_features = encoder.build_features(
        test_examples, additional_columns=additional_columns
    )

    expected_time_diff = np.concatenate(([0], np.diff(example_times)))[:, None]
    expected_locations = np.array(
        [[v[0][0].lng, v[0][0].lat] for v in test_examples.values()]
    )
    additional_column = self.input_catalog['depth'].values[time_logical][
        :, None
    ]
    expected_features = np.concatenate(
        (expected_time_diff, expected_locations, additional_column), axis=1
    )
    np.testing.assert_array_equal(expected_features, resulting_features)

  @parameterized.named_parameters(
      dict(testcase_name='time_shifted', shift_time=True, shift_location=False),
      dict(
          testcase_name='location_shifted',
          shift_time=False,
          shift_location=True,
      ),
      dict(
          testcase_name='time_and_location_shifted',
          shift_time=True,
          shift_location=True,
      ),
  )
  def test_build_features_examples_not_in_catalog(
      self, shift_time, shift_location
  ):
    rnd_seed = np.random.RandomState(seed=1905)
    max_time = 10
    time_logical = self.input_catalog['time'].values < max_time
    example_times = self.input_catalog['time'].values[time_logical]
    time_shifts = np.zeros_like(example_times)
    if shift_time:
      time_shifts = np.linspace(0.1, 0.8, len(example_times))
    example_times += time_shifts

    examples_locations_from_cat = self.input_catalog[
        ['longitude', 'latitude']
    ].values[time_logical]
    location_shifts = np.zeros_like(examples_locations_from_cat)
    if shift_location:
      location_shifts = (
          rnd_seed.uniform(examples_locations_from_cat.shape) * 2e-3 - 1e-3
      )
    examples_locations = examples_locations_from_cat + location_shifts

    domain = training_examples.CatalogDomain(
        self.train_start_time,
        self.validation_start_time,
        self.test_start_time,
        self.test_end_time,
        test_times=example_times,
        test_locations=examples_locations,
        earthquakes_catalog=self.input_catalog,
        user_magnitude_threshold=5,
    )
    additional_columns = ['depth']
    encoder = encoders.CatalogColumnsEncoder(domain.earthquakes_catalog)

    test_examples = {
        t: [[geometry.Point(loc[0], loc[1])]]
        for t, loc in zip(example_times, examples_locations)
    }
    resulting_features = encoder.build_features(
        test_examples, additional_columns=additional_columns
    )

    if shift_time:
      expected_time_diff = time_shifts[:, None]
    else:
      expected_time_diff = np.concatenate(([0], np.diff(example_times)))[
          :, None
      ]
    expected_locations = examples_locations
    additional_column = self.input_catalog['depth'].values[time_logical][
        :, None
    ]
    expected_features = np.concatenate(
        (expected_time_diff, expected_locations, additional_column), axis=1
    )
    np.testing.assert_array_almost_equal(expected_features, resulting_features)


def _random_catalog(dataframe_len=10, spatial_vector=None):
  rnd_seed = np.random.RandomState(seed=1905)
  time_vec = np.arange(dataframe_len) - (
      rnd_seed.uniform(size=(dataframe_len)) * 0.1 + 1e-4
  )
  if spatial_vector is None:
    spatial_vector = np.ones(dataframe_len)
  catalog_simulation = pd.DataFrame({
      'latitude': spatial_vector * rnd_seed.uniform(size=(dataframe_len)),
      'longitude': spatial_vector * rnd_seed.uniform(size=(dataframe_len)),
      'depth': rnd_seed.uniform(size=(dataframe_len)),
      'magnitude': 8 + rnd_seed.uniform(size=(dataframe_len)),
      'time': time_vec,
      'a': rnd_seed.uniform(size=(dataframe_len)),
      'b': rnd_seed.uniform(size=(dataframe_len)),
  })

  examples = {
      x.time: [[geometry.Point(x.longitude, x.latitude)]]
      for _, x in catalog_simulation.iterrows()
  }

  return catalog_simulation, examples


if __name__ == '__main__':
  absltest.main()

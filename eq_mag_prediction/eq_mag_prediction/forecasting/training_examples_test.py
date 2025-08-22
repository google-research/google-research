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

"""Tests for training_examples."""

import gin
import numpy as np
import pandas as pd

from eq_mag_prediction.forecasting import data_sources
# pylint: disable=unused-import - Required to parse the Gin config properly
from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import catalog_filters
from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.utilities import test_utils
# pylint: enable=unused-import
import os
from absl.testing import absltest
from absl.testing import parameterized

# The same time edges that are used in the config.
TRAIN_START_TIME = 1104505200
VALIDATION_START_TIME = 1167577200
TEST_START_TIME = 1230735600
TEST_END_TIME = 1293807600


class CatalogDomainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='all_events_counted',
          default_mag=5,
          alternate_mag=2,
          alternate_index=0,
          magnitude_threshold=3,
      ),
      dict(
          testcase_name='first_7_events_below_threshold',
          default_mag=5,
          alternate_mag=2,
          alternate_index=7,
          magnitude_threshold=3,
      ),
  )
  def test_examples(
      self, default_mag, alternate_mag, alternate_index, magnitude_threshold
  ):
    fake_catalog = self._fake_catalog(
        default_mag, alternate_mag, alternate_index
    )
    with gin.unlock_config():
      gin.bind_parameter('target_catalog.catalog', fake_catalog)
      gin.bind_parameter(
          'target_catalog.earthquake_criterion',
          catalog_filters.return_entire_catalog_criterion,
      )
    train_start_time = 0
    validation_start_time = 10
    test_start_time = 20
    test_end_time = 50
    catalog_domain = training_examples.CatalogDomain(
        train_start_time=train_start_time,
        validation_start_time=validation_start_time,
        test_start_time=test_start_time,
        test_end_time=test_end_time,
        user_magnitude_threshold=magnitude_threshold,
    )

    train_examples_result = catalog_domain.train_examples
    validation_examples_result = catalog_domain.validation_examples
    test_examples_result = catalog_domain.test_examples

    time_ranges = (
        (train_start_time + alternate_index, validation_start_time),
        (validation_start_time, test_start_time),
        (test_start_time, test_end_time),
    )
    lon_ranges = (
        (train_start_time + alternate_index + 140, validation_start_time + 140),
        (validation_start_time + 140, test_start_time + 140),
        (test_start_time + 140, test_end_time + 140),
    )
    lat_ranges = (
        (train_start_time + alternate_index + 30, validation_start_time + 30),
        (validation_start_time + 30, test_start_time + 30),
        (test_start_time + 30, test_end_time + 30),
    )

    for i, exmp in enumerate((
        train_examples_result,
        validation_examples_result,
        test_examples_result,
    )):
      self.assertEqual(list(exmp.keys()), list(range(*time_ranges[i])))
      expected_locations = [
          [[geometry.Point(*lon_lat_pair)]]
          for lon_lat_pair in zip(range(*lon_ranges[i]), range(*lat_ranges[i]))
      ]
      self.assertEqual(list(exmp.values()), expected_locations)

  def _fake_catalog(self, default_mag=5, alternate_mag=2, alternate_index=0):
    rnd_seed = np.random.RandomState(seed=1905)
    dataframe_len = 100
    time_vec = np.arange(dataframe_len)
    longitude_vec = np.arange(dataframe_len) + 140
    latitude_vec = np.arange(dataframe_len) + 30
    magnitude_vector = default_mag * np.ones((dataframe_len,))
    magnitude_vector[:alternate_index] = alternate_mag
    catalog_simulation = pd.DataFrame({
        'longitude': longitude_vec,
        'latitude': latitude_vec,
        'depth': rnd_seed.uniform(size=(dataframe_len)),
        'magnitude': magnitude_vector,
        'time': time_vec,
        'a': rnd_seed.uniform(size=(dataframe_len)),
        'b': rnd_seed.uniform(size=(dataframe_len)),
    })
    return catalog_simulation


class CatalogLabelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('times_included_in_catalog', 0, 10, 20, 100),
      ('all_labels_included_in_catalog', -5, 10, 20, 105),
      ('times_not_in_catalog', -0.2, 8.2, 20.6, 55.5),
  )
  def test_magnitude_prediction_labels(
      self,
      train_start_time,
      validation_start_time,
      test_start_time,
      test_end_time,
  ):
    catalog_len = 100
    fake_catalog = self._fake_catalog(catalog_len=catalog_len)
    with gin.unlock_config():
      gin.bind_parameter('target_catalog.catalog', fake_catalog)
      gin.bind_parameter(
          'target_catalog.earthquake_criterion',
          catalog_filters.return_entire_catalog_criterion,
      )
    domain = training_examples.CatalogDomain(
        train_start_time,
        validation_start_time,
        test_start_time,
        test_end_time,
        user_magnitude_threshold=-4,
    )
    labels = training_examples.magnitude_prediction_labels(
        catalog_domain=domain
    )

    train_time_limits = (train_start_time, validation_start_time)
    val_time_limits = (validation_start_time, test_start_time)
    test_time_limits = (test_start_time, test_end_time)

    expected_train_labels = 2 * np.arange(
        np.maximum(0, np.ceil(train_time_limits[0])),
        np.minimum(catalog_len, train_time_limits[1]),
        1,
    )
    np.testing.assert_array_equal(labels.train_labels, expected_train_labels)

    expected_val_labels = 2 * np.arange(
        np.maximum(0, np.ceil(val_time_limits[0])),
        np.minimum(catalog_len, val_time_limits[1]),
        1,
    )
    np.testing.assert_array_equal(labels.validation_labels, expected_val_labels)

    expected_test_labels = 2 * np.arange(
        np.maximum(0, np.ceil(test_time_limits[0])),
        np.minimum(catalog_len, test_time_limits[1]),
        1,
    )
    np.testing.assert_array_equal(labels.test_labels, expected_test_labels)

  def _times_to_longitudes(self, times):
    return times**2 + 5

  def _times_to_latitudes(self, times):
    return times * 4 - 2

  def _fake_catalog(self, catalog_len=100):
    rnd_seed = np.random.RandomState(seed=1905)
    time_vec = np.arange(catalog_len)
    catalog_simulation = pd.DataFrame({
        'time': time_vec,
        'magnitude': 2 * time_vec,
        'a': rnd_seed.uniform(size=(catalog_len)),
        'b': rnd_seed.uniform(size=(catalog_len)),
        'depth': rnd_seed.uniform(size=(catalog_len)),
        'longitude': self._times_to_longitudes(time_vec),
        'latitude': self._times_to_latitudes(time_vec),
    })
    return catalog_simulation


if __name__ == '__main__':
  absltest.main()

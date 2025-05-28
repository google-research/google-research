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

import gin
import numpy as np
import pandas as pd
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.forecasting import data_sources
from eq_mag_prediction.utilities import catalog_filters


class DataSourcesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataframe_len = 200
    self.train_start_time = 0
    self.validation_start_time = 50
    self.test_start_time = 120
    self.test_end_time = 200
    with gin.unlock_config():
      gin.bind_parameter(
          'return_entire_catalog_criterion.catalog', self._fake_catalog()
      )
      gin.bind_parameter('target_catalog.catalog', self._fake_catalog())
      gin.bind_parameter(
          'target_catalog.earthquake_criterion',
          catalog_filters.return_entire_catalog_criterion,
      )

  @parameterized.named_parameters(
      dict(testcase_name='separate_permute_false', separate_permute=False),
      dict(testcase_name='separate_permute_true', separate_permute=True),
  )
  def test_permuted_catalog(self, separate_permute):
    original_catalog = data_sources.target_catalog()
    columns_to_permute = ['longitude', 'magnitude']
    permuted_catalog = data_sources.permuted_catalog(
        catalog=self._fake_catalog(),
        earthquake_criterion=catalog_filters.return_entire_catalog_criterion,
        columns_to_permute=columns_to_permute,
        separate_permute=separate_permute,
    )

    non_permuted_columns = [
        col for col in original_catalog.keys() if col not in columns_to_permute
    ]
    np.testing.assert_array_equal(
        permuted_catalog[non_permuted_columns].values,
        original_catalog[non_permuted_columns].values,
    )

    if separate_permute:
      for col in columns_to_permute:
        np.testing.assert_array_equal(
            np.sort(permuted_catalog[col].values),
            np.sort(original_catalog[col].values),
        )
        self.assertFalse(
            np.all(permuted_catalog[col].values == original_catalog[col].values)
        )
      self.assertFalse(
          np.all(
              permuted_catalog[columns_to_permute]
              .sort_values(columns_to_permute[0])
              .values
              == original_catalog[columns_to_permute]
              .sort_values(columns_to_permute[0])
              .values
          )
      )
    else:
      np.testing.assert_array_equal(
          permuted_catalog[columns_to_permute].sort_values(
              columns_to_permute[0]
          ),
          original_catalog[columns_to_permute].sort_values(
              columns_to_permute[0]
          ),
      )

  def test_multiple_permutation_criteria(self):
    original_catalog = data_sources.target_catalog()
    columns_to_permute = ['magnitude']
    non_permuted_columns = [
        col for col in original_catalog.keys() if col not in columns_to_permute
    ]
    non_mixed_logical = (
        original_catalog.time.values >= self.validation_start_time
    ) & (original_catalog.time.values < self.test_start_time)
    mixed_logical = ~non_mixed_logical
    permuted_catalog = data_sources.permuted_catalog(
        catalog=self._fake_catalog(),
        earthquake_criterion=catalog_filters.return_entire_catalog_criterion,
        columns_to_permute=columns_to_permute,
        permute_earthquake_criterion=[
            self.permutation_criterion_train,
            self.permutation_criterion_test,
        ],
    )
    np.testing.assert_array_equal(
        permuted_catalog[non_permuted_columns].values,
        original_catalog[non_permuted_columns].values,
    )
    np.testing.assert_array_equal(
        permuted_catalog.loc[non_mixed_logical, columns_to_permute].values,
        original_catalog.loc[non_mixed_logical, columns_to_permute].values,
    )
    self.assertFalse(
        np.all(
            permuted_catalog.loc[mixed_logical, columns_to_permute].values
            == original_catalog.loc[mixed_logical, columns_to_permute].values
        )
    )

  def _fake_catalog(self):
    rnd_seed = np.random.RandomState(seed=1905)
    time_vec = np.arange(self.dataframe_len)
    longitude_vec = np.arange(self.dataframe_len) + 140
    latitude_vec = np.arange(self.dataframe_len) + 30
    catalog_simulation = pd.DataFrame({
        'longitude': longitude_vec,
        'latitude': latitude_vec,
        'depth': rnd_seed.uniform(size=(self.dataframe_len)),
        'magnitude': rnd_seed.uniform(size=(self.dataframe_len)) * 8,
        'time': time_vec,
        'a': rnd_seed.uniform(size=(self.dataframe_len)),
        'b': rnd_seed.uniform(size=(self.dataframe_len)),
    })
    return catalog_simulation

  def permutation_criterion_train(self, df):
    return catalog_filters.earthquake_criterion(
        df,
        longitude_range=(-180, 180),
        latitude_range=(-180, 180),
        start_timestamp=self.train_start_time,
        end_timestamp=self.validation_start_time,
        max_depth=1000,
        min_magnitude=-10,
    )

  def permutation_criterion_test(self, df):
    return catalog_filters.earthquake_criterion(
        df,
        longitude_range=(-180, 180),
        latitude_range=(-180, 180),
        start_timestamp=self.test_start_time,
        end_timestamp=self.test_end_time,
        max_depth=1000,
        min_magnitude=-10,
    )


if __name__ == '__main__':
  absltest.main()

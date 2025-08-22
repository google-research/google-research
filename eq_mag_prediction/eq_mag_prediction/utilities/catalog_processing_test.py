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

"""Tests for utilities.catalog_processing."""

import numpy as np
import pandas as pd
from absl.testing import absltest
from eq_mag_prediction.utilities import catalog_processing

DAYS_TO_SECONDS = 24 * 60 * 60


class CatalogProcessingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.catalog = self._fake_catalog()
    self.rnd_seed = np.random.RandomState(seed=1905)
    self.n_modifications = 3
    self.catalog_noise = 1e-3
    self.catalog_modifications = zip(
        self.rnd_seed.choice(
            self.catalog.shape[0], self.n_modifications, replace=False
        ),
        self.rnd_seed.choice(
            self.catalog.shape[1], self.n_modifications, replace=False
        ),
    )

  def test_unique_name_for_calculation_constant_time(self):
    arguments_str_list = []
    for _ in range(self.n_modifications):
      estimate_times = self.rnd_seed.uniform(
          low=self.catalog.time.min(), high=self.catalog.time.max(), size=10
      )
      for cat_mod in self.catalog_modifications:
        modified_cat = self.catalog.copy()
        modified_cat.iloc[cat_mod[0], cat_mod[1]] += self.rnd_seed.uniform(
            low=-self.catalog_noise, high=self.catalog_noise
        )
        arguments_str = (
            catalog_processing._gr_moving_window_constant_time_vars_to_str(
                estimate_times=estimate_times,
                catalog=modified_cat,
                window_time=10 * DAYS_TO_SECONDS,
                m_minimal=-100,
                n_above_complete=1,
                weight_on_past=1,
                default_beta=np.nan,
                default_mc=None,
                completeness_calculator=None,
            )
        )
        arguments_str_list.append(arguments_str)
    arguments_str_set = set(arguments_str_list)
    self.assertEqual(len(arguments_str_set), len(arguments_str_list))

  def test_unique_name_for_calculation_n_events(self):
    arguments_str_list = []
    for _ in range(self.n_modifications):
      estimate_times = self.rnd_seed.uniform(
          low=self.catalog.time.min(), high=self.catalog.time.max(), size=10
      )
      for cat_mod in self.catalog_modifications:
        modified_cat = self.catalog.copy()
        modified_cat.iloc[cat_mod[0], cat_mod[1]] += self.rnd_seed.uniform(
            low=-self.catalog_noise, high=self.catalog_noise
        )
        arguments_str = (
            catalog_processing._gr_moving_window_n_events_vars_to_str(
                estimate_times=estimate_times,
                catalog=modified_cat,
                n_events=250,
                m_minimal=-100,
                n_above_complete=1,
                weight_on_past=1,
                completeness_and_beta_calculator=None,
            )
        )
        arguments_str_list.append(arguments_str)
    arguments_str_set = set(arguments_str_list)
    self.assertEqual(len(arguments_str_set), len(arguments_str_list))

  def test_unique_name_for_calculation_spatial_gr(self):
    arguments_str_list = []
    for _ in range(self.n_modifications):
      estimate_longitudes = self.rnd_seed.uniform(
          low=self.catalog.longitude.min(),
          high=self.catalog.longitude.max(),
          size=10,
      )
      estimate_latitudes = self.rnd_seed.uniform(
          low=self.catalog.latitude.min(),
          high=self.catalog.latitude.max(),
          size=10,
      )
      estimate_coors = list(zip(estimate_longitudes, estimate_latitudes))
      for cat_mod in self.catalog_modifications:
        for mc_calc_method in [None, 'MAXC', 'MBS']:
          for completeness_magnitude in [None, 2, 4]:
            if mc_calc_method is not None:
              continue
            modified_cat = self.catalog.copy()
            modified_cat.iloc[cat_mod[0], cat_mod[1]] += self.rnd_seed.uniform(
                low=-self.catalog_noise, high=self.catalog_noise
            )
            arguments_str = catalog_processing._gr_spatial_vars_to_str(
                estimate_coors=estimate_coors,
                catalog=modified_cat,
                completeness_magnitude=completeness_magnitude,
                mc_calc_method=mc_calc_method,
                grid_spacing=0.1,
                smoothing_distance=0.5,
                discard_few_event_locations=True,
            )
            arguments_str_list.append(arguments_str)
    arguments_str_set = set(arguments_str_list)
    self.assertEqual(len(arguments_str_set), len(arguments_str_list))

  def _fake_catalog(self, catalog_len=1000):
    rnd_seed = np.random.RandomState(seed=1905)
    time_vec = np.arange(catalog_len)
    catalog_simulation = pd.DataFrame({
        'time': time_vec,
        'magnitude': 2 * time_vec,
        'a': rnd_seed.uniform(size=(catalog_len)),
        'b': rnd_seed.uniform(size=(catalog_len)),
        'depth': rnd_seed.uniform(size=(catalog_len)),
        'longitude': rnd_seed.uniform(low=-10, high=10, size=(catalog_len)),
        'latitude': rnd_seed.uniform(low=-10, high=10, size=(catalog_len)),
        'longitude': rnd_seed.uniform(low=-10, high=10, size=(catalog_len)),
        'latitude': rnd_seed.uniform(low=-10, high=10, size=(catalog_len)),
    })
    return catalog_simulation


if __name__ == '__main__':
  absltest.main()

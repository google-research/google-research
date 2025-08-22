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

"""Tests for magnitude_predictor_trainer."""

import tempfile
import gin
import numpy as np
import pandas as pd

import os
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.forecasting import magnitude_predictor_trainer
from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.forecasting import training_examples


class MagnitudePredictorTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    catalog_size = int(1e3)
    self.catalog_sample = _random_catalog(catalog_size)
    n_events = self.catalog_sample.shape[0]
    train_start_time = self.catalog_sample['time'][0]
    validation_start_time = self.catalog_sample['time'][int(0.7 * n_events)]
    test_start_time = self.catalog_sample['time'][int(0.9 * n_events)]
    test_end_time = catalog_size - 1
    with gin.unlock_config():
      gin.bind_parameter('CatalogDomain.train_start_time', train_start_time)
      gin.bind_parameter(
          'CatalogDomain.validation_start_time', validation_start_time
      )
      gin.bind_parameter('CatalogDomain.test_start_time', test_start_time)
      gin.bind_parameter('CatalogDomain.test_end_time', test_end_time)
      gin.bind_parameter(
          'CatalogDomain.earthquakes_catalog', self.catalog_sample
      )

    with open(
        os.path.join(
            os.path.dirname(__file__),
            'configs/magnitude_prediction/magnitude_predictor_trainer_test.gin',
        )
    ) as f:
      with gin.unlock_config():
        gin.parse_config(f.read(), skip_unknown=True)
    self.test_tmp_dir = tempfile.mkdtemp()
    self.cache_tmp_dir = tempfile.mkdtemp()
    gin.bind_parameter(
        'load_features_and_construct_models.cache_dir', self.cache_tmp_dir
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='do_not_use_depth_as_feature',
          additional_columns=[],
      ),
      dict(testcase_name='use_depth_as_feature', additional_columns=['depth']),
  )
  def test_fit_magnitude_predictor_trainer(self, additional_columns):
    with gin.unlock_config():
      gin.bind_parameter(
          'CatalogColumnsEncoder.prepare_features.additional_columns',
          additional_columns,
      )

    domain = training_examples.CatalogDomain()
    all_encoders = one_region_model.build_encoders(domain)
    one_region_model.compute_and_cache_features_scaler_encoder(
        domain, all_encoders, self.cache_tmp_dir
    )

    epochs = 1
    learning_rate = 1e-4
    batch_size = 16
    history, _ = (
        magnitude_predictor_trainer.train_and_evaluate_magnitude_prediction_model(
            self.test_tmp_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
    )

    self.assertLen(history.history['loss'], epochs)
    self.assertLen(history.history['val_loss'], epochs)
    self.assertFalse(np.any(np.isnan(history.history['loss'])))
    self.assertFalse(np.any(np.isnan(history.history['val_loss'])))


@gin.configurable
def _random_catalog(dataframe_len=10, spatial_vector=None):
  rnd_seed = np.random.RandomState(seed=1905)
  time_vec = np.arange(dataframe_len)
  if spatial_vector is None:
    spatial_vector = np.ones(dataframe_len)

  catalog_simulation = pd.DataFrame({
      'time': time_vec,
      'longitude': spatial_vector * rnd_seed.uniform(size=(dataframe_len)),
      'latitude': spatial_vector * rnd_seed.uniform(size=(dataframe_len)),
      'depth': rnd_seed.uniform(size=(dataframe_len)),
      'magnitude': 8 * rnd_seed.uniform(size=(dataframe_len)),
      'strike': rnd_seed.uniform(size=(dataframe_len)),
      'rake': rnd_seed.uniform(size=(dataframe_len)),
      'dip': rnd_seed.uniform(size=(dataframe_len)),
      'a': rnd_seed.uniform(size=(dataframe_len)),
      'b': rnd_seed.uniform(size=(dataframe_len)),
  })

  return catalog_simulation.astype(float)


if __name__ == '__main__':
  absltest.main()

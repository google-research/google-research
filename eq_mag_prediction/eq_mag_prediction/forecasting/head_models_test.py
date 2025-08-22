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

"""Tests for head_models."""

import gin
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from absl.testing import absltest
from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import head_models
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import catalog_filters


class HeadModelsTest(absltest.TestCase):

  def test_input_order(self):
    gridded_model_names = ['seismicity_rate', 'background_rate']
    non_gridded_model_names = ['recent_earthquakes', 'gps', 'seismic_waves']

    self.assertEqual(
        head_models.input_order(
            spatially_dependent_model_names=gridded_model_names,
            spatially_independent_model_names=non_gridded_model_names,
        ),
        (
            ['background_rate', 'seismicity_rate'],
            ['gps', 'recent_earthquakes', 'seismic_waves'],
        ),
    )


class MagnitudePredictionModelTest(absltest.TestCase):

  def test_magnitude_prediction_model(self):
    with open(
        os.path.join(os.path.dirname(__file__), 'configs/head_models_test.gin')
    ) as f:
      with gin.unlock_config():
        gin.parse_config(f.read(), skip_unknown=True)
    domain = training_examples.CatalogDomain()
    examples = domain.train_examples
    n_events = len(examples)
    grid_side = 0.5
    recent_encoder_spatial_encoding_kwargs = {
        'total_pixels': 0,
        'first_pixel_index': 0,
        'total_regions': 0,
        'region_index': 0,
    }
    encoders_list, features_list, models_list = (
        _create_encoders_features_models(
            examples, grid_side, recent_encoder_spatial_encoding_kwargs
        )
    )

    (
        recent_encoder,
        long_term_seismicity_encoder,
        short_term_seismicity_encoder,
    ) = encoders_list
    columns_encoder = encoders.CatalogColumnsEncoder()

    (
        recent_features,
        recent_grid_features,
        long_term_seismicity_features,
        short_term_seismicity_features,
    ) = features_list
    columns_features = columns_encoder.build_features(examples)

    (recent_model, long_term_seismicity_model, short_term_seismicity_model) = (
        models_list
    )
    columns_model = columns_encoder.build_model()

    recent_features_flat = recent_encoder.flatten_features(recent_features)
    recent_grid_features_flat = recent_encoder.flatten_location_features(
        recent_grid_features
    )
    long_term_seismicity_features_flat = (
        long_term_seismicity_encoder.flatten_features(
            long_term_seismicity_features
        )
    )
    short_term_seismicity_features_flat = (
        short_term_seismicity_encoder.flatten_features(
            short_term_seismicity_features
        )
    )

    n_model_parameters = 4
    hidden_layer_sizes = [16, 8]
    output_shift = 1e-3
    combined_model = head_models.magnitude_prediction_model(
        spatially_dependent_models={
            'short_seismicity': short_term_seismicity_model,
            'long_seismicity': long_term_seismicity_model,
            'columns': columns_model,
        },
        spatially_independent_models={
            'recent': recent_model,
        },
        n_model_parameters=n_model_parameters,
        hidden_layer_sizes=hidden_layer_sizes,
        output_shift=output_shift,
    )

    # Test that we can apply the model on the features, in the expected order.
    prediction = combined_model([
        columns_features,
        long_term_seismicity_features_flat,
        short_term_seismicity_features_flat,
        (recent_features_flat, recent_grid_features_flat),
    ])

    self.assertEqual(prediction.shape, (n_events, n_model_parameters))
    self.assertGreaterEqual(prediction.numpy().min(), output_shift)

    # Test some properties of the head model.
    self.assertEqual(
        combined_model.layers[-2].activation, tf.keras.activations.softplus
    )
    # Last two layers are a Lambda FC with a soflplus activation
    head_layers = combined_model.layers[-len(hidden_layer_sizes) - 2 : -2]
    for layer in head_layers:
      self.assertEqual(layer.activation, tf.keras.activations.tanh)
    self.assertEqual(
        combined_model.layers[-2].activation, tf.keras.activations.softplus
    )


@gin.configurable()
def random_catalog(dataframe_len=1000, spatial_vector=None):
  rnd_seed = np.random.RandomState(seed=1905)
  time_vec = np.arange(dataframe_len)
  if spatial_vector is None:
    spatial_vector = np.ones(dataframe_len)

  catalog_simulation = pd.DataFrame({
      'time': time_vec,
      'longitude': spatial_vector * rnd_seed.uniform(size=(dataframe_len)),
      'latitude': spatial_vector * rnd_seed.uniform(size=(dataframe_len)),
      'depth': rnd_seed.uniform(size=(dataframe_len)),
      'magnitude': 8 + rnd_seed.uniform(size=(dataframe_len)),
      'a': rnd_seed.uniform(size=(dataframe_len)),
      'b': rnd_seed.uniform(size=(dataframe_len)),
  })

  return catalog_simulation


def _create_encoders_features_models(
    examples, grid_side, recent_encoder_spatial_encoding_kwargs
):
  """Create encoders, features and models for tests generic use.

  Args:
    examples: Examples for building features.
    grid_side: Grid side in deg for the seismicity encoder.
    recent_encoder_spatial_encoding_kwargs: kwargs for the method
      RecentEarthquakesEncoder.spatial_encoding. Should include all kwargs:
      total_pixels, first_pixel_index, total_regions, region_index.

  Returns:
    3 lists:
      1) encoders_list = [recent_encoder, long_term_seismicity_encoder,
        short_term_seismicity_encoder]  # len=3
      2) features_list = [recent_features, recent_grid_features,
        long_term_seismicity_features, short_term_seismicity_features]  # len=4
      3) models_list = [recent_model, long_term_seismicity_model,
        short_term_seismicity_model]  # len=3
  """
  recent_encoder = encoders.RecentEarthquakesEncoder()
  long_term_seismicity_encoder = encoders.SeismicityRateEncoder()
  short_term_seismicity_encoder = encoders.SeismicityRateEncoder()
  short_term_seismicity_encoder.name = 'short_term_seismicity'

  encoders_list = [
      recent_encoder,
      long_term_seismicity_encoder,
      short_term_seismicity_encoder,
  ]

  recent_features = recent_encoder.build_features(
      examples,
      limit_lookback_seconds=864000,
      max_earthquakes=3,
  )
  recent_grid_features = recent_encoder.build_location_features(
      examples, **recent_encoder_spatial_encoding_kwargs
  )
  long_term_seismicity_features = long_term_seismicity_encoder.build_features(
      examples,
      grid_side_deg=grid_side,
      lookback_seconds=[86400 * 30, 86400 * 90, 86400 * 365],
      magnitudes=[5, 6, 7],
  )
  short_term_seismicity_features = short_term_seismicity_encoder.build_features(
      examples,
      grid_side_deg=grid_side,
      lookback_seconds=[86400, 86400 * 7],
      magnitudes=[4.5, 5, 5.5, 6, 6.5, 7],
  )

  features_list = [
      recent_features,
      recent_grid_features,
      long_term_seismicity_features,
      short_term_seismicity_features,
  ]

  recent_model = recent_encoder.build_model()
  long_term_seismicity_model = long_term_seismicity_encoder.build_model()
  short_term_seismicity_model = short_term_seismicity_encoder.build_model()

  models_list = [
      recent_model,
      long_term_seismicity_model,
      short_term_seismicity_model,
  ]

  return encoders_list, features_list, models_list


if __name__ == '__main__':
  absltest.main()

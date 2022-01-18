# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""Unit tests for ModelDefinition code.

Use python3 -m models.definitions.base_model_definition_test to run the tests.
"""
import unittest

import mock
import numpy as np
import pandas as pd

from covid_epidemiology.src import constants
from covid_epidemiology.src.feature_preprocessing import FeaturePreprocessingConfig
from covid_epidemiology.src.models.definitions import base_model_definition as base


# Patching the abstract class to allow for testing helper methods
class TestBaseCovidModelDefinition(unittest.TestCase):

  @mock.patch.object(base.BaseCovidModelDefinition, '__abstractmethods__',
                     set())
  def test_preprocess_static_data(self):
    preprocessing_config = FeaturePreprocessingConfig(
        imputation_strategy='median', standardize=True)
    model = base.BaseCovidModelDefinition(
        static_preprocessing_config=preprocessing_config)
    raw_data = {
        constants.POPULATION: {
            '01': 10,
            '02': 20,
            '03': 30,
            '04': 40,
        },
        constants.HOUSEHOLDS: {
            '01': 2,
            '02': None,
            '03': 40,
            '04': None,
        },
        constants.HOSPITAL_RATING_AVERAGE: {
            '01': 2,
            '02': 5,
            '03': None,
            '04': 3,
        },
    }
    model.get_static_features = mock.Mock(
        return_value={
            constants.POPULATION:
                constants.POPULATION,
            constants.HOUSEHOLDS:
                constants.HOUSEHOLDS,
            constants.HOSPITAL_RATING_AVERAGE:
                constants.HOSPITAL_RATING_AVERAGE,
        })

    # Normalizes everything but Population by default
    expected_data = {
        constants.POPULATION: {
            '01': 10,
            '02': 20,
            '03': 30,
            '04': 40
        },
        constants.HOUSEHOLDS: {
            '01': 0,
            '02': 19 / 38,
            '03': 1,
            '04': 19 / 39
        },
        constants.HOSPITAL_RATING_AVERAGE: {
            '01': 0,
            '02': 1,
            '03': 1 / 3,
            '04': 1 / 3
        },
    }

    transformed_data, _ = model.transform_static_features(raw_data)
    # Need to get around floating point issues so using pandas series
    for feature in expected_data:
      self.assertIn(feature, transformed_data)
      expected_feature = pd.Series(expected_data[feature])
      transformed_data_feature = pd.Series(expected_data[feature])
      pd.testing.assert_series_equal(expected_feature, transformed_data_feature)

  @mock.patch.object(base.BaseCovidModelDefinition, '__abstractmethods__',
                     set())
  def test_transform_static_features_without_population(self):
    preprocessing_config = FeaturePreprocessingConfig(
        imputation_strategy='median', standardize=True)
    model = base.BaseCovidModelDefinition(
        static_preprocessing_config=preprocessing_config)
    raw_data = {
        constants.HOUSEHOLDS: {
            '01': 2,
            '02': None,
            '03': 40,
            '04': None,
        },
        constants.HOSPITAL_RATING_AVERAGE: {
            '01': 2,
            '02': 5,
            '03': None,
            '04': 3,
        },
    }
    model.get_static_features = mock.Mock(
        return_value={
            constants.HOUSEHOLDS:
                constants.HOUSEHOLDS,
            constants.HOSPITAL_RATING_AVERAGE:
                constants.HOSPITAL_RATING_AVERAGE,
        })
    with self.assertRaisesRegex(ValueError,
                                'Static features must include population'):
      model.transform_static_features(raw_data)

  @mock.patch.object(base.BaseCovidModelDefinition, '__abstractmethods__',
                     set())
  def test_preprocess_ts_data(self):
    preprocessing_config = FeaturePreprocessingConfig(
        imputation_strategy='median',
        standardize=True,
        ffill_features=True,
        bfill_features=True)
    model = base.BaseCovidModelDefinition(
        static_preprocessing_config=preprocessing_config)

    # Needs deaths and confirmed
    raw_data = {
        constants.DEATH: {
            '01': np.array([0, 1, 2, np.nan, 4]),
            '02': np.array([1, 2, 3, 4, np.nan]),
            '03': np.array([np.nan, 3, 4, 5, 6]),
        },
        constants.CONFIRMED: {
            '01': np.array([10, 11, 12, np.nan, 14]),
            '02': np.array([11, 21, 31, 41, np.nan]),
            '03': np.array([np.nan, 13, 14, 15, 16]),
        },
        constants.MOBILITY_INDEX: {
            '01': np.array([4, 3, 2, np.nan, 4]),
            '02': np.array([np.nan, 1, 2, 1, 4]),
            '03': np.array([10, 1, 1, np.nan, 1]),
        },
    }

    expected_data = {
        constants.DEATH: {
            '01': np.array([0, 1, 2, np.nan, 4]),
            '02': np.array([1, 2, 3, 4, np.nan]),
            '03': np.array([np.nan, 3, 4, 5, 6]),
        },
        constants.CONFIRMED: {
            '01': np.array([10, 11, 12, np.nan, 14]),
            '02': np.array([11, 21, 31, 41, np.nan]),
            '03': np.array([np.nan, 13, 14, 15, 16]),
        },
        constants.MOBILITY_INDEX: {
            '01': (np.array([4, 3, 2, 2, 4]) - 1.0) / (10 - 1.0),
            '02': (np.array([1, 1, 2, 1, 4]) - 1.0) / (10 - 1.0),
            '03': (np.array([10, 1, 1, 1, 1]) - 1.0) / (10.0 - 1),
        },
    }

    static_raw_data = {
        constants.POPULATION: {
            '01': 10,
            '02': 20,
            '03': 30,
        },
        constants.HOUSEHOLDS: {
            '01': 2,
            '02': None,
            '03': 40,
        },
        constants.HOSPITAL_RATING_AVERAGE: {
            '01': 2,
            '02': 5,
            '03': None,
        },
    }
    model.get_static_features = mock.Mock(
        return_value={
            constants.POPULATION:
                constants.POPULATION,
            constants.HOUSEHOLDS:
                constants.HOUSEHOLDS,
            constants.HOSPITAL_RATING_AVERAGE:
                constants.HOSPITAL_RATING_AVERAGE,
        })

    static_transformed_data, _ = model.transform_static_features(
        static_raw_data)

    model.get_ts_features = mock.Mock(
        return_value={
            constants.DEATH: constants.DEATH,
            constants.MOBILITY_INDEX: constants.MOBILITY_INDEX,
        })

    # Normalizes nothing by default but adds 4 features.
    added_features = {
        constants.DEATH_PREPROCESSED,
        constants.DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO,
        constants.CONFIRMED_PREPROCESSED,
        constants.CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO
    }
    transformed_data, _ = model.transform_ts_features(raw_data,
                                                      static_transformed_data,
                                                      5)
    for new_feature in added_features:
      self.assertIn(new_feature, transformed_data)
      del transformed_data[new_feature]

    for feature in raw_data:
      for loc in raw_data[feature]:
        np.testing.assert_array_almost_equal(transformed_data[feature][loc],
                                             raw_data[feature][loc])

    # Add mobility to be pre-processed
    model.get_ts_features_to_preprocess = mock.Mock(
        return_value={constants.MOBILITY_INDEX})

    # Will now pre-process mobility index
    transformed_data, _ = model.transform_ts_features(raw_data, static_raw_data,
                                                      5)

    for new_feature in added_features:
      self.assertIn(new_feature, transformed_data)

    for feature in expected_data:
      for loc in expected_data[feature]:
        np.testing.assert_array_almost_equal(transformed_data[feature][loc],
                                             expected_data[feature][loc])

  @mock.patch.object(base.BaseCovidModelDefinition, '__abstractmethods__',
                     set())
  def test_transform_ts_features_without_death_feature(self):
    preprocessing_config = FeaturePreprocessingConfig(
        imputation_strategy='median', standardize=True)
    model = base.BaseCovidModelDefinition(
        static_preprocessing_config=preprocessing_config)
    # Missing deaths feature
    raw_data = {
        constants.CONFIRMED: {
            '01': np.array([10, 11, 12, np.nan, 14]),
            '02': np.array([11, 21, 31, 41, np.nan]),
            '03': np.array([np.nan, 13, 14, 15, 16]),
        },
    }
    static_raw_data = {
        constants.POPULATION: {
            '01': 10,
            '02': 20,
            '03': 30,
        },
    }
    model.get_static_features = mock.Mock(return_value={
        constants.POPULATION: constants.POPULATION,
    })

    static_transformed_data, _ = model.transform_static_features(
        static_raw_data)

    model.get_ts_features = mock.Mock(
        return_value={
            constants.DEATH: constants.DEATH,
            constants.MOBILITY_INDEX: constants.MOBILITY_INDEX,
        })
    with self.assertRaisesRegex(ValueError,
                                'death must be in the input features'):
      model.transform_ts_features(raw_data, static_transformed_data, 5)

  @mock.patch.object(base.BaseCovidModelDefinition, '__abstractmethods__',
                     set())
  def test_transform_ts_features_without_confirmed_feature(self):
    preprocessing_config = FeaturePreprocessingConfig(
        imputation_strategy='median', standardize=True)
    model = base.BaseCovidModelDefinition(
        static_preprocessing_config=preprocessing_config)
    # Missing confirmed feature
    raw_data = {
        constants.DEATH: {
            '01': np.array([0, 1, 2, np.nan, 4]),
            '02': np.array([1, 2, 3, 4, np.nan]),
            '03': np.array([np.nan, 3, 4, 5, 6]),
        },
    }
    static_raw_data = {
        constants.POPULATION: {
            '01': 10,
            '02': 20,
            '03': 30,
        },
    }
    model.get_static_features = mock.Mock(return_value={
        constants.POPULATION: constants.POPULATION,
    })

    static_transformed_data, _ = model.transform_static_features(
        static_raw_data)

    model.get_ts_features = mock.Mock(
        return_value={
            constants.DEATH: constants.DEATH,
            constants.MOBILITY_INDEX: constants.MOBILITY_INDEX,
        })
    with self.assertRaisesRegex(ValueError,
                                'confirmed must be in the input features'):
      model.transform_ts_features(raw_data, static_transformed_data, 5)

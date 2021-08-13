# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for models.definitions.us_model_definitions."""

import unittest

import numpy as np
import pandas as pd

from covid_epidemiology.src import constants
from covid_epidemiology.src.models.definitions import us_model_definitions


class TestStateModelDefinition(unittest.TestCase):

  def test_get_ts_features(self):
    expected_ts_features = {
        constants.DEATH:
            constants.JHU_DEATH_FEATURE_KEY,
        constants.CONFIRMED:
            constants.JHU_CONFIRMED_FEATURE_KEY,
        constants.RECOVERED_DOC:
            constants.RECOVERED_FEATURE_KEY,
        constants.HOSPITALIZED:
            constants.HOSPITALIZED_FEATURE_KEY,
        constants.HOSPITALIZED_INCREASE:
            constants.HOSPITALIZED_INCREASE_FEATURE_KEY,
        constants.ICU:
            constants.ICU_FEATURE_KEY,
        constants.VENTILATOR:
            constants.VENTILATOR_FEATURE_KEY,
        constants.MOBILITY_INDEX:
            constants.MOBILITY_INDEX,
        constants.MOBILITY_SAMPLES:
            constants.MOBILITY_SAMPLES,
        constants.TOTAL_TESTS:
            constants.TOTAL_TESTS,
        constants.AMP_RESTAURANTS:
            constants.AMP_RESTAURANTS,
        constants.AMP_NON_ESSENTIAL_BUSINESS:
            constants.AMP_NON_ESSENTIAL_BUSINESS,
        constants.AMP_STAY_AT_HOME:
            constants.AMP_STAY_AT_HOME,
        constants.AMP_SCHOOLS_SECONDARY_EDUCATION:
            constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
        constants.AMP_EMERGENCY_DECLARATION:
            constants.AMP_EMERGENCY_DECLARATION,
        constants.AMP_GATHERINGS:
            constants.AMP_GATHERINGS,
        constants.AMP_FACE_MASKS:
            constants.AMP_FACE_MASKS,
        constants.DOW_WINDOW:
            constants.DOW_WINDOW,
        constants.AVERAGE_TEMPERATURE:
            constants.AVERAGE_TEMPERATURE,
        constants.MAX_TEMPERATURE:
            constants.MAX_TEMPERATURE,
        constants.MIN_TEMPERATURE:
            constants.MIN_TEMPERATURE,
        constants.RAINFALL:
            constants.RAINFALL,
        constants.SNOWFALL:
            constants.SNOWFALL,
        constants.COMMERCIAL_SCORE:
            constants.COMMERCIAL_SCORE,
        constants.ANTIGEN_POSITIVE:
            constants.ANTIGEN_POSITIVE,
        constants.ANTIGEN_TOTAL:
            constants.ANTIGEN_TOTAL,
        constants.ANTIBODY_NEGATIVE:
            constants.ANTIBODY_NEGATIVE,
        constants.ANTIBODY_TOTAL:
            constants.ANTIBODY_TOTAL,
        constants.SYMPTOM_COUGH:
            constants.SYMPTOM_COUGH,
        constants.SYMPTOM_CHILLS:
            constants.SYMPTOM_CHILLS,
        constants.SYMPTOM_ANOSMIA:
            constants.SYMPTOM_ANOSMIA,
        constants.SYMPTOM_INFECTION:
            constants.SYMPTOM_INFECTION,
        constants.SYMPTOM_CHEST_PAIN:
            constants.SYMPTOM_CHEST_PAIN,
        constants.SYMPTOM_FEVER:
            constants.SYMPTOM_FEVER,
        constants.SYMPTOM_SHORTNESSBREATH:
            constants.SYMPTOM_SHORTNESSBREATH,
        constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL:
            constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL,
        constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL:
            constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL,
    }
    state_model = us_model_definitions.StateModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_ts_features = state_model.get_ts_features()
    np.testing.assert_equal(expected_ts_features, actual_ts_features)

  def test_get_ts_features_to_preprocess(self):
    expected_ts_features = {
        constants.MOBILITY_INDEX,
        constants.MOBILITY_SAMPLES,
        constants.AMP_RESTAURANTS,
        constants.AMP_NON_ESSENTIAL_BUSINESS,
        constants.AMP_STAY_AT_HOME,
        constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
        constants.AMP_EMERGENCY_DECLARATION,
        constants.AMP_GATHERINGS,
        constants.AMP_FACE_MASKS,
        constants.CONFIRMED_PER_TESTS,
        constants.DEATH_PREPROCESSED,
        constants.CONFIRMED_PREPROCESSED,
        constants.DOW_WINDOW,
        constants.TOTAL_TESTS_PER_CAPITA,
        constants.TOTAL_TESTS,
        constants.AVERAGE_TEMPERATURE,
        constants.MAX_TEMPERATURE,
        constants.MIN_TEMPERATURE,
        constants.RAINFALL,
        constants.SNOWFALL,
        constants.COMMERCIAL_SCORE,
        constants.ANTIGEN_POSITIVE_RATIO,
        constants.ANTIBODY_NEGATIVE_RATIO,
        constants.SYMPTOM_COUGH,
        constants.SYMPTOM_CHILLS,
        constants.SYMPTOM_ANOSMIA,
        constants.SYMPTOM_INFECTION,
        constants.SYMPTOM_CHEST_PAIN,
        constants.SYMPTOM_FEVER,
        constants.SYMPTOM_SHORTNESSBREATH,
        constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED,
        constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED,
    }
    state_model = us_model_definitions.StateModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_ts_features = state_model.get_ts_features_to_preprocess()
    np.testing.assert_equal(expected_ts_features, actual_ts_features)

  def test_extract_ts_state_features(self):
    ts_data = pd.DataFrame([
        {
            "feature_name": constants.JHU_CONFIRMED_FEATURE_KEY,
            "feature_value": 100,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.JHU_CONFIRMED_FEATURE_KEY,
            "feature_value": 200,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.JHU_DEATH_FEATURE_KEY,
            "feature_value": 10,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.JHU_DEATH_FEATURE_KEY,
            "feature_value": float("nan"),  # Not populated should ffill to 10.
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.HOSPITALIZED_FEATURE_KEY,
            "feature_value": 100,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.HOSPITALIZED_FEATURE_KEY,
            "feature_value": 200,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ICU_FEATURE_KEY,
            "feature_value": 2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ICU_FEATURE_KEY,
            "feature_value": 5,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VENTILATOR_FEATURE_KEY,
            "feature_value": 50,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VENTILATOR_FEATURE_KEY,
            "feature_value": 100,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_INDEX,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_INDEX,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_SAMPLES,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_SAMPLES,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.TOTAL_TESTS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.TOTAL_TESTS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_GATHERINGS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_GATHERINGS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_EMERGENCY_DECLARATION,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_EMERGENCY_DECLARATION,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_RESTAURANTS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_RESTAURANTS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_NON_ESSENTIAL_BUSINESS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_NON_ESSENTIAL_BUSINESS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_STAY_AT_HOME,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_STAY_AT_HOME,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_FACE_MASKS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_FACE_MASKS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AVERAGE_TEMPERATURE,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AVERAGE_TEMPERATURE,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MAX_TEMPERATURE,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MAX_TEMPERATURE,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MIN_TEMPERATURE,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MIN_TEMPERATURE,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.RAINFALL,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.RAINFALL,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SNOWFALL,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SNOWFALL,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.COMMERCIAL_SCORE,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.COMMERCIAL_SCORE,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIGEN_POSITIVE,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIGEN_POSITIVE,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIGEN_TOTAL,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIGEN_TOTAL,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIBODY_NEGATIVE,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIBODY_NEGATIVE,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIBODY_TOTAL,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.ANTIBODY_TOTAL,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.RECOVERED_FEATURE_KEY,
            "feature_value": 12,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.RECOVERED_FEATURE_KEY,
            "feature_value": 11,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.HOSPITALIZED_INCREASE_FEATURE_KEY,
            "feature_value": 16,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.HOSPITALIZED_INCREASE_FEATURE_KEY,
            "feature_value": 14,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_COUGH,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_COUGH,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_CHILLS,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_CHILLS,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_ANOSMIA,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_ANOSMIA,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_INFECTION,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_INFECTION,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_CHEST_PAIN,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_CHEST_PAIN,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_FEVER,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_FEVER,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_SHORTNESSBREATH,
            "feature_value": 0.6,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.SYMPTOM_SHORTNESSBREATH,
            "feature_value": 0.7,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL,
            "feature_value": 10,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL,
            "feature_value": 20,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL,
            "feature_value": 5,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL,
            "feature_value": 10,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
    ])

    static_data = pd.DataFrame([{
        "feature_name": constants.AQI_MEAN,
        "feature_value": 105,
        "geo_id": "4059"
    }, {
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4058"
    }, {
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4059"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 120,
        "geo_id": "4058"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 100,
        "geo_id": "4059"
    }, {
        "feature_name": constants.POPULATION,
        "feature_value": 70,
        "geo_id": "4059"
    }, {
        "feature_name": constants.POPULATION,
        "feature_value": 50,
        "geo_id": "4058"
    }, {
        "feature_name": constants.POPULATION,
        "feature_value": 10,
        "geo_id": "4057"
    }])

    state_model = us_model_definitions.StateModelDefinition(gt_source="JHU")
    static_features, _ = state_model._extract_static_features(
        static_data=static_data, locations=["4059"])
    actual, _ = state_model._extract_ts_features(
        ts_data=ts_data,
        static_features=static_features,
        locations=["4059"],
        training_window_size=2)
    expected = {
        constants.CONFIRMED: {
            "4059": np.array([100, 200], dtype="float32")
        },
        constants.DEATH: {
            "4059": [10, np.nan]
        },
        constants.DEATH_PREPROCESSED: {
            "4059": [0, 0]
        },
        constants.ICU: {
            "4059": np.array([2, 5], dtype="float32")
        },
        constants.INFECTED: None,
        constants.HOSPITALIZED: {
            "4059": np.array([100, 200], dtype="float32")
        },
        constants.MOBILITY_INDEX: {
            "4059": np.array([1, 0], dtype="float32")
        },
        constants.VENTILATOR: {
            "4059": np.array([50, 100], dtype="float32")
        },
        constants.RECOVERED_DOC: {
            "4059": np.array([11, 12], dtype="float32")
        },
        constants.HOSPITALIZED_INCREASE: {
            "4059": np.array([14, 16], dtype="float32")
        },
        constants.HOSPITALIZED_CUMULATIVE: {
            "4059": np.array([14, 30], dtype="float32")
        },
        constants.TOTAL_TESTS_PER_CAPITA: {
            "4059": np.array([1, 0], dtype="float32")
        },
    }
    for ts_feature_name in expected:
      self.assertIn(ts_feature_name, actual)
      np.testing.assert_equal(
          actual[ts_feature_name], expected[ts_feature_name],
          "Feature name {} is not aligned.".format(ts_feature_name))

  def test_get_static_features(self):
    expected_static_features = {
        constants.POPULATION:
            constants.POPULATION,
        constants.INCOME_PER_CAPITA:
            constants.INCOME_PER_CAPITA,
        constants.POPULATION_DENSITY_PER_SQKM:
            constants.POPULATION_DENSITY_PER_SQKM,
        constants.HOUSEHOLD_FOOD_STAMP:
            constants.HOUSEHOLD_FOOD_STAMP,
        constants.KAISER_POPULATION:
            constants.KAISER_POPULATION,
        constants.KAISER_60P_POPULATION:
            constants.KAISER_60P_POPULATION,
        constants.ICU_BEDS:
            constants.ICU_BEDS,
        constants.HOUSEHOLDS:
            constants.HOUSEHOLDS,
        constants.HOSPITAL_RATING1:
            constants.HOSPITAL_RATING1,
        constants.HOSPITAL_RATING2:
            constants.HOSPITAL_RATING2,
        constants.HOSPITAL_RATING3:
            constants.HOSPITAL_RATING3,
        constants.HOSPITAL_RATING4:
            constants.HOSPITAL_RATING4,
        constants.HOSPITAL_RATING5:
            constants.HOSPITAL_RATING5,
        constants.AQI_MEAN:
            constants.AQI_MEAN,
        constants.NON_EMERGENCY_SERVICES:
            constants.NON_EMERGENCY_SERVICES,
        constants.EMERGENCY_SERVICES:
            constants.EMERGENCY_SERVICES,
        constants.HOSPITAL_ACUTE_CARE:
            constants.HOSPITAL_ACUTE_CARE,
        constants.CRITICAL_ACCESS_HOSPITAL:
            constants.CRITICAL_ACCESS_HOSPITAL,
        constants.PATIENCE_EXPERIENCE_SAME:
            constants.PATIENCE_EXPERIENCE_SAME,
        constants.PATIENCE_EXPERIENCE_BELOW:
            constants.PATIENCE_EXPERIENCE_BELOW,
        constants.PATIENCE_EXPERIENCE_ABOVE:
            constants.PATIENCE_EXPERIENCE_ABOVE,
    }
    state_model = us_model_definitions.StateModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_static_features = state_model.get_static_features()
    np.testing.assert_equal(expected_static_features, actual_static_features)

  def test_extract_state_static_features(self):
    static_data = pd.DataFrame([{
        "feature_name": constants.AQI_MEAN,
        "feature_value": 105,
        "geo_id": "4059"
    }, {
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4058"
    }, {
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4059"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 120,
        "geo_id": "4058"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 100,
        "geo_id": "4059"
    }, {
        "feature_name": constants.POPULATION,
        "feature_value": 70,
        "geo_id": "4059"
    }, {
        "feature_name": constants.POPULATION,
        "feature_value": 50,
        "geo_id": "4058"
    }, {
        "feature_name": constants.POPULATION,
        "feature_value": 10,
        "geo_id": "4057"
    }])

    state_model = us_model_definitions.StateModelDefinition(gt_source="JHU")
    actual, _ = state_model._extract_static_features(
        static_data=static_data, locations=["4059", "4058"])
    expected = {
        constants.AQI_MEAN: {
            "4059": 0,
            "4058": 0
        },
        constants.INCOME_PER_CAPITA: {
            "4059": 0,
            "4058": 1
        },
        constants.POPULATION: {
            "4059": 70,
            "4058": 50
        },
        constants.POPULATION_DENSITY_PER_SQKM: {
            "4059": 0,
            "4058": 0
        },
    }

    for static_feature_name in expected:
      self.assertEqual(actual[static_feature_name],
                       expected[static_feature_name])


class TestCountyModelDefinition(unittest.TestCase):

  def test_get_ts_features(self):
    expected_ts_features = {
        constants.DEATH:
            constants.JHU_COUNTY_DEATH_FEATURE_KEY,
        constants.CONFIRMED:
            constants.JHU_COUNTY_CONFIRMED_FEATURE_KEY,
        constants.RECOVERED_DOC:
            constants.CSRP_RECOVERED_FEATURE_KEY,
        constants.HOSPITALIZED:
            constants.CHA_HOSPITALIZED_FEATURE_KEY,
        constants.HOSPITALIZED_CUMULATIVE:
            constants.CHA_HOSPITALIZED_CUMULATIVE_FEATURE_KEY,
        constants.ICU:
            constants.CSRP_ICU_FEATURE_KEY,
        constants.MOBILITY_INDEX:
            constants.MOBILITY_INDEX,
        constants.MOBILITY_SAMPLES:
            constants.MOBILITY_SAMPLES,
        constants.CSRP_TESTS:
            constants.CSRP_TESTS,
        constants.AMP_RESTAURANTS:
            constants.AMP_RESTAURANTS,
        constants.AMP_NON_ESSENTIAL_BUSINESS:
            constants.AMP_NON_ESSENTIAL_BUSINESS,
        constants.AMP_STAY_AT_HOME:
            constants.AMP_STAY_AT_HOME,
        constants.AMP_SCHOOLS_SECONDARY_EDUCATION:
            constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
        constants.AMP_EMERGENCY_DECLARATION:
            constants.AMP_EMERGENCY_DECLARATION,
        constants.AMP_GATHERINGS:
            constants.AMP_GATHERINGS,
        constants.AMP_FACE_MASKS:
            constants.AMP_FACE_MASKS,
        constants.DOW_WINDOW:
            constants.DOW_WINDOW,
        constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL:
            constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL,
        constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL:
            constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL,
    }
    county_model = us_model_definitions.CountyModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_ts_features = county_model.get_ts_features()
    np.testing.assert_equal(expected_ts_features, actual_ts_features)

  def test_get_ts_features_to_preprocess(self):
    expected_ts_features = {
        constants.MOBILITY_INDEX,
        constants.MOBILITY_SAMPLES,
        constants.CSRP_TESTS,
        constants.CONFIRMED_PER_CSRP_TESTS,
        constants.TOTAL_TESTS_PER_CAPITA,
        constants.AMP_RESTAURANTS,
        constants.AMP_NON_ESSENTIAL_BUSINESS,
        constants.AMP_STAY_AT_HOME,
        constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
        constants.AMP_EMERGENCY_DECLARATION,
        constants.AMP_GATHERINGS,
        constants.AMP_FACE_MASKS,
        constants.DEATH_PREPROCESSED,
        constants.CONFIRMED_PREPROCESSED,
        constants.DOW_WINDOW,
        constants.TOTAL_TESTS_PER_CAPITA,
        constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED,
        constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED,
    }
    county_model = us_model_definitions.CountyModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_ts_features = county_model.get_ts_features_to_preprocess()
    np.testing.assert_equal(expected_ts_features, actual_ts_features)

  def test_extract_ts_county_features(self):
    ts_data = pd.DataFrame([
        {
            "feature_name": "confirmed_cases",
            "feature_value": 100,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": "confirmed_cases",
            "feature_value": 200,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": "deaths",
            "feature_value": 10,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": "deaths",
            "feature_value": 13,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_INDEX,
            "feature_value": 0.0,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_INDEX,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_SAMPLES,
            "feature_value": 10,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.MOBILITY_SAMPLES,
            "feature_value": 12,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.CSRP_TESTS,
            "feature_value": 70,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.CSRP_TESTS,
            "feature_value": 140,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_GATHERINGS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_GATHERINGS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_EMERGENCY_DECLARATION,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_EMERGENCY_DECLARATION,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_RESTAURANTS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_RESTAURANTS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_NON_ESSENTIAL_BUSINESS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_NON_ESSENTIAL_BUSINESS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_STAY_AT_HOME,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_STAY_AT_HOME,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_FACE_MASKS,
            "feature_value": 1.0,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.AMP_FACE_MASKS,
            "feature_value": 1.2,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.CSRP_RECOVERED_FEATURE_KEY,
            "feature_value": 12,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CSRP_RECOVERED_FEATURE_KEY,
            "feature_value": 11,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CHA_HOSPITALIZED_FEATURE_KEY,
            "feature_value": 100,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CHA_HOSPITALIZED_FEATURE_KEY,
            "feature_value": 200,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CHA_HOSPITALIZED_CUMULATIVE_FEATURE_KEY,
            "feature_value": 200,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CHA_HOSPITALIZED_CUMULATIVE_FEATURE_KEY,
            "feature_value": 300,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CSRP_ICU_FEATURE_KEY,
            "feature_value": 20,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.CSRP_ICU_FEATURE_KEY,
            "feature_value": 30,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059",
        },
        {
            "feature_name": constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL,
            "feature_value": 10,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL,
            "feature_value": 20,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL,
            "feature_value": 5,
            "dt": np.datetime64("2020-01-22"),
            "geo_id": "4059"
        },
        {
            "feature_name": constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL,
            "feature_value": 10,
            "dt": np.datetime64("2020-01-23"),
            "geo_id": "4059"
        },
    ])

    static_data = pd.DataFrame([{
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4059"
    }, {
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4058"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 120,
        "geo_id": "4058"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 100,
        "geo_id": "4059"
    }, {
        "feature_name": constants.COUNTY_POPULATION,
        "feature_value": 70,
        "geo_id": "4059"
    }, {
        "feature_name": constants.COUNTY_POPULATION,
        "feature_value": 50,
        "geo_id": "4058"
    }, {
        "feature_name": constants.COUNTY_POPULATION,
        "feature_value": 10,
        "geo_id": "4057"
    }])

    state_model = us_model_definitions.CountyModelDefinition(
        gt_source="USAFACTS")

    static_features, _ = state_model._extract_static_features(
        static_data=static_data, locations=["4059"])

    actual, _ = state_model._extract_ts_features(
        ts_data=ts_data,
        static_features=static_features,
        locations=["4059"],
        training_window_size=2)

    expected = {
        constants.DEATH: {
            "4059": np.array([10, 13], dtype="float32")
        },
        constants.CONFIRMED: {
            "4059": np.array([100, 200], dtype="float32")
        },
        constants.MOBILITY_SAMPLES: {
            "4059": np.array([0, 1], dtype="float32")
        },
        constants.MOBILITY_INDEX: {
            "4059": np.array([0, 1], dtype="float32")
        },
        constants.CSRP_TESTS: {
            "4059": np.array([0, 1], dtype="float32")
        },
        constants.RECOVERED_DOC: {
            "4059": np.array([11, 12], dtype="float32"),
        },
        constants.HOSPITALIZED: {
            "4059": np.array([100, 200], dtype="float32"),
        },
        constants.HOSPITALIZED_CUMULATIVE: {
            "4059": np.array([200, 300], dtype="float32"),
        },
        constants.ICU: {
            "4059": np.array([20, 30], dtype="float32"),
        },
        constants.TOTAL_TESTS_PER_CAPITA: {
            "4059": np.array([0, 0], dtype="float32"),
        },
    }

    for ts_feature_name in expected:
      self.assertIn(ts_feature_name, actual)
      np.testing.assert_equal(
          actual[ts_feature_name], expected[ts_feature_name],
          "Unexpected value for feature %s" % ts_feature_name)

  def test_get_static_features(self):
    county_model = us_model_definitions.CountyModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_static_features = county_model.get_static_features()
    self.assertEqual(len(actual_static_features), 51)

  def test_get_all_locations(self):
    input_df = pd.DataFrame(
        {constants.GEO_ID_COLUMN: ["4059", "4060", "4061", "4062"]})
    # Exclude FIPS 15005 (Kalawao County, no longer exist)
    expected_locations = {"4059", "4060", "4061", "4062"}
    county_model = us_model_definitions.CountyModelDefinition(
        gt_source=constants.GT_SOURCE_JHU)
    actual_locations = county_model.get_all_locations(input_df)
    np.testing.assert_equal(expected_locations, actual_locations)

  def test_extract_county_static_features(self):
    static_data = pd.DataFrame([{
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4059"
    }, {
        "feature_name": constants.AREA,
        "feature_value": 10,
        "geo_id": "4058"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 120,
        "geo_id": "4058"
    }, {
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 100,
        "geo_id": "4059"
    }, {
        "feature_name": constants.COUNTY_POPULATION,
        "feature_value": 70,
        "geo_id": "4059"
    }, {
        "feature_name": constants.COUNTY_POPULATION,
        "feature_value": 50,
        "geo_id": "4058"
    }, {
        "feature_name": constants.COUNTY_POPULATION,
        "feature_value": 10,
        "geo_id": "4057"
    }])

    county_model = us_model_definitions.CountyModelDefinition(gt_source="JHU")
    actual, _ = county_model._extract_static_features(
        static_data=static_data, locations=["4059", "4058"])
    expected = {
        constants.INCOME_PER_CAPITA: {
            "4059": 0,
            "4058": 1
        },
        constants.POPULATION: {
            "4059": 70,
            "4058": 50
        }
    }

    for static_feature_name in expected:
      self.assertEqual(actual[static_feature_name],
                       expected[static_feature_name],
                       "Unexpected value for feature %s" % static_feature_name)


if __name__ == "__main__":
  unittest.main()

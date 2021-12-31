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
from covid_epidemiology.src.models.definitions import japan_model_definitions


class TestPrefectureModelDefinition(unittest.TestCase):

  def test_get_ts_features(self):
    expected_ts_features = {
        constants.DEATH:
            constants.JAPAN_PREFECTURE_DEATH_FEATURE_KEY,
        constants.CONFIRMED:
            constants.JAPAN_PREFECTURE_CONFIRMED_FEATURE_KEY,
        constants.RECOVERED_DOC:
            constants.JAPAN_PREFECTURE_DISCHARGED_FEATURE_KEY,
        constants.HOSPITALIZED:
            constants.JAPAN_PREFECTURE_HOSPITALIZED_FEATURE_KEY,
        constants.GOOGLE_MOBILITY_PARKS:
            constants
            .JAPAN_PREFECTURE_MOBILITY_PARKS_PERCENT_FROM_BASELINE_FEATURE_KEY,
        constants.GOOGLE_MOBILITY_WORK:
            constants.
            JAPAN_PREFECTURE_MOBILITY_WORKPLACE_PERCENT_FROM_BASELINE_FEATURE_KEY,
        constants.GOOGLE_MOBILITY_RES:
            constants.
            JAPAN_PREFECTURE_MOBILITY_RESIDENTIAL_PERCENT_FROM_BASELINE_FEATURE_KEY,
        constants.GOOGLE_MOBILITY_TRANSIT:
            constants.
            JAPAN_PREFECTURE_MOBILITY_TRAIN_STATION_PERCENT_FROM_BASELINE_FEATURE_KEY,
        constants.GOOGLE_MOBILITY_GROCERY:
            constants.
            JAPAN_PREFECTURE_MOBILITY_GROCERY_AND_PHARMACY_PERCENT_FROM_BASELINE_FEATURE_KEY,
        constants.GOOGLE_MOBILITY_RETAIL:
            constants.
            JAPAN_PREFECTURE_MOBILITY_RETAIL_AND_RECREATION_PERCENT_FROM_BASELINE_FEATURE_KEY,
        constants.TOTAL_TESTS:
            constants.JAPAN_PREFECTURE_TESTED_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_STATE_OF_EMERGENCY_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_STATE_OF_EMERGENCY_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_DISCHARGED_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_DISCHARGED_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_HOSPITALIZED_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_HOSPITALIZED_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_EFFECTIVE_REPRODUCTIVE_NUMBER_FEATURE_KEY:
            constants
            .JAPAN_PREFECTURE_EFFECTIVE_REPRODUCTIVE_NUMBER_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_SURVEY_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_SURVEY_FEATURE_KEY,
        constants
        .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_UNWEIGHTED_SURVEY_FEATURE_KEY:
            constants
            .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_UNWEIGHTED_SURVEY_FEATURE_KEY,
        constants
        .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_SURVEY_FEATURE_KEY:
            constants
            .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_SURVEY_FEATURE_KEY,
        constants.
        JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_UNWEIGHTED_SURVEY_FEATURE_KEY:
            constants.
            JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_UNWEIGHTED_SURVEY_FEATURE_KEY,
        constants.DOW_WINDOW:
            constants.DOW_WINDOW,
    }
    japan_model = japan_model_definitions.PrefectureModelDefinition()
    actual_ts_features = japan_model.get_ts_features()
    self.assertDictEqual(expected_ts_features, actual_ts_features)

  def test_get_ts_features_to_preprocess(self):
    expected_ts_features = {
        constants.GOOGLE_MOBILITY_PARKS,
        constants.GOOGLE_MOBILITY_WORK,
        constants.GOOGLE_MOBILITY_RES,
        constants.GOOGLE_MOBILITY_TRANSIT,
        constants.GOOGLE_MOBILITY_GROCERY,
        constants.GOOGLE_MOBILITY_RETAIL,
        constants.TOTAL_TESTS,
        constants.JAPAN_PREFECTURE_STATE_OF_EMERGENCY_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_DISCHARGED_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_HOSPITALIZED_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_EFFECTIVE_REPRODUCTIVE_NUMBER_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_SURVEY_FEATURE_KEY,
        constants
        .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_UNWEIGHTED_SURVEY_FEATURE_KEY,
        constants
        .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_SURVEY_FEATURE_KEY,
        constants.
        JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_UNWEIGHTED_SURVEY_FEATURE_KEY,
        constants.DOW_WINDOW,
    }
    japan_model = japan_model_definitions.PrefectureModelDefinition()
    actual_ts_features = japan_model.get_ts_features_to_preprocess()
    np.testing.assert_equal(expected_ts_features, actual_ts_features)

  def test_extract_japan_prefecture_ts_prefectures(self):
    model = japan_model_definitions.PrefectureModelDefinition()
    feature_name_map = model.get_ts_features()

    def _dummy_dict(feat):
      return {
          "feature_name": feat,
          "feature_value": 100,
          "dt": np.datetime64("2020-01-22"),
          "geo_id": "4059"
      }

    ts_data = pd.DataFrame(
        [_dummy_dict(feat) for feat in feature_name_map.values()])

    static_data = pd.DataFrame([{
        "feature_name": constants.POPULATION,
        "feature_value": 120,
        "geo_id": "4059"
    }])

    static_features, _ = model._extract_static_features(
        static_data=static_data, locations=["4059"])

    actual, _ = model._extract_ts_features(
        ts_data=ts_data,
        static_features=static_features,
        locations=["4059"],
        training_window_size=2)
    self.assertIsNotNone(actual)

  def test_get_static_features(self):
    expected_static_features = {
        # Population and demographics.
        constants.POPULATION:
            constants.JAPAN_PREFECTURE_NUM_PEOPLE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_MALE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_MALE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_FEMALE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_FEMALE_FEATURE_KEY,
        constants.DENSITY:
            constants.JAPAN_PREFECTURE_POPULATION_DENSITY_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_0_TO_14_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_AGE_0_TO_14_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_15_TO_64_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_AGE_15_TO_64_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_64_PLUS_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_AGE_64_PLUS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_75_PLUS_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_AGE_75_PLUS_FEATURE_KEY,
        constants.INCOME_PER_CAPITA:
            constants.JAPAN_PREFECTURE_GDP_PER_CAPITA_FEATURE_KEY,
        # Hospital resources.
        constants.JAPAN_PREFECTURE_NUM_DOCTORS_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_DOCTORS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_DOCTORS_PER_100K_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_DOCTORS_PER_100K_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_PER_100K_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_PER_100K_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_PER_100K_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_PER_100K_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_NEW_ICU_BEDS_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_NUM_NEW_ICU_BEDS_FEATURE_KEY,
        # Wellness and health.
        constants.JAPAN_PREFECTURE_H1N1_in_2010_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_H1N1_in_2010_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_ALCOHOL_INTAKE_SCORE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_ALCOHOL_INTAKE_SCORE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_MALE_AVERAGE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_BMI_MALE_AVERAGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_MALE_LOWER_RANGE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_BMI_MALE_LOWER_RANGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_MALE_UPPER_RANGE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_BMI_MALE_UPPER_RANGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_FEMALE_AVERAGE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_BMI_FEMALE_AVERAGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_FEMALE_LOWER_RANGE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_BMI_FEMALE_LOWER_RANGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_FEMALE_UPPER_RANGE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_BMI_FEMALE_UPPER_RANGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_SMOKERS_MALE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_SMOKERS_MALE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_SMOKERS_FEMALE_FEATURE_KEY:
            constants.JAPAN_PREFECTURE_SMOKERS_FEMALE_FEATURE_KEY,
    }
    japan_model = japan_model_definitions.PrefectureModelDefinition()
    actual_static_features = japan_model.get_static_features()
    np.testing.assert_equal(expected_static_features, actual_static_features)

  def test_extract_japan_prefecture_static_features(self):
    static_data = pd.DataFrame([{
        "feature_name": constants.INCOME_PER_CAPITA,
        "feature_value": 120,
        "geo_id": "4058"
    }])

    model = japan_model_definitions.PrefectureModelDefinition()
    actual, _ = model._extract_static_features(
        static_data=static_data, locations=["US", "IR"])
    # TODO(joelshor): Add actual checks.
    self.assertIsNotNone(actual)


if __name__ == "__main__":
  unittest.main()

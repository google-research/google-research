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

"""Contains model definitions for Japan."""
from typing import Dict, Optional, Set, Tuple

import numpy as np
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src.models import generic_seir_japan_prefecture_model_constructor
from covid_epidemiology.src.models import generic_seir_specs_japan_prefecture
from covid_epidemiology.src.models.definitions import compartmental_model_definitions
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib
from covid_epidemiology.src.models.shared import model_utils
from covid_epidemiology.src.models.shared import typedefs


class PrefectureModelDefinition(
    compartmental_model_definitions.BaseSeirHospitalModelDefinition):
  """Defines the compartmental model for Japan's Prefectures."""

  def get_ts_features(self):
    return {
        # pylint:disable=line-too-long
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
        # pylint:enable=line-too-long
    }

  def get_ts_features_to_preprocess(self):
    ts_features_to_not_preprocess = {
        constants.DEATH, constants.CONFIRMED, constants.RECOVERED_DOC,
        constants.HOSPITALIZED
    }
    return {
        feature_alias for feature_alias in self.get_ts_features()
        if feature_alias not in ts_features_to_not_preprocess
    }

  def transform_ts_features(
      self,
      ts_features,
      static_features,
      initial_train_window_size,
  ):
    transformed_features, feature_scalers = super().transform_ts_features(
        ts_features, static_features, initial_train_window_size)

    transformed_features[constants.INFECTED], feature_scalers[
        constants.INFECTED] = None, None
    transformed_features[constants.HOSPITALIZED_INCREASE], feature_scalers[
        constants.HOSPITALIZED_INCREASE] = None, None
    transformed_features[constants.HOSPITALIZED_CUMULATIVE], feature_scalers[
        constants.HOSPITALIZED_CUMULATIVE] = None, None
    return transformed_features, feature_scalers

  def get_static_features(self):
    return {
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

  def get_static_features_to_preprocess(self):
    static_features_to_not_preprocess = {
        constants.POPULATION, constants.JAPAN_PREFECTURE_NUM_MALE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_FEMALE_FEATURE_KEY
    }
    return {
        feature_alias for feature_alias in self.get_static_features()
        if feature_alias not in static_features_to_not_preprocess
    }

  def get_model_spec(
      self,
      model_type,
      covariate_delay = 0,
      **kwargs,
  ):
    if model_type in {
        constants.MODEL_TYPE_TIME_VARYING_WITH_COVARIATES,
        constants.MODEL_TYPE_STATIC_SEIR, constants.MODEL_TYPE_TREND_FOLLOWING
    }:
      return generic_seir_specs_japan_prefecture.get_model_specs(
          covariate_delay)[model_type]
    else:
      raise ValueError(f"No model type {model_type} for {type(self).__name__}.")

  def get_model_constructor(
      self,
      model_spec,
      random_seed,
  ):
    """Returns the model constructor for the model.

    Args:
      model_spec: A definition of the model spec. Returned by the get_model_spec
        function.
      random_seed: A seed used for initialization of pseudo-random numbers.

    Returns:
      The model constructor instance for the country.
    """
    return generic_seir_japan_prefecture_model_constructor.JapanPrefectureModelConstructor(
        model_spec=model_spec, random_seed=random_seed)

  # Customize compartmental dynamics
  def bound_variables(
      self,
      seir_timeseries_variables,
  ):
    """Modifies the bounds to match Tokyo's data."""

    (vaccinated_ratio_list, vaccine_effectiveness_list, average_contact_id_list,
     average_contact_iud_list, reinfectable_rate_list, alpha_list,
     diagnosis_rate_list, recovery_rate_id_list, recovery_rate_iud_list,
     recovery_rate_h_list, hospitalization_rate_list, death_rate_id_list,
     death_rate_h_list) = seir_timeseries_variables

    vaccinated_ratio = model_utils.apply_relu_bounds(vaccinated_ratio_list[-1],
                                                     0.0, 1.0)
    vaccine_effectiveness = model_utils.apply_relu_bounds(
        vaccine_effectiveness_list[-1], 0.0, 1.0)
    average_contact_id = 1.0 * tf.nn.sigmoid(average_contact_id_list[-1])
    average_contact_iud = 1.0 * tf.nn.sigmoid(average_contact_iud_list[-1])
    reinfectable_rate = 0.001 * tf.nn.sigmoid(reinfectable_rate_list[-1])
    alpha = 0.2 * tf.nn.sigmoid(alpha_list[-1])
    diagnosis_rate = 0.01 + 0.29 * tf.nn.sigmoid(diagnosis_rate_list[-1])
    recovery_rate_id = 0.1 * tf.nn.sigmoid(recovery_rate_id_list[-1])
    recovery_rate_iud = 0.1 * tf.nn.sigmoid(recovery_rate_iud_list[-1])
    recovery_rate_h = 0.1 * tf.nn.sigmoid(recovery_rate_h_list[-1])
    hospitalization_rate = 1.0 * tf.nn.sigmoid(hospitalization_rate_list[-1])
    death_rate_id = 0.01 * tf.nn.sigmoid(death_rate_id_list[-1])
    death_rate_h = 0.01 * tf.nn.sigmoid(death_rate_h_list[-1])

    return (vaccinated_ratio, vaccine_effectiveness, average_contact_id,
            average_contact_iud, reinfectable_rate, alpha, diagnosis_rate,
            recovery_rate_id, recovery_rate_iud, recovery_rate_h,
            hospitalization_rate, death_rate_id, death_rate_h)

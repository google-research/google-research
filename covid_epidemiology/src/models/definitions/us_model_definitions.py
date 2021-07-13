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

"""Contains model definitions for the US."""
# from typing import Dict, List, Optional, Set, Tuple

import numpy as np
# import pandas as pd
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src import feature_preprocessing as preprocessing
from covid_epidemiology.src.features import get_gt_source_features
from covid_epidemiology.src.models import generic_seir_county_model_constructor
from covid_epidemiology.src.models import generic_seir_specs_county
from covid_epidemiology.src.models import generic_seir_specs_state
from covid_epidemiology.src.models import generic_seir_state_model_constructor
from covid_epidemiology.src.models.definitions import compartmental_model_definitions
# from covid_epidemiology.src.models.shared import model_spec as model_spec_lib
from covid_epidemiology.src.models.shared import model_utils
# from covid_epidemiology.src.models.shared import typedefs


class CountyModelDefinition(
    compartmental_model_definitions.BaseSeirHospitalModelDefinition):
  """The US county level compartmental model.

  Attributes:
    ts_preprocessing_config: The default configuration to use for pre-processing
      time-series features.
    static_preprocessing_config: The default configuration to use for
      pre-processing static features.
    random_seed: A number to be used as a random seed for the model.
    gt_source: The ground truth used.
  """

  def __init__(self,
               ts_preprocessing_config = None,
               static_preprocessing_config = None,
               random_seed = 0,
               gt_source = "",
               **kwargs):
    """Creates the compartmental model.

    Args:
      ts_preprocessing_config: The default configuration to use for
        pre-processing time-series features.
      static_preprocessing_config: The default configuration to use for
        pre-processing static features.
      random_seed: A random seed to use in the model.
      gt_source: The ground truth source to be used for this model (e.g. JHU).
      **kwargs: Model specific keyword arguments.
    """
    super().__init__(ts_preprocessing_config, static_preprocessing_config,
                     random_seed, **kwargs)

    if not gt_source:
      raise ValueError("The gt_source must be specified.")
    self.gt_source = gt_source

  def get_ts_features(self):
    gt_key_death, gt_key_confirmed = get_gt_source_features(
        "COUNTY", self.gt_source)
    return {
        constants.DEATH:
            gt_key_death,
        constants.CONFIRMED:
            gt_key_confirmed,
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

  def get_ts_features_to_preprocess(self):
    return {
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
        constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED,
        constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED,
    }

  def transform_ts_features(
      self,
      ts_features,
      static_features,
      initial_train_window_size,
  ):
    transformed_features, feature_scalers = super().transform_ts_features(
        ts_features, static_features, initial_train_window_size)
    transformed_features[constants.CONFIRMED_PER_CSRP_TESTS], feature_scalers[
        constants.CONFIRMED_PER_CSRP_TESTS] = (
            self._preprocess_ts_feature(
                preprocessing.normalize_ts_feature(
                    ts_features[constants.CONFIRMED],
                    ts_features[constants.CSRP_TESTS],
                    epsilon=1.0,
                    upper_limit=1.0,
                    lower_limit=0.0), initial_train_window_size))

    transformed_features[constants.TOTAL_TESTS_PER_CAPITA], feature_scalers[
        constants.TOTAL_TESTS_PER_CAPITA] = (
            self._preprocess_ts_feature(
                preprocessing.normalize_ts_feature(
                    ts_features[constants.CSRP_TESTS],
                    static_features[constants.COUNTY_POPULATION],
                    epsilon=1.0,
                    upper_limit=3.0,
                    lower_limit=0.0),
                initial_train_window_size,
                initial_value=0,
            ))

    if constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL in ts_features.keys():
      # Cumulative to current values
      first_dose_vaccine_current_per_day = ts_features[
          constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL].copy()
      for location_name in first_dose_vaccine_current_per_day.keys():
        first_dose_vaccine_current_per_day[location_name][1:] = np.diff(
            first_dose_vaccine_current_per_day[location_name])

      transformed_features[
          constants
          .VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED], feature_scalers[
              constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED] = (
                  self._preprocess_ts_feature(
                      preprocessing.normalize_ts_feature(
                          first_dose_vaccine_current_per_day,
                          static_features[constants.COUNTY_POPULATION],
                          epsilon=1.0,
                          upper_limit=1.0,
                          lower_limit=0.0),
                      initial_train_window_size,
                      bfill_features=False,
                      imputation_strategy="constant",
                      standardize=False))

    if constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL in ts_features.keys():
      # Cumulative to current values
      second_dose_vaccine_current_per_day = ts_features[
          constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL].copy()
      for location_name in second_dose_vaccine_current_per_day.keys():
        second_dose_vaccine_current_per_day[location_name][1:] = np.diff(
            second_dose_vaccine_current_per_day[location_name])

      transformed_features[
          constants
          .VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED], feature_scalers[
              constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED] = (
                  self._preprocess_ts_feature(
                      preprocessing.normalize_ts_feature(
                          second_dose_vaccine_current_per_day,
                          static_features[constants.COUNTY_POPULATION],
                          epsilon=1.0,
                          upper_limit=1.0,
                          lower_limit=0.0),
                      initial_train_window_size,
                      bfill_features=False,
                      imputation_strategy="constant",
                      standardize=False))

    transformed_features[constants.INFECTED], feature_scalers[
        constants.INFECTED] = None, None
    transformed_features[constants.HOSPITALIZED_INCREASE], feature_scalers[
        constants.HOSPITALIZED_INCREASE] = None, None

    return transformed_features, feature_scalers

  def get_static_features(self):
    static_features = {
        constants.POPULATION:
            constants.COUNTY_POPULATION,
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
        constants.PATIENCE_EXPERIENCE_SAME:
            constants.PATIENCE_EXPERIENCE_SAME,
        constants.PATIENCE_EXPERIENCE_BELOW:
            constants.PATIENCE_EXPERIENCE_BELOW,
        constants.PATIENCE_EXPERIENCE_ABOVE:
            constants.PATIENCE_EXPERIENCE_ABOVE,
        constants.EOC_CARE_SAME:
            constants.EOC_CARE_SAME,
        constants.EOC_CARE_ABOVE:
            constants.EOC_CARE_ABOVE,
        constants.EOC_CARE_BELOW:
            constants.EOC_CARE_BELOW,
        constants.COUNTY_POVERTY:
            constants.COUNTY_POVERTY,
        constants.COUNTY_POP_BASED_POVERTY:
            constants.COUNTY_POP_BASED_POVERTY,
        constants.COUNTY_MEDIAN_INCOME:
            constants.COUNTY_MEDIAN_INCOME,
        constants.COUNTY_HOUSEHOLD_FOOD_STAMP:
            constants.COUNTY_HOUSEHOLD_FOOD_STAMP,
        constants.COUNTY_INCOME_PER_CAPITA:
            constants.COUNTY_INCOME_PER_CAPITA,
        constants.COUNTY_HOUSEHOLDS:
            constants.COUNTY_HOUSEHOLDS,
        constants.COUNTY_GROUP_QUARTERS:
            constants.COUNTY_GROUP_QUARTERS,
    }

    # Add all the age bins that we will use
    census_age_dict = {constants.COUNTY_POPULATION: constants.COUNTY_POPULATION}
    for range_groups in constants.COUNTY_POP_AGE_RANGES.values():
      for census_subgroup in range_groups:
        census_age_dict[census_subgroup] = census_subgroup

    static_features.update(census_age_dict)

    return static_features

  def extract_all_features(
      self,
      static_data,
      ts_data,
      locations,
      training_window_size,
  ):
    (static_features, static_scalers), (ts_features, ts_scalers) = (
        super().extract_all_features(static_data, ts_data, locations,
                                     training_window_size))
    if generic_seir_specs_county.INCLUDE_AGE_AS_TIME_COVARIATE:
      if static_data is None or ts_data is None:
        raise ValueError("Both static and time series data must be provided.")
      all_dates = preprocessing.get_all_valid_dates(ts_data)
      preprocessing.convert_static_features_to_constant_ts(
          static_features, static_scalers, ts_features, ts_scalers,
          constants.COUNTY_POP_AGE_RANGES.keys(), len(all_dates))
    return (static_features, static_scalers), (ts_features, ts_scalers)

  def transform_static_features(
      self, static_features
  ):
    # Add population age ranges. These are currently NOT preprocessed because
    # they are not returned by `get_static_features()`.
    preprocessing.create_population_age_ranges(static_features)

    transformed_features, feature_scalers = super().transform_static_features(
        static_features)

    transformed_features[constants.POPULATION_60P_RATIO], feature_scalers[
        constants.POPULATION_60P_RATIO] = self._preprocess_static_feature(
            preprocessing.normalize_static_feature(
                static_features[constants.KAISER_60P_POPULATION],
                static_features[constants.KAISER_POPULATION]))
    transformed_features[constants.HOUSEHOLD_FOOD_STAMP], feature_scalers[
        constants.HOUSEHOLD_FOOD_STAMP] = self._preprocess_static_feature(
            preprocessing.normalize_static_feature(
                static_features[constants.HOUSEHOLD_FOOD_STAMP],
                static_features[constants.POPULATION]))
    transformed_features[constants.HOSPITAL_RATING_AVERAGE], feature_scalers[
        constants.HOSPITAL_RATING_AVERAGE] = self._preprocess_static_feature(
            _calculate_hospital_rating_average(static_features))

    return transformed_features, feature_scalers

  def get_all_locations(self, input_df):
    all_locations = super().get_all_locations(input_df)
    if "15005" in all_locations:
      # Exclude FIPS 15005 (Kalawao County, no longer exist)
      all_locations.remove("15005")
    return all_locations

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
      return generic_seir_specs_county.get_model_specs(
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
    return generic_seir_county_model_constructor.CountyModelConstructor(
        model_spec=model_spec, random_seed=random_seed)

  def bound_variables(
      self,
      seir_timeseries_variables,
  ):
    """See parent class."""

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
    diagnosis_rate = 0.01 + 0.09 * tf.nn.sigmoid(diagnosis_rate_list[-1])
    recovery_rate_id = 0.1 * tf.nn.sigmoid(recovery_rate_id_list[-1])
    recovery_rate_iud = 0.1 * tf.nn.sigmoid(recovery_rate_iud_list[-1])
    recovery_rate_h = 0.005 + 0.095 * tf.nn.sigmoid(recovery_rate_h_list[-1])
    hospitalization_rate = 0.005 + 0.095 * tf.nn.sigmoid(
        hospitalization_rate_list[-1])
    death_rate_id = 0.01 * tf.nn.sigmoid(death_rate_id_list[-1])
    death_rate_h = 0.1 * tf.nn.sigmoid(death_rate_h_list[-1])

    return (vaccinated_ratio, vaccine_effectiveness, average_contact_id,
            average_contact_iud, reinfectable_rate, alpha, diagnosis_rate,
            recovery_rate_id, recovery_rate_iud, recovery_rate_h,
            hospitalization_rate, death_rate_id, death_rate_h)


class StateModelDefinition(
    compartmental_model_definitions.BaseSeirIcuVentilatorModelDefinition):
  """The US State level compartmental model.

  Attributes:
    ts_preprocessing_config: The default configuration to use for pre-processing
      time-series features.
    static_preprocessing_config: The default configuration to use for
      pre-processing static features.
    random_seed: A number to be used as a random seed for the model.
    gt_source: The ground truth used.
  """

  def __init__(
      self,
      ts_preprocessing_config = None,
      static_preprocessing_config = None,
      random_seed = 0,
      gt_source = "",
      **kwargs,
  ):
    """Creates the compartmental model.

    Args:
      ts_preprocessing_config: The default configuration to use for
        pre-processing time-series features.
      static_preprocessing_config: The default configuration to use for
        pre-processing static features.
      random_seed: A random seed to use in the model.
      gt_source: The ground truth source to be used for this model (e.g. JHU).
      **kwargs: Model specific keyword arguments.
    """
    super().__init__(ts_preprocessing_config, static_preprocessing_config,
                     random_seed, **kwargs)

    if not gt_source:
      raise ValueError("The gt_source must be specified.")
    self.gt_source = gt_source

  def get_ts_features(self):
    gt_key_death, gt_key_confirmed = get_gt_source_features(
        "STATE", self.gt_source)
    return {
        constants.DEATH:
            gt_key_death,
        constants.CONFIRMED:
            gt_key_confirmed,
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

  def get_ts_features_to_preprocess(self):
    return {
        constants.MOBILITY_INDEX,
        constants.MOBILITY_SAMPLES,
        constants.TOTAL_TESTS,
        constants.TOTAL_TESTS_PER_CAPITA,
        constants.CONFIRMED_PER_TESTS,
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

  def transform_ts_features(
      self,
      ts_features,
      static_features,
      initial_train_window_size,
  ):
    transformed_features, feature_scalers = super().transform_ts_features(
        ts_features, static_features, initial_train_window_size)

    transformed_features[constants.CONFIRMED_PER_TESTS], feature_scalers[
        constants.CONFIRMED_PER_TESTS] = (
            self._preprocess_ts_feature(
                preprocessing.normalize_ts_feature(
                    ts_features[constants.CONFIRMED],
                    ts_features[constants.TOTAL_TESTS],
                    epsilon=1.0,
                    upper_limit=1.0,
                    lower_limit=0.0), initial_train_window_size))

    transformed_features[constants.TOTAL_TESTS_PER_CAPITA], feature_scalers[
        constants.TOTAL_TESTS_PER_CAPITA] = (
            self._preprocess_ts_feature(
                preprocessing.normalize_ts_feature(
                    ts_features[constants.TOTAL_TESTS],
                    static_features[constants.POPULATION],
                    epsilon=1.0,
                    upper_limit=3.0,
                    lower_limit=0.0),
                initial_train_window_size,
                initial_value=0,
            ))

    transformed_features[constants.ANTIGEN_POSITIVE_RATIO], feature_scalers[
        constants.ANTIGEN_POSITIVE_RATIO] = (
            self._preprocess_ts_feature(
                preprocessing.normalize_ts_feature(
                    ts_features[constants.ANTIGEN_POSITIVE],
                    ts_features[constants.ANTIGEN_TOTAL],
                    epsilon=1.0,
                    upper_limit=1.0,
                    lower_limit=0.0), initial_train_window_size))

    transformed_features[constants.ANTIBODY_NEGATIVE_RATIO], feature_scalers[
        constants.ANTIBODY_NEGATIVE_RATIO] = (
            self._preprocess_ts_feature(
                preprocessing.normalize_ts_feature(
                    ts_features[constants.ANTIBODY_NEGATIVE],
                    ts_features[constants.ANTIBODY_TOTAL],
                    epsilon=1.0,
                    upper_limit=1.0,
                    lower_limit=0.0), initial_train_window_size))

    # 'HOSPITALIZED_CUMULATIVE' is the cumulative sum of 'HOSPITALIZED_INCREASE'
    transformed_features[constants.HOSPITALIZED_CUMULATIVE] = (
        preprocessing.cumulative_sum_ts_feature(
            ts_feature=ts_features[constants.HOSPITALIZED_INCREASE],
            initial_value=0))
    feature_scalers[constants.HOSPITALIZED_CUMULATIVE] = None

    # Cumulative to current values
    first_dose_vaccine_current_per_day = ts_features[
        constants.VACCINES_GOVEX_FIRST_DOSE_TOTAL].copy()
    for location_name in first_dose_vaccine_current_per_day.keys():
      first_dose_vaccine_current_per_day[location_name][1:] = np.diff(
          first_dose_vaccine_current_per_day[location_name])

    transformed_features[
        constants
        .VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED], feature_scalers[
            constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED] = (
                self._preprocess_ts_feature(
                    preprocessing.normalize_ts_feature(
                        first_dose_vaccine_current_per_day,
                        static_features[constants.POPULATION],
                        epsilon=1.0,
                        upper_limit=1.0,
                        lower_limit=0.0),
                    initial_train_window_size,
                    bfill_features=False,
                    imputation_strategy="constant",
                    standardize=False,
                    initial_value=0,
                ))

    # Cumulative to current values
    second_dose_vaccine_current_per_day = ts_features[
        constants.VACCINES_GOVEX_SECOND_DOSE_TOTAL].copy()
    for location_name in second_dose_vaccine_current_per_day.keys():
      second_dose_vaccine_current_per_day[location_name][1:] = np.diff(
          second_dose_vaccine_current_per_day[location_name])

    transformed_features[
        constants
        .VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED], feature_scalers[
            constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED] = (
                self._preprocess_ts_feature(
                    preprocessing.normalize_ts_feature(
                        second_dose_vaccine_current_per_day,
                        static_features[constants.POPULATION],
                        epsilon=1.0,
                        upper_limit=1.0,
                        lower_limit=0.0),
                    initial_train_window_size,
                    bfill_features=False,
                    imputation_strategy="constant",
                    standardize=False,
                    initial_value=0,
                ))

    transformed_features[constants.INFECTED], feature_scalers[
        constants.INFECTED] = None, None
    return transformed_features, feature_scalers

  def get_static_features(self):
    return {
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

  def transform_static_features(
      self, static_features
  ):
    transformed_features, feature_scalers = super().transform_static_features(
        static_features)

    transformed_features[constants.POPULATION_60P_RATIO], feature_scalers[
        constants.POPULATION_60P_RATIO] = self._preprocess_static_feature(
            preprocessing.normalize_static_feature(
                static_features[constants.KAISER_60P_POPULATION],
                static_features[constants.KAISER_POPULATION]))
    transformed_features[constants.HOUSEHOLD_FOOD_STAMP], feature_scalers[
        constants.HOUSEHOLD_FOOD_STAMP] = self._preprocess_static_feature(
            preprocessing.normalize_static_feature(
                static_features[constants.HOUSEHOLD_FOOD_STAMP],
                static_features[constants.POPULATION]))
    transformed_features[constants.HOSPITAL_RATING_AVERAGE], feature_scalers[
        constants.HOSPITAL_RATING_AVERAGE] = self._preprocess_static_feature(
            _calculate_hospital_rating_average(static_features))

    return transformed_features, feature_scalers

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
      return generic_seir_specs_state.get_model_specs(
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
    return generic_seir_state_model_constructor.StateModelConstructor(
        model_spec=model_spec, random_seed=random_seed)


def _calculate_hospital_rating_average(
    static_features,
    epsilon = 1e-8,
):
  """Calculates the average rating for hospitals in each location.

  Args:
    static_features: Map of all features for all locations.
    epsilon: A small value to avoid dividing by zero.

  Returns:
    Map of locations with average hospital rating or None.
  """
  rating_average = {}
  all_locations = static_features[constants.HOSPITAL_RATING1].keys()
  for location in all_locations:
    try:
      rating_average[location] = (
          (1 * static_features[constants.HOSPITAL_RATING1][location] +
           2 * static_features[constants.HOSPITAL_RATING2][location] +
           3 * static_features[constants.HOSPITAL_RATING3][location] +
           4 * static_features[constants.HOSPITAL_RATING4][location] +
           5 * static_features[constants.HOSPITAL_RATING5][location]) /
          (static_features[constants.HOSPITAL_RATING1][location] +
           static_features[constants.HOSPITAL_RATING2][location] +
           static_features[constants.HOSPITAL_RATING3][location] +
           static_features[constants.HOSPITAL_RATING4][location] +
           static_features[constants.HOSPITAL_RATING5][location] + epsilon))
    except TypeError:
      rating_average[location] = None

  return rating_average

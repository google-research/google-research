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

# Lint as: python3
"""Encoder specifications and hyperparameters used for county-level models.

Each encoder outputs the variables of the corresponding county-level
compartmental model.
"""
from typing import List

from covid_epidemiology.src import constants
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib

# Static feature candidates
population_density_per_sq_km = model_spec_lib.FeatureSpec(
    name=constants.POPULATION_DENSITY_PER_SQKM, initializer=None)
income_per_capita = model_spec_lib.FeatureSpec(
    name=constants.INCOME_PER_CAPITA, initializer=None)
household_food_stamp = model_spec_lib.FeatureSpec(
    name=constants.HOUSEHOLD_FOOD_STAMP, initializer=None)
household = model_spec_lib.FeatureSpec(
    name=constants.HOUSEHOLDS, initializer=None)
kaiser_60plus_population = model_spec_lib.FeatureSpec(
    name=constants.KAISER_60P_POPULATION, initializer=None)
kaiser_population = model_spec_lib.FeatureSpec(
    name=constants.KAISER_POPULATION, initializer=None)
population_60plus_ratio = model_spec_lib.FeatureSpec(
    name=constants.POPULATION_60P_RATIO, initializer=None)
patient_experience_above_the_national_average = model_spec_lib.FeatureSpec(
    name=constants.PATIENCE_EXPERIENCE_ABOVE, initializer=None)
patient_experience_below_the_national_average = model_spec_lib.FeatureSpec(
    name=constants.PATIENCE_EXPERIENCE_BELOW, initializer=None)
patient_experience_same_as_the_national_average = model_spec_lib.FeatureSpec(
    name=constants.PATIENCE_EXPERIENCE_SAME, initializer=None)
aqi_mean_2018 = model_spec_lib.FeatureSpec(
    name=constants.AQI_MEAN, initializer=None)
rating_average = model_spec_lib.FeatureSpec(
    name=constants.HOSPITAL_RATING_AVERAGE, initializer=None)
icu_beds = model_spec_lib.FeatureSpec(name=constants.ICU_BEDS, initializer=None)

# Time-varying feature candidates
preprocessed_confirmed = model_spec_lib.FeatureSpec(
    name=constants.CONFIRMED_PREPROCESSED,
    initializer=None,
    forecast_method=model_spec_lib.ForecastMethod.NONE)
preprocessed_death = model_spec_lib.FeatureSpec(
    name=constants.DEATH_PREPROCESSED,
    initializer=None,
    forecast_method=model_spec_lib.ForecastMethod.NONE)
preprocessed_confirmed_mean_to_sum = model_spec_lib.FeatureSpec(
    name=constants.CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO,
    initializer=None,
    forecast_method=model_spec_lib.ForecastMethod.NONE)
preprocessed_death_mean_to_sum = model_spec_lib.FeatureSpec(
    name=constants.DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO,
    initializer=None,
    forecast_method=model_spec_lib.ForecastMethod.NONE)
mobility_index = model_spec_lib.FeatureSpec(
    name=constants.MOBILITY_INDEX,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.POSITIVE)
mobility_samples = model_spec_lib.FeatureSpec(
    name=constants.MOBILITY_SAMPLES, initializer=None)
csrp_tests = model_spec_lib.FeatureSpec(
    name=constants.CSRP_TESTS, initializer=None)
confirmed_per_csrp_tests = model_spec_lib.FeatureSpec(
    name=constants.CONFIRMED_PER_CSRP_TESTS, initializer=None)
total_tests_per_capita = model_spec_lib.FeatureSpec(
    name=constants.TOTAL_TESTS_PER_CAPITA, initializer=None)
first_dose_vaccine_ratio_per_day = model_spec_lib.FeatureSpec(
    name=constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED,
    initializer=None)
second_dose_vaccine_ratio_per_day = model_spec_lib.FeatureSpec(
    name=constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED,
    initializer=None)
first_dose_vaccine_effectiveness = model_spec_lib.FeatureSpec(
    name=constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE, initializer=None)
second_dose_vaccine_effectiveness = model_spec_lib.FeatureSpec(
    name=constants.VACCINATED_EFFECTIVENESS_SECOND_DOSE, initializer=None)
dow_feature = model_spec_lib.FeatureSpec(
    name=constants.DOW_WINDOW,
    initializer=None,
    forecast_method=model_spec_lib.ForecastMethod.PERIODIC_WEEKLY,
    apply_lasso=True,
)

county_age_features: List[model_spec_lib.FeatureSpec] = [
    model_spec_lib.FeatureSpec(age_group_name, initializer=None)
    for age_group_name in constants.COUNTY_POP_AGE_RANGES
]

# NPI Features
restaurants = model_spec_lib.FeatureSpec(
    name=constants.AMP_RESTAURANTS,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)
nonessential_business = model_spec_lib.FeatureSpec(
    name=constants.AMP_NON_ESSENTIAL_BUSINESS,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)
stay_at_home = model_spec_lib.FeatureSpec(
    name=constants.AMP_STAY_AT_HOME,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)
higher_education = model_spec_lib.FeatureSpec(
    name=constants.AMP_SCHOOLS_SECONDARY_EDUCATION,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)
emergency = model_spec_lib.FeatureSpec(
    name=constants.AMP_EMERGENCY_DECLARATION,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)
gather = model_spec_lib.FeatureSpec(
    name=constants.AMP_GATHERINGS,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)
masks = model_spec_lib.FeatureSpec(
    name=constants.AMP_FACE_MASKS,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE,
)

npi_features: List[model_spec_lib.FeatureSpec] = [
    restaurants,
    nonessential_business,
    stay_at_home,
    higher_education,
    emergency,
    gather,
    masks,
]

INCLUDE_AGE_AS_TIME_COVARIATE = True

if INCLUDE_AGE_AS_TIME_COVARIATE:
  extra_static_features = []
  extra_covariates = county_age_features
else:
  extra_static_features = county_age_features
  extra_covariates = []

DEFAULT_WINDOW_SIZE = 7
USE_FIXED_COVARIATE_MASK = True


def get_model_spec_time_varying_with_covariates(covariate_delay):
  """Return model specification with hyperparameters and covariates."""
  result = model_spec_lib.ModelSpec(
      hparams={
          "initial_learning_rate": 0.003061242915556316,
          "momentum": 0.2,
          "decay_steps": 1000,
          "fine_tuning_steps": 100,
          "fine_tuning_decay": 1.0,
          "decay_rate": 1.0,
          "location_dependent_init": False,
          "infected_threshold": 3,
          "restart_threshold": 1000,
          "time_scale_weight": 0.00006243159539906051,
          "train_loss_coefs": [0, 0.001, 0.2, 0.1, 0.01, 0.01, 0.2, 0.1, 0.01],
          "valid_loss_coefs": [0, 0.001, 0.2, 0.1, 0.01, 0.01, 0.2, 0.1, 0.01],
          "sync_coef": 1.0,
          "reduced_sync_undoc": 1.0,
          "smooth_coef": 0.5,
          "first_dose_vaccine_ratio_per_day_init": 0.0,
          "second_dose_vaccine_ratio_per_day_init": 0.0,
          "average_contact_id_rate_init": -1.9131825459930378,
          "average_contact_iud_rate_init": -1.071866945725303,
          "reinfectable_rate_init": -5.548940468292865,
          "alpha_rate_init": -2.272765554778715,
          "diagnosis_rate_init": -2.095597433974376,
          "recovery_id_rate_init": -1.495660223962899,
          "recovery_iud_rate_init": -1.475605314236803,
          "recovery_h_rate_init": -1.9032896753850963,
          "hospitalization_rate_init": -1.4331763640928012,
          "death_id_rate_init": -1.8060447968974489,
          "death_h_rate_init": -1.4886876719378206,
          "bias_penalty_coef": 0.2835406167308398,
          "r_eff_penalty_coef": 2.0,
          "acceleration_death_coef": 0.1,
          "acceleration_confirm_coef": 0.1,
          "acceleration_hospital_coef": 0.1,
          "quantile_encoding_window": 7,
          "quantile_smooth_coef": 0.9,
          "quantile_training_iteration_ratio": 0.5,
          "width_coef_train": 10.0,
          "width_coef_valid": 5.0,
          "quantile_cum_viol_coef": 500.0,
          "increment_loss_weight": 0.0,
          "lasso_penalty_coef": 1.0,
          "covariate_training_mixing_coef": 1.0,
          "train_window_range": 2.0,
          "partial_mean_interval": 4,
          "direction_loss_coef": 2000.0,
          "train_crps_weight": 0.25,
          "valid_crps_weight": 0.25
      },
      encoder_specs=[
          model_spec_lib.EncoderSpec(
              encoder_name="first_dose_vaccine_ratio_per_day",
              encoder_type="vaccine",
              vaccine_type="first_dose",
              covariate_feature_specs=[
                  first_dose_vaccine_ratio_per_day,
                  first_dose_vaccine_effectiveness,
                  second_dose_vaccine_effectiveness
              ],
          ),
          model_spec_lib.EncoderSpec(
              encoder_name="second_dose_vaccine_ratio_per_day",
              encoder_type="vaccine",
              vaccine_type="second_dose",
              covariate_feature_specs=[
                  second_dose_vaccine_ratio_per_day,
                  first_dose_vaccine_effectiveness,
                  second_dose_vaccine_effectiveness
              ],
          ),
          model_spec_lib.EncoderSpec(
              encoder_name="average_contact_id_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km,
                  income_per_capita,
                  aqi_mean_2018,
                  household,
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed, preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, mobility_index,
                  mobility_samples, confirmed_per_csrp_tests
              ] + extra_covariates + npi_features,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="average_contact_iud_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km,
                  income_per_capita,
                  aqi_mean_2018,
                  household,
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed, preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, mobility_index,
                  mobility_samples, confirmed_per_csrp_tests
              ] + extra_covariates + npi_features,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="reinfectable_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population
              ] + extra_static_features,
              covariate_feature_specs=None,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="alpha_rate",
              encoder_type="gam",
              static_feature_specs=[],
              covariate_feature_specs=None,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="diagnosis_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  confirmed_per_csrp_tests,
                  total_tests_per_capita,
                  dow_feature,
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="recovery_id_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="recovery_iud_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed, preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="recovery_h_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population,
                  icu_beds
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="hospitalization_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
                  confirmed_per_csrp_tests,
                  total_tests_per_capita,
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="death_id_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population,
                  icu_beds
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="death_h_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  rating_average, aqi_mean_2018, kaiser_60plus_population,
                  icu_beds
              ] + extra_static_features,
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ] + extra_covariates,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
      ])
  return result


STATIC_MODEL = model_spec_lib.ModelSpec(
    hparams={
        "initial_learning_rate": 0.001338395965177607,
        "decay_steps": 5,
        "decay_rate": 0.99,
        "fine_tuning_steps": 1000,
        "location_dependent_init": True,
        "infected_threshold": 10,
        "restart_threshold": 300,
        "time_scale_weight": 0.000021644778644786284,
        "train_loss_coefs": [0, 0, 1, 0.0011310082312404356],
        "valid_loss_coefs": [0, 0, 1, 0],
        "sync_coef": 0.5,
        "first_dose_vaccine_ratio_per_day_init": 0.0,
        "second_dose_vaccine_ratio_per_day_init": 0.0,
        "first_dose_vaccine_effectiveness_init": 0.0,
        "second_dose_vaccine_effectiveness_init": 0.0,
        "average_contact_id_rate_init": -2.4387148524945594,
        "average_contact_iud_rate_init": -1.561117964142531,
        "reinfectable_rate_init": -5.447821316098564,
        "alpha_rate_init": -1.1906569150841058,
        "diagnosis_rate_init": -1.589940307765024,
        "recovery_id_rate_init": -1.5723201455595701,
        "recovery_iud_rate_init": -1.8295556397198884,
        "recovery_h_rate_init": -1.8295556397198884,
        "death_id_rate_init": -1.0766994750650696,
        "death_h_rate_init": -1.0766994750650696,
        "compartmental_penalty_coef": 96.12757205042173,
        "variable_smoothness_penalty_coef": 5.988152939976956,
        "bias_penalty_coef": 0.27315803186400106,
        "boundary_penalty_coef": 2.600265396382176,
        "lasso_penalty_coef": 1.0,
        "r_eff_penalty_coef": 1.0,
        "quantile_encoding_window": 7,
        "quantile_smooth_coef": 0.8,
        "increment_loss_weight": 0.0,
        "covariate_training_mixing_coef": 1.0,
        "direction_loss_coef": 2000.0
    },
    encoder_specs=[
        model_spec_lib.EncoderSpec(
            encoder_name="first_dose_vaccine_ratio_per_day",
            encoder_type="vaccine",
            vaccine_type="first_dose",
            covariate_feature_specs=[
                first_dose_vaccine_ratio_per_day,
                first_dose_vaccine_effectiveness,
                second_dose_vaccine_effectiveness
            ],
        ),
        model_spec_lib.EncoderSpec(
            encoder_name="second_dose_vaccine_ratio_per_day",
            encoder_type="vaccine",
            vaccine_type="second_dose",
            covariate_feature_specs=[
                second_dose_vaccine_ratio_per_day,
                first_dose_vaccine_effectiveness,
                second_dose_vaccine_effectiveness
            ],
        ),
        model_spec_lib.EncoderSpec(
            encoder_name="first_dose_vaccine_effectiveness",
            encoder_type="passthrough",
            covariate_feature_specs=[first_dose_vaccine_effectiveness],
        ),
        model_spec_lib.EncoderSpec(
            encoder_name="second_dose_vaccine_effectiveness",
            encoder_type="passthrough",
            covariate_feature_specs=[second_dose_vaccine_effectiveness],
        ),
        model_spec_lib.EncoderSpec(
            encoder_name="average_contact_id_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="average_contact_iud_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="reinfectable_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="alpha_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="diagnosis_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="recovery_id_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="recovery_iud_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="recovery_h_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_id_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_h_rate", encoder_type="static")
    ])


def get_model_specs(covariate_delay):
  result = {
      "TIME_VARYING_WITH_COVARIATES":
          get_model_spec_time_varying_with_covariates(covariate_delay),
      "STATIC_SEIR":
          STATIC_MODEL,
      "TREND_FOLLOWING":
          get_model_spec_time_varying_with_covariates(covariate_delay),
  }
  return result

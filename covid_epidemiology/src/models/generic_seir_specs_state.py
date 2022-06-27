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

"""Encoder specifications and hyperparameters used for state-level models.

Each encoder outputs the variables of the corresponding state-level
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
icu_beds = model_spec_lib.FeatureSpec(name=constants.ICU_BEDS, initializer=None)
patient_experience_above_the_national_average = model_spec_lib.FeatureSpec(
    name=constants.PATIENCE_EXPERIENCE_ABOVE, initializer=None)
patient_experience_below_the_national_average = model_spec_lib.FeatureSpec(
    name=constants.PATIENCE_EXPERIENCE_BELOW, initializer=None)
patient_experience_same_as_the_national_average = model_spec_lib.FeatureSpec(
    name=constants.PATIENCE_EXPERIENCE_SAME, initializer=None)
hospital_type_critical_access_hospitals = model_spec_lib.FeatureSpec(
    name=constants.CRITICAL_ACCESS_HOSPITAL, initializer=None)
hospital_type_acute_care_department_of_defense = model_spec_lib.FeatureSpec(
    name=constants.HOSPITAL_ACUTE_CARE, initializer=None)
service_type_emergency_services_supported = model_spec_lib.FeatureSpec(
    name=constants.EMERGENCY_SERVICES, initializer=None)
service_type_non_emergency_services = model_spec_lib.FeatureSpec(
    name=constants.NON_EMERGENCY_SERVICES, initializer=None)
aqi_mean_2018 = model_spec_lib.FeatureSpec(
    name=constants.AQI_MEAN, initializer=None)
rating_average = model_spec_lib.FeatureSpec(
    name=constants.HOSPITAL_RATING_AVERAGE, initializer=None)

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

total_tests = model_spec_lib.FeatureSpec(
    name=constants.TOTAL_TESTS, initializer=None)
total_tests_per_capita = model_spec_lib.FeatureSpec(
    name=constants.TOTAL_TESTS_PER_CAPITA, initializer=None)
confirmed_per_total_tests = model_spec_lib.FeatureSpec(
    name=constants.CONFIRMED_PER_TESTS, initializer=None)
google_mobility_parks = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_PARKS, initializer=None)
google_mobility_work = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_WORK, initializer=None)
google_mobility_res = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_RES, initializer=None)
google_mobility_transit = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_TRANSIT, initializer=None)
google_mobility_grocery = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_GROCERY, initializer=None)
google_mobility_retail = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_RETAIL, initializer=None)
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
average_temperature = model_spec_lib.FeatureSpec(
    name=constants.AVERAGE_TEMPERATURE, initializer=None)
rainfall = model_spec_lib.FeatureSpec(name=constants.RAINFALL, initializer=None)
snowfall = model_spec_lib.FeatureSpec(name=constants.SNOWFALL, initializer=None)
commercial_score = model_spec_lib.FeatureSpec(
    name=constants.COMMERCIAL_SCORE, initializer=None)
antigen_positive_ratio = model_spec_lib.FeatureSpec(
    name=constants.ANTIGEN_POSITIVE_RATIO, initializer=None)
antibody_negative_ratio = model_spec_lib.FeatureSpec(
    name=constants.ANTIBODY_NEGATIVE_RATIO, initializer=None)

symptom_cough = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_COUGH, initializer=None, apply_lasso=False)
symptom_chills = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_CHILLS, initializer=None, apply_lasso=False)
symptom_anosmia = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_ANOSMIA, initializer=None, apply_lasso=False)
symptom_infection = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_INFECTION, initializer=None, apply_lasso=False)
symptom_chestpain = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_CHEST_PAIN, initializer=None, apply_lasso=False)
symptom_fever = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_FEVER, initializer=None, apply_lasso=False)
symptom_shortnessbreath = model_spec_lib.FeatureSpec(
    name=constants.SYMPTOM_SHORTNESSBREATH, initializer=None, apply_lasso=False)
symptoms_all = [
    symptom_cough, symptom_chills, symptom_anosmia, symptom_infection,
    symptom_chestpain, symptom_fever, symptom_shortnessbreath
]

DEFAULT_WINDOW_SIZE = 7
USE_FIXED_COVARIATE_MASK = True


def get_model_spec_time_varying_with_covariates(covariate_delay):
  """Return model specification with hyperparameters and covariates."""
  result = model_spec_lib.ModelSpec(
      hparams={
          "initial_learning_rate": 0.003778327764151733,
          "momentum": 0.1,
          "decay_steps": 1000,
          "fine_tuning_steps": 100,
          "fine_tuning_decay": 1.0,
          "decay_rate": 1.0,
          "location_dependent_init": False,
          "infected_threshold": 10,
          "restart_threshold": 1000,
          "time_scale_weight": 0.00006243159539906051,
          "train_loss_coefs": [0, 0.001, 1.0, 0.1, 0.05, 0.05, 0.01, 0.001],
          "valid_loss_coefs": [0, 0.001, 1.0, 0.1, 0.05, 0.05, 0.01, 0.001],
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
          "recovery_i_rate_init": -1.354652366086861,
          "recovery_v_rate_init": -1.2872596880548643,
          "hospitalization_rate_init": -1.4331763640928012,
          "icu_rate_init": -1.7697853537123167,
          "ventilator_rate_init": -1.747746088542831,
          "death_id_rate_init": -1.8060447968974489,
          "death_h_rate_init": -1.4886876719378206,
          "death_i_rate_init": -1.525990911868795,
          "death_v_rate_init": -1.0072190496934017,
          "bias_penalty_coef": 0.2835406167308398,
          "r_eff_penalty_coef": 2.0,
          "r_eff_penalty_cutoff": 5.5,
          "acceleration_death_coef": 0.1,
          "acceleration_confirm_coef": 0.1,
          "acceleration_hospital_coef": 0.1,
          "quantile_encoding_window": 7,
          "quantile_smooth_coef": 0.9,
          "quantile_training_iteration_ratio": 1.0,
          "width_coef_train": 1.0,
          "width_coef_valid": 1.0,
          "quantile_cum_viol_coef": 1000.0,
          "increment_loss_weight": 0.0,
          "lasso_penalty_coef": 1.0,
          "covariate_training_mixing_coef": 1.0,
          "direction_loss_coef": 15000.0,
          "train_window_range": 2.0,
          "partial_mean_interval": 4,
          "train_crps_weight": 0.5,
          "valid_crps_weight": 0.5
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
                  population_density_per_sq_km, income_per_capita,
                  aqi_mean_2018, household, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed, preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, mobility_index,
                  mobility_samples, average_temperature, rainfall, snowfall,
                  antigen_positive_ratio, antibody_negative_ratio,
                  confirmed_per_total_tests
              ] + npi_features,
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
                  population_density_per_sq_km, income_per_capita,
                  aqi_mean_2018, household, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed, preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, mobility_index,
                  mobility_samples, average_temperature, rainfall, snowfall,
                  antigen_positive_ratio, antibody_negative_ratio,
                  confirmed_per_total_tests
              ] + npi_features,
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
                  population_density_per_sq_km, income_per_capita,
                  household_food_stamp, household,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, population_60plus_ratio
              ],
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
                  population_density_per_sq_km, income_per_capita, household,
                  rating_average, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  total_tests_per_capita,
                  confirmed_per_total_tests,
                  dow_feature,
                  antigen_positive_ratio,
                  antibody_negative_ratio,
                  average_temperature,
                  rainfall,
                  snowfall,
              ] + symptoms_all,
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
                  population_density_per_sq_km, household, income_per_capita,
                  household_food_stamp,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
                  antigen_positive_ratio,
                  antibody_negative_ratio,
                  average_temperature,
                  rainfall,
                  snowfall,
              ],
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
                  population_density_per_sq_km, household, income_per_capita,
                  household_food_stamp,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  antigen_positive_ratio,
                  antibody_negative_ratio,
                  average_temperature,
                  rainfall,
                  snowfall,
              ],
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
                  population_density_per_sq_km, household, income_per_capita,
                  household_food_stamp,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="recovery_i_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, income_per_capita,
                  household_food_stamp,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="recovery_v_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, income_per_capita,
                  household_food_stamp,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
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
                  population_density_per_sq_km,
                  household,
                  household_food_stamp,
                  household,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services,
                  rating_average,
                  aqi_mean_2018,
                  icu_beds,
                  population_60plus_ratio,
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed, preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, dow_feature,
                  total_tests_per_capita, confirmed_per_total_tests,
                  antigen_positive_ratio, antibody_negative_ratio,
                  average_temperature, rainfall, snowfall
              ] + symptoms_all,
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="icu_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  household, income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, icu_beds, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="ventilator_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  household, income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, icu_beds, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
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
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, icu_beds, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
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
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, icu_beds, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="death_i_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, icu_beds, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
              covariate_feature_time_offset=covariate_delay,
              covariate_feature_window=DEFAULT_WINDOW_SIZE,
              encoder_kwargs={
                  "initial_bias": 0,
                  "location_dependent_bias": True,
                  "use_fixed_covariate_mask": USE_FIXED_COVARIATE_MASK,
              }),
          model_spec_lib.EncoderSpec(
              encoder_name="death_v_rate",
              encoder_type="gam",
              static_feature_specs=[
                  population_density_per_sq_km, household, household_food_stamp,
                  income_per_capita,
                  patient_experience_above_the_national_average,
                  patient_experience_below_the_national_average,
                  patient_experience_same_as_the_national_average,
                  hospital_type_critical_access_hospitals,
                  hospital_type_acute_care_department_of_defense,
                  service_type_emergency_services_supported,
                  service_type_non_emergency_services, rating_average,
                  aqi_mean_2018, icu_beds, population_60plus_ratio
              ],
              covariate_feature_specs=[
                  preprocessed_confirmed,
                  preprocessed_death,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ],
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
        "initial_learning_rate": 0.003046320146869829,
        "decay_steps": 5,
        "decay_rate": 1.0,
        "fine_tuning_steps": 1000,
        "location_dependent_init": True,
        "infected_threshold": 10,
        "restart_threshold": 300,
        "time_scale_weight": 0.00007493264308148026,
        "train_loss_coefs": [0, 0, 1, 0.001131, 1e-7, 1, 0.0045],
        "valid_loss_coefs": [0, 0, 1, 0.0, 0.0, 0.0, 0.0],
        "sync_coef": 0.5,
        "first_dose_vaccine_ratio_per_day_init": 0.0,
        "second_dose_vaccine_ratio_per_day_init": 0.0,
        "average_contact_id_rate_init": -2.4387148524945594,
        "average_contact_iud_rate_init": -1.561117964142531,
        "reinfectable_rate_init": -5.447821316098564,
        "alpha_rate_init": -1.1906569150841058,
        "diagnosis_rate_init": -1.589940307765024,
        "recovery_id_rate_init": -1.5723201455595701,
        "recovery_iud_rate_init": -1.8295556397198884,
        "recovery_h_rate_init": -1.828568757553072,
        "recovery_i_rate_init": -1.5923711129624136,
        "recovery_v_rate_init": -1.7109958618683396,
        "hospitalization_rate_init": -1.5327844461092095,
        "icu_rate_init": -1.6822889372036114,
        "ventilator_rate_init": -1.6721172663784372,
        "death_id_rate_init": -1.0766994750650696,
        "death_h_rate_init": -1.3775460164801863,
        "death_i_rate_init": -1.8531814769372583,
        "death_v_rate_init": -1.3251172286148205,
        "compartmental_penalty_coef": 96.12757205042173,
        "variable_smoothness_penalty_coef": 5.988152939976956,
        "bias_penalty_coef": 0.27315803186400106,
        "boundary_penalty_coef": 2.600265396382176,
        "lasso_penalty_coef": 1.0,
        "quantile_encoding_window": 7,
        "quantile_smooth_coef": 0.8,
        "width_coef": 5.0,
        "increment_loss_weight": 0.0,
        "covariate_training_mixing_coef": 1.0,
        "direction_loss_coef": 20000.0
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
            encoder_name="recovery_i_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="recovery_v_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="hospitalization_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="icu_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="ventilator_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_id_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_h_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_i_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_v_rate", encoder_type="static")
    ])


def get_model_specs(covariate_delay):
  model_spec = {
      "TIME_VARYING_WITH_COVARIATES":
          get_model_spec_time_varying_with_covariates(covariate_delay),
      "STATIC_SEIR":
          STATIC_MODEL,
      "TREND_FOLLOWING":
          get_model_spec_time_varying_with_covariates(covariate_delay),
  }
  return model_spec

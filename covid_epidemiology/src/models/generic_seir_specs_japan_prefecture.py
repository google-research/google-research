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
"""Encoder specifications and hyperparameters used for Japan prefecture models.

Each encoder outputs the variables of the corresponding Japan
compartmental model.
"""

from covid_epidemiology.src import constants
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib


# Static feature candidates
density = model_spec_lib.FeatureSpec(name=constants.DENSITY, initializer=None)
ages_0_to_14 = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_AGE_0_TO_14_FEATURE_KEY, initializer=None)
ages_15_to_64 = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_AGE_15_TO_64_FEATURE_KEY, initializer=None)
ages_64_plus = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_AGE_64_PLUS_FEATURE_KEY, initializer=None)
ages_75_plus = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_AGE_75_PLUS_FEATURE_KEY, initializer=None)
gdp_per_capita = model_spec_lib.FeatureSpec(
    name=constants.INCOME_PER_CAPITA, initializer=None)
demographics = [
    density, ages_0_to_14, ages_15_to_64, ages_64_plus, ages_75_plus,
    gdp_per_capita
]

doctors = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_NUM_DOCTORS_FEATURE_KEY, initializer=None)
doctors_per_100k = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_DOCTORS_PER_100K_FEATURE_KEY,
    initializer=None)
hospital_beds = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_FEATURE_KEY,
    initializer=None)
hospital_beds_per_100k = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_PER_100K_FEATURE_KEY,
    initializer=None)
clinic_beds = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_FEATURE_KEY,
    initializer=None)
clinic_beds_per_100k = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_PER_100K_FEATURE_KEY,
    initializer=None)
new_icu_beds = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_NUM_NEW_ICU_BEDS_FEATURE_KEY,
    initializer=None)
healthcare = [
    doctors,
    doctors_per_100k,
    hospital_beds,
    hospital_beds_per_100k,
    clinic_beds,
    clinic_beds_per_100k,
    new_icu_beds,
]

h1n1 = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_H1N1_in_2010_FEATURE_KEY, initializer=None)
alcohol = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_ALCOHOL_INTAKE_SCORE_FEATURE_KEY,
    initializer=None)
bmi_male_average = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_BMI_MALE_AVERAGE_FEATURE_KEY,
    initializer=None)
bmi_female_average = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_BMI_FEMALE_LOWER_RANGE_FEATURE_KEY,
    initializer=None)
smokers_male = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_SMOKERS_MALE_FEATURE_KEY, initializer=None)
smokers_female = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_SMOKERS_FEMALE_FEATURE_KEY,
    initializer=None)
wellness = [
    h1n1, alcohol, bmi_male_average, bmi_female_average, smokers_male,
    smokers_female
]


# Time-varying feature candidates
confirmed_preprocessed = model_spec_lib.FeatureSpec(
    name=constants.CONFIRMED_PREPROCESSED,
    initializer=None,
    forecast_method=model_spec_lib.ForecastMethod.NONE)
death_preprocessed = model_spec_lib.FeatureSpec(
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
state_of_emergency = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_STATE_OF_EMERGENCY_FEATURE_KEY,
    initializer=None)
total_tests = model_spec_lib.FeatureSpec(
    name=constants.TOTAL_TESTS, initializer=None)
reproductive_number = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_EFFECTIVE_REPRODUCTIVE_NUMBER_FEATURE_KEY,
    initializer=None)
survey = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_SURVEY_FEATURE_KEY,
    initializer=None)
survey_unweighted = model_spec_lib.FeatureSpec(
    name=constants
    .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_UNWEIGHTED_SURVEY_FEATURE_KEY,
    initializer=None)
survey_percent = model_spec_lib.FeatureSpec(
    name=constants
    .JAPAN_PREFECTURE_COVID_LIKE_ILLNESS_PERCENT_SURVEY_FEATURE_KEY,
    initializer=None)
survey_all = [survey, survey_unweighted, survey_percent]
mobility_parks = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_PARKS,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.POSITIVE)
mobility_work = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_WORK,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.POSITIVE)
mobility_res = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_RES,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.NEGATIVE)
mobility_transit = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_TRANSIT,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.POSITIVE)
mobility_grocery = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_GROCERY,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.POSITIVE)
mobility_retail = model_spec_lib.FeatureSpec(
    name=constants.GOOGLE_MOBILITY_RETAIL,
    initializer=None,
    weight_sign_constraint=model_spec_lib.EncoderWeightSignConstraint.POSITIVE)
mobility_all = [
    mobility_parks, mobility_work, mobility_res, mobility_transit,
    mobility_grocery, mobility_retail
]
discharged = model_spec_lib.FeatureSpec(
    name=constants.JAPAN_PREFECTURE_DISCHARGED_FEATURE_KEY, initializer=None)
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
          "restart_threshold": 3000,
          "time_scale_weight": 0.00006243159539906051,
          "train_loss_coefs": [0.01, 0.01, 0.1, 0.3, 0.01, 0.01],
          "valid_loss_coefs": [0.01, 0.01, 0.1, 0.3, 0.01, 0.01],
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
          "recovery_h_rate_init": -1.9032896753850963,
          "recovery_id_rate_init": -1.495660223962899,
          "recovery_iud_rate_init": -1.475605314236803,
          "hospitalization_rate_init": -1.4331763640928012,
          "death_h_rate_init": -1.4886876719378206,
          "death_id_rate_init": -1.8060447968974489,
          "bias_penalty_coef": 0.2835406167308398,
          "r_eff_penalty_coef": 2.0,
          "r_eff_penalty_cutoff": 5.0,
          "acceleration_death_coef": 0.1,
          "acceleration_confirm_coef": 0.1,
          "acceleration_hospital_coef": 0.1,
          "quantile_encoding_window": 7,
          "quantile_smooth_coef": 0.90,
          "quantile_training_iteration_ratio": 1.0,
          "width_coef_train": 1.0,
          "width_coef_valid": 1.0,
          "quantile_cum_viol_coef": 500.0,
          "increment_loss_weight": 0.0,
          "lasso_penalty_coef": 1.0,
          "covariate_training_mixing_coef": 1.0,
          "train_window_range": 2.0,
          "partial_mean_interval": 3,
          "direction_loss_coef": 2000.0,
          "train_crps_weight": 0.75,
          "valid_crps_weight": 0.75
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
              static_feature_specs=demographics + wellness,
              covariate_feature_specs=[
                  confirmed_preprocessed, death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, state_of_emergency,
                  reproductive_number
              ] + survey_all + mobility_all,
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
              static_feature_specs=demographics + wellness,
              covariate_feature_specs=[
                  confirmed_preprocessed, death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum, state_of_emergency,
                  reproductive_number
              ] + survey_all + mobility_all,
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
              static_feature_specs=demographics + wellness,
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
              static_feature_specs=healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  total_tests,
                  dow_feature,
              ] + survey_all,
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
              static_feature_specs=demographics + wellness + healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  dow_feature,
              ] + survey_all,
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
              static_feature_specs=demographics + wellness + healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
              ] + survey_all,
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
              static_feature_specs=demographics + wellness + healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  discharged,
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
              static_feature_specs=demographics + wellness + healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  discharged,
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
              static_feature_specs=demographics + wellness + healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
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
              static_feature_specs=demographics + wellness + healthcare,
              covariate_feature_specs=[
                  confirmed_preprocessed,
                  death_preprocessed,
                  preprocessed_confirmed_mean_to_sum,
                  preprocessed_death_mean_to_sum,
                  discharged,
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
        "average_contact_id_rate_init": -2.4387148524945594,
        "average_contact_iud_rate_init": -1.561117964142531,
        "reinfectable_rate_init": -5.447821316098564,
        "alpha_rate_init": -1.1906569150841058,
        "diagnosis_rate_init": -1.589940307765024,
        "recovery_id_rate_init": -1.5723201455595701,
        "recovery_iud_rate_init": -1.8295556397198884,
        "death_id_rate_init": -1.0766994750650696,
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
        "direction_loss_coef": 2000.0,
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
            encoder_name="hospitalization_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_id_rate", encoder_type="static"),
        model_spec_lib.EncoderSpec(
            encoder_name="death_h_rate", encoder_type="static")
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

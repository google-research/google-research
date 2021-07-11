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

"""Unit tests for tf_seir.py.

Use python3 -m models.tf_seir_test to run the tests.
"""
import unittest
from dateutil import parser
from mock import patch
import numpy as np
from covid_epidemiology.src import constants
from covid_epidemiology.src.models import tf_seir
from covid_epidemiology.src.models.definitions import example_model_definition
from covid_epidemiology.src.models.definitions import japan_model_definitions
from covid_epidemiology.src.models.definitions import us_model_definitions


class TFSEIRTestCase(unittest.TestCase):

  def test_us(self):
    train_window_size = 90
    actual_training_window_end_date = parser.parse("4/3/2020 00:00 UTC")
    total_population = 100000.0
    train_window_end_index = train_window_size

    full_sim_steps = train_window_size + 7
    # Confirmed cases and deaths are monotonically increasing by 1 in each time
    # step. There are no recoveries.
    gt_confirmed = np.array(range(1, full_sim_steps + 1))
    gt_confirmed_rate = np.zeros(full_sim_steps)
    gt_recovered = np.zeros(full_sim_steps)
    gt_death = np.array(range(0, full_sim_steps))
    gt_death_rate = np.zeros(full_sim_steps)
    gt_hospitalized = np.zeros(full_sim_steps)
    gt_hospitalized_cumulative = np.zeros(full_sim_steps)
    gt_hospitalized_increase = np.zeros(full_sim_steps)
    gt_icu = np.zeros(full_sim_steps)
    gt_ventilator = np.zeros(full_sim_steps)

    train_window_end_index = train_window_size

    # ts_features and static_features must include all the constants from
    # generic_seir_specs_state:
    ts_features = {
        constants.CONFIRMED: {
            "US": gt_confirmed,
        },
        # infected = confirmed - recovered
        constants.INFECTED: {
            "US": np.subtract(gt_confirmed, gt_recovered)
        },
        constants.RECOVERED_DOC: {
            "US": gt_recovered
        },
        constants.DEATH: {
            "US": gt_death
        },
        # All the covariates duplicate deaths
        constants.HOSPITALIZED: {
            "US": gt_hospitalized
        },
        constants.HOSPITALIZED_CUMULATIVE: {
            "US": gt_hospitalized_cumulative
        },
        constants.HOSPITALIZED_INCREASE: {
            "US": gt_hospitalized_increase
        },
        constants.ICU: {
            "US": gt_icu
        },
        constants.VENTILATOR: {
            "US": gt_ventilator
        },
        constants.MOBILITY_INDEX: {
            "US": gt_death
        },
        constants.MOBILITY_SAMPLES: {
            "US": gt_death
        },
        constants.TOTAL_TESTS: {
            "US": gt_death
        },
        constants.CONFIRMED_PREPROCESSED: {
            "US": gt_death
        },
        constants.CONFIRMED_PER_TESTS: {
            "US": gt_death
        },
        constants.CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO: {
            "US": gt_confirmed_rate
        },
        constants.DEATH_PREPROCESSED: {
            "US": gt_death
        },
        constants.DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO: {
            "US": gt_death_rate
        },
        constants.AMP_RESTAURANTS: {
            "US": gt_death
        },
        constants.AMP_NON_ESSENTIAL_BUSINESS: {
            "US": gt_death
        },
        constants.AMP_STAY_AT_HOME: {
            "US": gt_death
        },
        constants.AMP_SCHOOLS_SECONDARY_EDUCATION: {
            "US": gt_death
        },
        constants.AMP_EMERGENCY_DECLARATION: {
            "US": gt_death
        },
        constants.AMP_GATHERINGS: {
            "US": gt_death
        },
        constants.AMP_FACE_MASKS: {
            "US": gt_death
        },
        constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED: {
            "US": gt_death_rate
        },
        constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED: {
            "US": gt_death_rate
        },
        constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE: {
            "US": gt_death_rate
        },
        constants.VACCINATED_EFFECTIVENESS_SECOND_DOSE: {
            "US": gt_death_rate
        },
        constants.VACCINE_EFFECTIVENESS: {
            "US": gt_death
        },
        constants.DOW_WINDOW: {
            "US": gt_death
        },
        constants.TOTAL_TESTS_PER_CAPITA: {
            "US": gt_death
        },
        constants.AVERAGE_TEMPERATURE: {
            "US": gt_death
        },
        constants.MAX_TEMPERATURE: {
            "US": gt_death
        },
        constants.MIN_TEMPERATURE: {
            "US": gt_death
        },
        constants.RAINFALL: {
            "US": gt_death
        },
        constants.SNOWFALL: {
            "US": gt_death
        },
        constants.COMMERCIAL_SCORE: {
            "US": gt_death
        },
        constants.ANTIGEN_POSITIVE: {
            "US": gt_death
        },
        constants.ANTIGEN_TOTAL: {
            "US": gt_death
        },
        constants.ANTIBODY_NEGATIVE: {
            "US": gt_death
        },
        constants.ANTIBODY_TOTAL: {
            "US": gt_death
        },
        constants.ANTIGEN_POSITIVE_RATIO: {
            "US": gt_death
        },
        constants.ANTIBODY_NEGATIVE_RATIO: {
            "US": gt_death
        },
        constants.SYMPTOM_COUGH: {
            "US": gt_death
        },
        constants.SYMPTOM_CHILLS: {
            "US": gt_death
        },
        constants.SYMPTOM_ANOSMIA: {
            "US": gt_death
        },
        constants.SYMPTOM_INFECTION: {
            "US": gt_death
        },
        constants.SYMPTOM_CHEST_PAIN: {
            "US": gt_death
        },
        constants.SYMPTOM_FEVER: {
            "US": gt_death
        },
        constants.SYMPTOM_SHORTNESSBREATH: {
            "US": gt_death
        },
    }

    static_features = {
        constants.POPULATION: {
            "US": total_population
        },
        constants.POPULATION_DENSITY_PER_SQKM: {
            "US": 0.5
        },
        constants.INCOME_PER_CAPITA: {
            "US": 10000
        },
        constants.HOUSEHOLD_FOOD_STAMP: {
            "US": 10
        },
        constants.HOUSEHOLDS: {
            "US": 10
        },
        constants.KAISER_60P_POPULATION: {
            "US": 100
        },
        constants.KAISER_POPULATION: {
            "US": 100
        },
        constants.POPULATION_60P_RATIO: {
            "US": 0.3
        },
        constants.ICU_BEDS: {
            "US": 100
        },
        constants.PATIENCE_EXPERIENCE_ABOVE: {
            "US": 20
        },
        constants.PATIENCE_EXPERIENCE_BELOW: {
            "US": 20
        },
        constants.PATIENCE_EXPERIENCE_SAME: {
            "US": 20
        },
        constants.CRITICAL_ACCESS_HOSPITAL: {
            "US": 20
        },
        constants.HOSPITAL_ACUTE_CARE: {
            "US": 20
        },
        constants.EMERGENCY_SERVICES: {
            "US": 20
        },
        constants.NON_EMERGENCY_SERVICES: {
            "US": 20
        },
        constants.AQI_MEAN: {
            "US": 1.
        },
        constants.HOSPITAL_RATING_AVERAGE: {
            "US": 1.7
        },
    }

    model_definition = us_model_definitions.StateModelDefinition(
        gt_source="TEST")
    num_forecast_steps = 7
    hparams_overrides = {"fine_tuning_steps": 1}  # Saves time in fine tuning.
    tf_seir_model = tf_seir.TfSeir(
        model_type=constants.MODEL_TYPE_TIME_VARYING_WITH_COVARIATES,
        location_granularity=constants.LOCATION_GRANULARITY_STATE,
        model_definition=model_definition,
        covariate_delay=0,
        random_seed=1,
        **hparams_overrides)

    with patch("tensorflow.function", lambda func: func):
      # Tests with @tf.function turned into no-op to save time.
      (model_output_forecast,
       model_output_all) = tf_seir_model.fit_forecast_fixed(
           train_window_end_index=train_window_end_index,
           train_window_end_date=actual_training_window_end_date,
           num_forecast_steps=num_forecast_steps,
           num_train_forecast_steps=num_forecast_steps,
           static_features=static_features,
           static_overrides=None,
           ts_features=ts_features,
           ts_overrides=None,
           ts_categorical_features=None,
           ts_state_features=ts_features,
           locations=["US"],
           num_iterations=1,
           display_iterations=100,
           optimization="RMSprop",
           training_data_generator=False,
           quantile_regression=False,
           static_scalers=None,
           ts_scalers=None,
           ts_state_scalers=None)

    self.assertEqual(
        len(model_output_forecast.location_to_window_predictions["US"]), 1)
    window_predictions_forecast = (
        model_output_forecast.location_to_window_predictions["US"][0])
    self.assertEqual(window_predictions_forecast.training_window_end,
                     actual_training_window_end_date,
                     actual_training_window_end_date)

    for time_horizon in range(1, num_forecast_steps + 1):
      confirmed_prediction = window_predictions_forecast.metric_to_predictions[
          "confirmed"][time_horizon - 1]
      death_prediction = window_predictions_forecast.metric_to_predictions[
          "death"][time_horizon - 1]
      recovered_prediction = window_predictions_forecast.metric_to_predictions[
          "recovered_documented"][time_horizon - 1]
      infected_prediction = window_predictions_forecast.metric_to_predictions[
          "infected_documented"][time_horizon - 1]
      self.assertEqual(confirmed_prediction.time_horizon, time_horizon)
      self.assertEqual(death_prediction.time_horizon, time_horizon)
      self.assertEqual(recovered_prediction.time_horizon, time_horizon)
      self.assertEqual(infected_prediction.time_horizon, time_horizon)
      self.assertEqual(death_prediction.ground_truth,
                       train_window_size + time_horizon - 1)
      self.assertEqual(infected_prediction.ground_truth,
                       train_window_size + time_horizon)
      self.assertEqual(confirmed_prediction.ground_truth,
                       train_window_size + time_horizon)
      self.assertEqual(recovered_prediction.ground_truth, 0)

      self.assertEqual(
          len(model_output_all.location_to_window_predictions["US"]), 1)
    window_predictions_all = (
        model_output_all.location_to_window_predictions["US"][0])
    self.assertEqual(window_predictions_all.training_window_end,
                     actual_training_window_end_date,
                     actual_training_window_end_date)

    for i in range(0, train_window_size + num_forecast_steps):
      # time_horizon: training window: [-train_window_size+1,0]
      # forecast window: [1, num_forecast_steps]
      time_horizon = i - train_window_size + 1
      confirmed_prediction = window_predictions_all.metric_to_predictions[
          "confirmed"][i]
      death_prediction = window_predictions_all.metric_to_predictions[
          "death"][i]
      recovered_prediction = window_predictions_all.metric_to_predictions[
          "recovered_documented"][i]
      infected_prediction = window_predictions_all.metric_to_predictions[
          "infected_documented"][i]
      self.assertEqual(confirmed_prediction.time_horizon, time_horizon)
      self.assertEqual(death_prediction.time_horizon, time_horizon)
      self.assertEqual(recovered_prediction.time_horizon, time_horizon)
      self.assertEqual(infected_prediction.time_horizon, time_horizon)
      self.assertEqual(death_prediction.ground_truth, i)
      self.assertEqual(infected_prediction.ground_truth, i + 1)
      self.assertEqual(confirmed_prediction.ground_truth, i + 1)
      self.assertEqual(recovered_prediction.ground_truth, 0)

  def test_japan_sanity(self):
    # Create the basic model.
    model_definition = japan_model_definitions.PrefectureModelDefinition()
    tf_seir_model = tf_seir.TfSeir(
        model_type="TIME_VARYING_WITH_COVARIATES",
        location_granularity="JAPAN_PREFECTURE",
        model_definition=model_definition,
        covariate_delay=0,
        random_seed=1,
        fine_tuning_steps=1)

    # Generate some data that is shaped correctly but is nonsense.
    train_window_size = 90
    full_sim_steps = train_window_size + 7
    gt_confirmed = np.array(range(1, full_sim_steps + 1))

    model_spec = model_definition.get_model_spec(
        constants.MODEL_TYPE_TIME_VARYING_WITH_COVARIATES)

    # Format the data as the fit pipeline expects.
    required_ts_constants = [
        constants.CONFIRMED,
        constants.INFECTED,
        constants.RECOVERED_DOC,
        constants.HOSPITALIZED,
        constants.HOSPITALIZED_CUMULATIVE,
        constants.HOSPITALIZED_INCREASE,
        constants.DEATH,
        constants.ICU,
        constants.VENTILATOR,
    ] + model_spec.covariate_names
    required_static_constants = [
        constants.POPULATION,
        constants.DENSITY,
        constants.JAPAN_PREFECTURE_AGE_0_TO_14_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_15_TO_64_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_64_PLUS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_AGE_75_PLUS_FEATURE_KEY,
        constants.INCOME_PER_CAPITA,
        constants.JAPAN_PREFECTURE_NUM_DOCTORS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_DOCTORS_PER_100K_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_HOSPITAL_BEDS_PER_100K_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_CLINIC_BEDS_PER_100K_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_NUM_NEW_ICU_BEDS_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_H1N1_in_2010_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_ALCOHOL_INTAKE_SCORE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_MALE_AVERAGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_BMI_FEMALE_LOWER_RANGE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_SMOKERS_MALE_FEATURE_KEY,
        constants.JAPAN_PREFECTURE_SMOKERS_FEMALE_FEATURE_KEY,
    ]
    jp_ = "JAPAN_PREFECTURE"
    ts_features = {c: {jp_: gt_confirmed} for c in required_ts_constants}
    static_features = {c: {jp_: 0.0} for c in required_static_constants}
    # Sanity check the fit forecast function.
    # TODO(joelshor): Consider using `fit_forecast_moving_window`, which is what
    # is actually used in `fit_forecast_pipeline`.
    with patch("tensorflow.function", lambda func: func):
      # Tests with @tf.function turned into no-op to save time.
      tf_seir_model.fit_forecast_fixed(
          train_window_end_index=90,
          train_window_end_date=parser.parse("4/3/2020 00:00 UTC"),
          num_forecast_steps=7,
          num_train_forecast_steps=7,
          static_features=static_features,
          static_overrides=None,
          ts_features=ts_features,
          ts_overrides=None,
          ts_categorical_features=None,
          ts_state_features=None,
          locations=["JAPAN_PREFECTURE"],
          num_iterations=1,  # execute quickly, not converge.
          display_iterations=100,
          optimization="RMSprop",
          training_data_generator=False,
          quantile_regression=False,
          static_scalers=None,
          ts_scalers=None,
          ts_state_scalers=None)

  def test_example_sanity(self):
    # Create the basic model.
    model_definition = example_model_definition.ExampleModelDefinition()
    model_type = constants.MODEL_TYPE_TIME_VARYING_WITH_COVARIATES
    tf_seir_model = tf_seir.TfSeir(
        model_type=model_type,
        location_granularity="STATE",
        model_definition=model_definition,
        covariate_delay=0,
        random_seed=1,
        fine_tuning_steps=1)

    # Generate some data that is shaped correctly but is nonsense.
    train_window_size = 90
    full_sim_steps = train_window_size + 7
    gt_confirmed = np.array(range(1, full_sim_steps + 1))

    model_spec = model_definition.get_model_spec(model_type)

    # Format the data as the fit pipeline expects.
    required_ts_constants = [
        constants.CONFIRMED,
        constants.INFECTED,
        constants.RECOVERED_DOC,
        constants.HOSPITALIZED,
        constants.HOSPITALIZED_CUMULATIVE,
        constants.HOSPITALIZED_INCREASE,
        constants.DEATH,
    ] + model_spec.covariate_names
    required_static_constants = [
        constants.POPULATION,
        constants.POPULATION_DENSITY_PER_SQKM,
    ]
    loc = "EX_LOCATION"
    ts_features = {c: {loc: gt_confirmed} for c in required_ts_constants}
    static_features = {c: {loc: 0.0} for c in required_static_constants}
    # Sanity check the fit forecast function.
    with patch("tensorflow.function", lambda func: func):
      # Tests with @tf.function turned into no-op to save time.
      tf_seir_model.fit_forecast_fixed(
          train_window_end_index=90,
          train_window_end_date=parser.parse("4/3/2020 00:00 UTC"),
          num_forecast_steps=7,
          num_train_forecast_steps=7,
          static_features=static_features,
          static_overrides=None,
          ts_features=ts_features,
          ts_overrides=None,
          ts_categorical_features=None,
          ts_state_features=None,
          locations=[loc],
          num_iterations=1,  # execute quickly, not converge.
          display_iterations=100,
          optimization="RMSprop",
          training_data_generator=False,
          quantile_regression=False,
          static_scalers=None,
          ts_scalers=None,
          ts_state_scalers=None)


if __name__ == "__main__":
  unittest.main()

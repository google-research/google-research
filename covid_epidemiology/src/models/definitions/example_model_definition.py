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

"""A model definition for US States without ICU or Ventilator."""
from typing import Dict, Set, Tuple
import warnings

import numpy as np
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src.models import generic_seir_county_model_constructor
from covid_epidemiology.src.models.definitions import compartmental_model_definitions
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib
from covid_epidemiology.src.models.shared import model_utils
from covid_epidemiology.src.models.shared import typedefs


class ExampleModelDefinition(
    compartmental_model_definitions.BaseSeirHospitalModelDefinition,):
  """Example model definition.

  This model uses the same compartmental dynamics as the US County model with a
  small subset of features from the US State data. It is simply intended as an
  example of a custom model.
  """

  def get_static_features(self):
    return {
        constants.POPULATION:
            constants.POPULATION,
        constants.POPULATION_DENSITY_PER_SQKM:
            constants.POPULATION_DENSITY_PER_SQKM,
    }

  def get_ts_features(self):
    return {
        constants.DEATH:
            constants.JHU_DEATH_FEATURE_KEY,
        constants.CONFIRMED:
            constants.JHU_CONFIRMED_FEATURE_KEY,
        constants.RECOVERED_DOC:
            constants.RECOVERED_FEATURE_KEY,
        constants.HOSPITALIZED:
            constants.HOSPITALIZED_FEATURE_KEY,
        constants.HOSPITALIZED_CUMULATIVE:
            constants.HOSPITALIZED_CUMULATIVE_FEATURE_KEY,
        constants.HOSPITALIZED_INCREASE:
            constants.HOSPITALIZED_INCREASE_FEATURE_KEY,
        constants.MOBILITY_INDEX:
            constants.MOBILITY_INDEX,
    }

  def get_ts_features_to_preprocess(self):
    return {
        constants.MOBILITY_INDEX,
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
    return transformed_features, feature_scalers

  def get_model_spec(
      self,
      model_type,
      covariate_delay = 0,
      **kwargs,
  ):
    # Static feature candidates
    population_density_per_sq_km = model_spec_lib.FeatureSpec(
        name=constants.POPULATION_DENSITY_PER_SQKM, initializer=None)

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
    mobility_index = model_spec_lib.FeatureSpec(
        name=constants.MOBILITY_INDEX, initializer=None)

    all_static_covariates = [population_density_per_sq_km]

    base_ts_covariates = [
        preprocessed_confirmed,
        preprocessed_death,
        preprocessed_confirmed_mean_to_sum,
        preprocessed_death_mean_to_sum,
    ]

    default_window_size = 7
    use_fixed_covariate_mask = True

    return model_spec_lib.ModelSpec(
        hparams={
            "initial_learning_rate": 0.003061242915556316,
            "momentum": 0.2,
            "decay_steps": 1000,
            "fine_tuning_steps": 500,
            "fine_tuning_decay": 1.0,
            "decay_rate": 1.0,
            "location_dependent_init": False,
            "infected_threshold": 3,
            "restart_threshold": 300,
            "time_scale_weight": 0.00006243159539906051,
            "train_loss_coefs": [0, 0.001, 0.2, 0.1, 0.01, 0.01],
            "valid_loss_coefs": [0, 0.001, 0.2, 0.1, 0.01, 0.01],
            "sync_coef": 0.3,
            "reduced_sync_undoc": 2.0,
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
            "direction_loss_coef": 5000.0,
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
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates + [mobility_index],
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="average_contact_iud_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates + [mobility_index],
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="reinfectable_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
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
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="recovery_id_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="recovery_iud_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="recovery_h_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="hospitalization_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="death_id_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
            model_spec_lib.EncoderSpec(
                encoder_name="death_h_rate",
                encoder_type="gam",
                static_feature_specs=all_static_covariates,
                covariate_feature_specs=base_ts_covariates,
                covariate_feature_window=default_window_size,
                encoder_kwargs={
                    "initial_bias": 0,
                    "location_dependent_bias": True,
                    "use_fixed_covariate_mask": use_fixed_covariate_mask,
                }),
        ])

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
    return ExampleModelConstructor(
        model_spec=model_spec, random_seed=random_seed)

  def bound_variables(
      self,
      seir_timeseries_variables,
  ):
    # Remove this warning after making sure the bounds are reasonable for the
    # data.
    warnings.warn("Check SEIR Variable bounds.")
    return super().bound_variables(seir_timeseries_variables)


class ExampleModelConstructor(
    generic_seir_county_model_constructor.CountyModelConstructor):
  """Constructs a county Tensorflow model, to be used in tf_seir."""

  def compute_coef(self,
                   ground_truth_timeseries,
                   ground_truth_state,
                   num_train_steps,
                   num_known_steps,
                   power=2.0):
    """Compute train/valid coefficients for loss computation.

    Args:
      ground_truth_timeseries: ground truth compartments
      ground_truth_state: ground truth state level compartments
      num_train_steps: number of timesteps for training
      num_known_steps: number of known timesteps
      power: 2 for MSE and 1 for MAE

    Returns:
      train_coefs: training coeffcients for each compartment
      valid_coefs: valid coeffcients for each compartment
    """
    # We will not use any State level data for this model
    del ground_truth_state
    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    # Recovered
    recovered_train, recovered_valid = model_utils.compartment_base(
        gt_list["recovered"], gt_indicator["recovered"], num_train_steps,
        num_known_steps)
    # Death
    death_train, death_valid = model_utils.compartment_base(
        gt_list["death"], gt_indicator["death"], num_train_steps,
        num_known_steps)
    # Confirmed
    confirmed_train, confirmed_valid = model_utils.compartment_base(
        gt_list["confirmed"], gt_indicator["confirmed"], num_train_steps,
        num_known_steps)
    # Hospitalized
    hospitalized_train, hospitalized_valid = model_utils.compartment_base(
        gt_list["hospitalized"], gt_indicator["hospitalized"], num_train_steps,
        num_known_steps)
    # Hospitalized cumulative
    hospitalized_cumulative_train, hospitalized_cumulative_valid = model_utils.compartment_base(
        gt_list["hospitalized_cumulative"],
        gt_indicator["hospitalized_cumulative"], num_train_steps,
        num_known_steps)

    train_coefs = [
        0,
        (death_train / recovered_train)**power,
        1,
        (death_train / confirmed_train)**power,
        (death_train / hospitalized_train)**power,
        (death_train / hospitalized_cumulative_train)**power,
    ]

    valid_coefs = [
        0,
        (death_valid / recovered_valid)**power,
        1,
        (death_valid / confirmed_valid)**power,
        (death_valid / hospitalized_valid)**power,
        (death_valid / hospitalized_cumulative_valid)**power,
    ]

    train_coefs = np.nan_to_num(train_coefs).tolist()
    valid_coefs = np.nan_to_num(valid_coefs).tolist()

    return train_coefs, valid_coefs

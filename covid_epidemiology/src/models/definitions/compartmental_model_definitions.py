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

"""Base compartmental model definitions.

These compartmental models can be extended to share compartmental dynamics
across multiple models.
"""
import abc
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src.models.definitions import base_model_definition
from covid_epidemiology.src.models.shared import model_utils
from covid_epidemiology.src.models.shared import typedefs


# noinspection PyMethodMayBeStatic
class BaseSeirHospitalModelDefinition(
    base_model_definition.BaseCovidModelDefinition, abc.ABC):
  """An extended SEIR model including a hospitalized compartment.

  This extends the common SEIR model to have the following compartments:
    * Susceptible
    * Exposed
    * Undocumented infected
    * Undocumented recovered
    * Documented infected
    * Documented recovered
    * Death
    * Hospitalized

  The model also includes the possibility of becoming reinfected, and
  vaccination effects. The US County Model and the Japan Prefecture model
  are examples of model's based on the structure.
  """

  # Define the encoders for the model
  _ENCODER_RATE_LIST = constants.HOSPITAL_RATE_LIST

  _NUM_STATES: int = 15

  def transform_ts_features(
      self,
      ts_features,
      static_features,
      initial_train_window_size,
  ):
    """Transforms timeseries features (scales them, removes NaNs, etc).

    Can also create new features (e.g., ratios of existing features).

    Args:
      ts_features: A mapping from the feature name to its value, the value of
        each feature is a map from location to np.ndarray.
      static_features: A mapping from the static feature name to its value, the
        value of each feature is a map from location to float.
      initial_train_window_size: Size of initial training window.

    Returns:
      A mapping from the feature name to its value, the value of each feature
      is a map from location to np.ndarray.
    """
    # This should call the base covid model definition
    transformed_features, feature_scalers = super().transform_ts_features(
        ts_features, static_features, initial_train_window_size)
    _add_vaccine_features(ts_features[constants.DEATH], transformed_features,
                          feature_scalers)
    return transformed_features, feature_scalers

  def initialize_seir_state(self, ground_truth_timeseries, infected_threshold,
                            trainable):
    """Returns initialized states for seir dynamics."""

    np.random.seed(self.random_seed)

    (chosen_populations_list, gt_list, gt_indicator, _,
     _) = ground_truth_timeseries

    tensor_shape = np.shape(gt_list["confirmed"][0])
    exposed_initial = np.maximum(
        np.random.uniform(0, 10) * infected_threshold,
        np.random.uniform(0, 10, size=tensor_shape) * gt_list["confirmed"][0])

    # Set a minimal value so that the differential equations do not get stuck
    # at all 0 state for missing data.
    infected_ud_initial = np.maximum(
        np.random.uniform(0, 10) * infected_threshold,
        np.random.uniform(0, 10, size=tensor_shape) * gt_list["confirmed"][0])

    infected_ud_increase_initial = np.zeros(tensor_shape)

    # Set the infected if the data is not missing.
    infected_d_initial = (
        gt_indicator["infected"][0] * gt_list["infected"][0] +
        (1 - gt_indicator["infected"][0]) * gt_list["confirmed"][0])

    # Set the recovered if the data is not missing.
    recovered_d_initial = gt_list["recovered"][0]

    recovered_ud_initial = np.random.uniform(
        0, 5, size=tensor_shape) * gt_list["recovered"][0]

    # Set the hospitalized if the data is not missing.
    hospitalized_d_initial = (
        gt_indicator["hospitalized"][0] * gt_list["hospitalized"][0] +
        (1 - gt_indicator["hospitalized"][0]) *
        np.random.uniform(0, 0.3, size=tensor_shape) * gt_list["confirmed"][0])

    hospitalized_cumulative_d_initial = (
        gt_indicator["hospitalized_cumulative"][0] *
        gt_list["hospitalized_cumulative"][0] +
        (1 - gt_indicator["hospitalized_cumulative"][0]) *
        np.random.uniform(0, 0.3, size=tensor_shape) * gt_list["confirmed"][0])

    hospitalized_increase_d_initial = np.zeros(tensor_shape)

    # Assume all deaths are reported.
    death_d_initial = gt_list["death"][0]

    # We make the total population fixed, i.e. should not be considered in
    # gradient computations for updates.
    population_d_initial = tf.stop_gradient(chosen_populations_list)

    reinfectable_d_initial = (
        np.random.uniform(0, 0.01, size=tensor_shape) * gt_list["confirmed"][0])

    reinfectable_ud_initial = (
        np.random.uniform(0, 0.01, size=tensor_shape) * gt_list["confirmed"][0])

    reinfectable_vaccine_initial = np.zeros(tensor_shape)

    vaccine_immuned_initial = np.zeros(tensor_shape)

    return tf.Variable(
        np.asarray([
            exposed_initial, infected_d_initial, infected_ud_initial,
            recovered_d_initial, recovered_ud_initial, hospitalized_d_initial,
            hospitalized_cumulative_d_initial, hospitalized_increase_d_initial,
            death_d_initial, population_d_initial, reinfectable_d_initial,
            reinfectable_ud_initial, reinfectable_vaccine_initial,
            vaccine_immuned_initial, infected_ud_increase_initial
        ]),
        dtype=tf.float32,
        trainable=trainable,
    )

  def bound_variables(
      self,
      seir_timeseries_variables,
  ):
    """See parent class."""

    (first_dose_vaccine_ratio_per_day_list,
     second_dose_vaccine_ratio_per_day_list, average_contact_id_list,
     average_contact_iud_list, reinfectable_rate_list, alpha_list,
     diagnosis_rate_list, recovery_rate_id_list, recovery_rate_iud_list,
     recovery_rate_h_list, hospitalization_rate_list, death_rate_id_list,
     death_rate_h_list) = seir_timeseries_variables
    first_dose_vaccine_ratio_per_day = model_utils.apply_relu_bounds(
        first_dose_vaccine_ratio_per_day_list[-1], 0.0, 1.0)
    second_dose_vaccine_ratio_per_day = model_utils.apply_relu_bounds(
        second_dose_vaccine_ratio_per_day_list[-1], 0.0, 1.0)
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

    return (first_dose_vaccine_ratio_per_day, second_dose_vaccine_ratio_per_day,
            average_contact_id, average_contact_iud, reinfectable_rate, alpha,
            diagnosis_rate, recovery_rate_id, recovery_rate_iud,
            recovery_rate_h, hospitalization_rate, death_rate_id, death_rate_h)

  def initialize_ground_truth_timeseries(
      self,
      static_features,
      ts_features,
      chosen_locations,
      num_observed_timesteps,
      infected_threshold,
  ):
    """See parent class."""
    chosen_populations = static_features[constants.POPULATION]
    num_locations = len(chosen_locations)

    chosen_populations_list = np.zeros(num_locations, dtype=np.float32)

    gt_input = {
        "infected": ts_features[constants.INFECTED],
        "recovered": ts_features[constants.RECOVERED_DOC],
        "confirmed": ts_features[constants.CONFIRMED],
        "hospitalized": ts_features[constants.HOSPITALIZED],
        "hospitalized_cumulative":
            (ts_features[constants.HOSPITALIZED_CUMULATIVE]),
        "death": ts_features[constants.DEATH],
    }
    gt_names = list(gt_input.keys())

    gt_list = dict()
    gt_indicator = dict()

    # Initialize GT
    for gt_name in gt_names:
      gt_list[gt_name] = np.zeros((num_locations, num_observed_timesteps),
                                  dtype=np.float32)
      gt_indicator[gt_name] = np.zeros((num_locations, num_observed_timesteps),
                                       dtype=np.float32)

    for location_index, location in enumerate(chosen_locations):
      if chosen_populations[location]:
        chosen_populations_list[location_index] = chosen_populations[location]

      for gt_name in gt_names:
        gt_list[gt_name], gt_indicator[gt_name] = model_utils.populate_gt_list(
            location_index, gt_input[gt_name], location, num_observed_timesteps,
            gt_list[gt_name], gt_indicator[gt_name])

    for gt_name in gt_names:
      gt_list[gt_name] = tf.transpose(
          tf.constant(gt_list[gt_name], dtype=tf.float32))
      gt_indicator[gt_name] = tf.transpose(
          tf.constant(gt_indicator[gt_name], dtype=tf.float32))
    chosen_populations_list = tf.constant(
        chosen_populations_list, dtype=tf.float32)

    orig_gt = gt_input
    ground_truth_timeseries = (chosen_populations_list, gt_list, gt_indicator,
                               gt_names, orig_gt)
    infection_active_mask = model_utils.construct_infection_active_mask(
        gt_list["confirmed"], num_locations, num_observed_timesteps,
        infected_threshold)
    infection_active_mask = tf.constant(infection_active_mask, dtype=tf.float32)
    return ground_truth_timeseries, infection_active_mask

  def sync_values(
      self,
      hparams,
      last_state,
      ground_truth_timeseries,
      timestep,
      is_training,
  ):
    """See parent class."""
    sync_coef = hparams["sync_coef"]
    # At inference, we use sync_coef = 1.0
    if not is_training:
      sync_coef = 1.0
    # Unstack the last state (for updating)
    (exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
     hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
     death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
     reinfectable_vaccine_t, vaccine_immuned_t,
     infected_ud_increase_t) = tf.unstack(last_state)

    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    # Inferred from the confirmed condition
    recovered_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["recovered"][timestep],
        gt_indicator=gt_indicator["recovered"][timestep],
        compartment=recovered_d_t + reinfectable_d_t,
        sync_coef=sync_coef)
    hospitalized_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["hospitalized"][timestep],
        gt_indicator=gt_indicator["hospitalized"][timestep],
        compartment=hospitalized_t,
        sync_coef=sync_coef)
    death_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["death"][timestep],
        gt_indicator=gt_indicator["death"][timestep],
        compartment=death_t,
        sync_coef=sync_coef)

    # Infected
    infected_d_t = model_utils.sync_compartment(
        gt_list=(gt_list["confirmed"][timestep] - recovered_gt_est -
                 hospitalized_gt_est - death_gt_est),
        gt_indicator=gt_indicator["confirmed"][timestep],
        compartment=infected_d_t,
        sync_coef=sync_coef)

    # Recovered
    recovered_d_t = model_utils.sync_compartment(
        gt_list=gt_list["recovered"][timestep],
        gt_indicator=gt_indicator["recovered"][timestep],
        compartment=recovered_d_t + reinfectable_d_t,
        sync_coef=sync_coef)
    recovered_d_t = recovered_d_t - reinfectable_d_t

    # Hospitalized
    hospitalized_t = model_utils.sync_compartment(
        gt_list=gt_list["hospitalized"][timestep],
        gt_indicator=gt_indicator["hospitalized"][timestep],
        compartment=hospitalized_t,
        sync_coef=sync_coef)

    # Hospitalized Cumulative
    hospitalized_cumulative_t = model_utils.sync_compartment(
        gt_list=gt_list["hospitalized_cumulative"][timestep],
        gt_indicator=gt_indicator["hospitalized_cumulative"][timestep],
        compartment=hospitalized_cumulative_t,
        sync_coef=sync_coef)

    # Death
    death_t = model_utils.sync_compartment(
        gt_list=gt_list["death"][timestep],
        gt_indicator=gt_indicator["death"][timestep],
        compartment=death_t,
        sync_coef=sync_coef)

    # Update the last state
    last_state = tf.stack([
        exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
        hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
        death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
        reinfectable_vaccine_t, vaccine_immuned_t, infected_ud_increase_t
    ])

    return last_state

  def sync_undoc(self, hparams, last_state,
                 ground_truth_timeseries,
                 last_variable, timestep,
                 is_training):
    """Synchronize the undocumented infected counts using confirmed increment.

    Args:
      hparams: Model's hyper-parameters. Usually contains sync_coef to define
        the amount of teacher forcing.
      last_state: The model's previous state.
      ground_truth_timeseries: The ground truth values to sync with.
      last_variable: The model's variables from the previous step. Should
        include the diagnosis rate.
      timestep: The current time step.
      is_training: True if the model is being trained.

    Returns:
      The updated values for the last_state.
    """
    # Divided by hparams["reduced_sync_undoc"] to synchronize with
    # less stable GT (GT times predicted values)
    sync_coef = hparams["sync_coef"] / hparams["reduced_sync_undoc"]
    # At inference, we use sync_coef = 1.0
    if not is_training:
      sync_coef = 1.0 / hparams["reduced_sync_undoc"]
    # Unstack the last state (for updating)
    (exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
     hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
     death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
     reinfectable_vaccine_t, vaccine_immuned_t,
     infected_ud_increase_t) = tf.unstack(last_state)

    # Get the previous diagnosis rate
    diagnosis_rate = last_variable[6]

    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    confirmed_increase_gt = (
        gt_list["confirmed"][timestep] - gt_list["confirmed"][timestep - 1])
    confirmed_increase_gt_indicator = (
        gt_indicator["confirmed"][timestep] *
        gt_indicator["confirmed"][timestep - 1])

    infected_ud_t = model_utils.sync_compartment(
        gt_list=confirmed_increase_gt * (1.0 / diagnosis_rate),
        gt_indicator=confirmed_increase_gt_indicator,
        compartment=infected_ud_t,
        sync_coef=sync_coef)

    # Update the last state
    last_state = tf.stack([
        exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
        hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
        death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
        reinfectable_vaccine_t, vaccine_immuned_t, infected_ud_increase_t
    ])

    return last_state

  def seir_dynamics(self, current_state,
                    seir_variables):
    raise NotImplementedError("To be ported from the model_constructor class")

  def compute_losses(
      self, hparams, propagated_states,
      ground_truth_timeseries
  ):
    raise NotImplementedError("To be ported from the model_constructor class")


# noinspection PyMethodMayBeStatic
class BaseSeirIcuVentilatorModelDefinition(
    base_model_definition.BaseCovidModelDefinition, abc.ABC):
  """An extended SEIR model including ICU & Ventilator compartments.

  This extends the common SEIR model to have the following compartments:
    * Susceptible
    * Exposed
    * Undocumented infected
    * Undocumented recovered
    * Documented infected
    * Documented recovered
    * Death
    * Hospitalized with ICU and Ventilator sub-compartments

  The model also includes the possibility of becoming reinfected, and
  vaccination effects. The US State Modelis an example of a model based on this
  structure.
  """

  # Define the encoders for the model
  _ENCODER_RATE_LIST = constants.ICU_AND_VENTILATOR_RATE_LIST

  _NUM_STATES: int = 17

  def transform_ts_features(
      self,
      ts_features,
      static_features,
      initial_train_window_size,
  ):
    """Transforms timeseries features (scales them, removes NaNs, etc).

    Can also create new features (e.g., ratios of existing features).

    Args:
      ts_features: A mapping from the feature name to its value, the value of
        each feature is a map from location to np.ndarray.
      static_features: A mapping from the static feature name to its value, the
        value of each feature is a map from location to float.
      initial_train_window_size: Size of initial training window.

    Returns:
      A mapping from the feature name to its value, the value of each feature
      is a map from location to np.ndarray.
    """
    # This should call the base covid model definition
    transformed_features, feature_scalers = super().transform_ts_features(
        ts_features, static_features, initial_train_window_size)
    _add_vaccine_features(ts_features[constants.DEATH], transformed_features,
                          feature_scalers)
    return transformed_features, feature_scalers

  def initialize_seir_state(self, ground_truth_timeseries, infected_threshold,
                            trainable):
    """Returns initialized states for seir dynamics."""

    np.random.seed(self.random_seed)

    (chosen_populations_list, gt_list, gt_indicator, _,
     _) = ground_truth_timeseries

    tensor_shape = np.shape(gt_list["confirmed"][0])
    exposed_initial = np.maximum(
        np.random.uniform(0, 10) * infected_threshold,
        np.random.uniform(0, 10, size=tensor_shape) * gt_list["confirmed"][0])

    # Set a minimal value so that the differential equations do not get stuck
    # at all 0 state for missing data.
    infected_ud_initial = np.maximum(
        np.random.uniform(0, 10) * infected_threshold,
        np.random.uniform(0, 10, size=tensor_shape) * gt_list["confirmed"][0])

    infected_ud_increase_initial = np.zeros(tensor_shape)

    # Set the infected if the data is not missing.
    infected_d_initial = (
        gt_indicator["infected"][0] * gt_list["infected"][0] +
        (1 - gt_indicator["infected"][0]) * gt_list["confirmed"][0])

    # Set the recovered if the data is not missing.
    recovered_d_initial = gt_list["recovered"][0]

    recovered_ud_initial = np.random.uniform(
        0, 5, size=tensor_shape) * gt_list["recovered"][0]

    # Set the hospitalized, ventilator and ICU if the data is not missing.
    hospitalized_d_initial = (
        gt_indicator["hospitalized"][0] * gt_indicator["icu"][0] *
        tf.nn.relu(gt_list["hospitalized"][0] - gt_list["icu"][0]) +
        (1 - gt_indicator["hospitalized"][0] * gt_indicator["icu"][0]) *
        np.random.uniform(0, 0.3, size=tensor_shape) * gt_list["confirmed"][0])

    hospitalized_cumulative_d_initial = (
        gt_indicator["hospitalized_cumulative"][0] *
        gt_list["hospitalized_cumulative"][0] +
        (1 - gt_indicator["hospitalized_cumulative"][0]) *
        np.random.uniform(0, 0.3, size=tensor_shape) * gt_list["confirmed"][0])

    hospitalized_increase_d_initial = np.zeros(tensor_shape)

    icu_d_initial = (
        gt_indicator["icu"][0] * gt_indicator["ventilator"][0] *
        tf.nn.relu(gt_list["icu"][0] - gt_list["ventilator"][0]) +
        (1 - gt_indicator["icu"][0] * gt_indicator["ventilator"][0]) *
        np.random.uniform(0, 0.15, size=tensor_shape) * gt_list["confirmed"][0])

    ventilator_d_initial = (
        gt_indicator["ventilator"][0] * gt_list["ventilator"][0] +
        (1 - gt_indicator["ventilator"][0]) *
        np.random.uniform(0, 0.1, size=tensor_shape) * gt_list["confirmed"][0])

    # Assume all deaths are reported.
    death_d_initial = gt_list["death"][0]

    # We make the total population fixed, i.e. should not be considered in
    # gradient computations for updates.
    population_d_initial = tf.stop_gradient(chosen_populations_list)

    reinfectable_d_initial = (
        np.random.uniform(0, 0.01, size=tensor_shape) * gt_list["confirmed"][0])

    reinfectable_ud_initial = (
        np.random.uniform(0, 0.01, size=tensor_shape) * gt_list["confirmed"][0])

    reinfectable_vaccine_initial = np.zeros(tensor_shape)

    vaccine_immuned_initial = np.zeros(tensor_shape)

    return tf.Variable(
        np.asarray([
            exposed_initial, infected_d_initial, infected_ud_initial,
            recovered_d_initial, recovered_ud_initial, hospitalized_d_initial,
            hospitalized_cumulative_d_initial, hospitalized_increase_d_initial,
            icu_d_initial, ventilator_d_initial, death_d_initial,
            population_d_initial, reinfectable_d_initial,
            reinfectable_ud_initial, reinfectable_vaccine_initial,
            vaccine_immuned_initial, infected_ud_increase_initial
        ]),
        dtype=tf.float32,
        trainable=trainable,
    )

  def bound_variables(
      self,
      seir_timeseries_variables,
  ):
    """See parent class."""

    (first_dose_vaccine_ratio_per_day_list,
     second_dose_vaccine_ratio_per_day_list, average_contact_id_list,
     average_contact_iud_list, reinfectable_rate_list, alpha_list,
     diagnosis_rate_list, recovery_rate_id_list, recovery_rate_iud_list,
     recovery_rate_h_list, recovery_rate_i_list, recovery_rate_v_list,
     hospitalization_rate_list, icu_rate_list, ventilator_rate_list,
     death_rate_id_list, death_rate_h_list, death_rate_i_list,
     death_rate_v_list) = seir_timeseries_variables

    first_dose_vaccine_ratio_per_day = model_utils.apply_relu_bounds(
        first_dose_vaccine_ratio_per_day_list[-1], 0.0, 1.0)
    second_dose_vaccine_ratio_per_day = model_utils.apply_relu_bounds(
        second_dose_vaccine_ratio_per_day_list[-1], 0.0, 1.0)
    average_contact_id = 1.0 * tf.nn.sigmoid(average_contact_id_list[-1])
    average_contact_iud = 1.0 * tf.nn.sigmoid(average_contact_iud_list[-1])
    reinfectable_rate = 0.001 * tf.nn.sigmoid(reinfectable_rate_list[-1])
    alpha = 0.2 * tf.nn.sigmoid(alpha_list[-1])
    diagnosis_rate = 0.01 + 0.09 * tf.nn.sigmoid(diagnosis_rate_list[-1])
    recovery_rate_id = 0.1 * tf.nn.sigmoid(recovery_rate_id_list[-1])
    recovery_rate_iud = 0.1 * tf.nn.sigmoid(recovery_rate_iud_list[-1])
    recovery_rate_h = 0.1 * tf.nn.sigmoid(recovery_rate_h_list[-1])
    recovery_rate_i = 0.1 * tf.nn.sigmoid(recovery_rate_i_list[-1])
    recovery_rate_v = 0.1 * tf.nn.sigmoid(recovery_rate_v_list[-1])
    hospitalization_rate = 0.1 * tf.nn.sigmoid(hospitalization_rate_list[-1])
    icu_rate = 0.1 * tf.nn.sigmoid(icu_rate_list[-1])
    ventilator_rate = 0.01 + 0.19 * tf.nn.sigmoid(ventilator_rate_list[-1])
    death_rate_id = 0.01 * tf.nn.sigmoid(death_rate_id_list[-1])
    death_rate_h = 0.1 * tf.nn.sigmoid(death_rate_h_list[-1])
    death_rate_i = 0.1 * tf.nn.sigmoid(death_rate_i_list[-1])
    death_rate_v = 0.01 + 0.09 * tf.nn.sigmoid(death_rate_v_list[-1])

    return (first_dose_vaccine_ratio_per_day, second_dose_vaccine_ratio_per_day,
            average_contact_id, average_contact_iud, reinfectable_rate, alpha,
            diagnosis_rate, recovery_rate_id, recovery_rate_iud,
            recovery_rate_h, recovery_rate_i, recovery_rate_v,
            hospitalization_rate, icu_rate, ventilator_rate, death_rate_id,
            death_rate_h, death_rate_i, death_rate_v)

  def initialize_ground_truth_timeseries(
      self,
      static_features,
      ts_features,
      chosen_locations,
      num_observed_timesteps,
      infected_threshold,
  ):
    """Returns initialized ground truth timeseries."""
    chosen_populations = static_features[constants.POPULATION]
    num_locations = len(chosen_locations)
    chosen_populations_list = np.zeros(num_locations, dtype=np.float32)

    gt_input = {
        "infected": ts_features[constants.INFECTED],
        "recovered": ts_features[constants.RECOVERED_DOC],
        "confirmed": ts_features[constants.CONFIRMED],
        "hospitalized": ts_features[constants.HOSPITALIZED],
        "hospitalized_cumulative":
            (ts_features[constants.HOSPITALIZED_CUMULATIVE]),
        "hospitalized_increase": ts_features[constants.HOSPITALIZED_INCREASE],
        "icu": ts_features[constants.ICU],
        "ventilator": ts_features[constants.VENTILATOR],
        "death": ts_features[constants.DEATH],
    }
    gt_names = list(gt_input.keys())

    gt_list = dict()
    gt_indicator = dict()

    # Initialize GT
    for gt_name in gt_names:
      gt_list[gt_name] = np.zeros((num_locations, num_observed_timesteps),
                                  dtype=np.float32)
      gt_indicator[gt_name] = np.zeros((num_locations, num_observed_timesteps),
                                       dtype=np.float32)

    # Assign GT
    for location_index, location in enumerate(chosen_locations):
      if chosen_populations[location]:
        chosen_populations_list[location_index] = chosen_populations[location]

      for gt_name in gt_names:
        gt_list[gt_name], gt_indicator[gt_name] = model_utils.populate_gt_list(
            location_index, gt_input[gt_name], location, num_observed_timesteps,
            gt_list[gt_name], gt_indicator[gt_name])

    for gt_name in gt_names:
      gt_list[gt_name] = tf.transpose(
          tf.constant(gt_list[gt_name], dtype=tf.float32))
      gt_indicator[gt_name] = tf.transpose(
          tf.constant(gt_indicator[gt_name], dtype=tf.float32))
    chosen_populations_list = tf.constant(
        chosen_populations_list, dtype=tf.float32)

    orig_gt = gt_input
    ground_truth_timeseries = (chosen_populations_list, gt_list, gt_indicator,
                               gt_names, orig_gt)
    infection_active_mask = model_utils.construct_infection_active_mask(
        gt_list["confirmed"], num_locations, num_observed_timesteps,
        infected_threshold)
    infection_active_mask = tf.constant(infection_active_mask, dtype=tf.float32)
    return ground_truth_timeseries, infection_active_mask

  def sync_values(
      self,
      hparams,
      last_state,
      ground_truth_timeseries,
      timestep,
      is_training,
  ):
    """See parent class."""
    sync_coef = hparams["sync_coef"]
    # At inference, we use sync_coef = 1.0
    if not is_training:
      sync_coef = 1.0

    # Unstack the last state (for updating)
    (exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
     hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t, icu_t,
     ventilator_t, death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
     reinfectable_vaccine_t, vaccine_immuned_t,
     infected_ud_increase_t) = tf.unstack(last_state)

    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    # Inferred from the confirmed condition
    recovered_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["recovered"][timestep],
        gt_indicator=gt_indicator["recovered"][timestep],
        compartment=recovered_d_t + reinfectable_d_t,
        sync_coef=sync_coef)
    hospitalized_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["hospitalized"][timestep],
        gt_indicator=gt_indicator["hospitalized"][timestep],
        compartment=hospitalized_t + icu_t + ventilator_t,
        sync_coef=sync_coef)
    icu_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["icu"][timestep],
        gt_indicator=gt_indicator["icu"][timestep],
        compartment=icu_t + ventilator_t,
        sync_coef=sync_coef)
    ventilator_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["ventilator"][timestep],
        gt_indicator=gt_indicator["ventilator"][timestep],
        compartment=ventilator_t,
        sync_coef=sync_coef)
    death_gt_est = model_utils.sync_compartment(
        gt_list=gt_list["death"][timestep],
        gt_indicator=gt_indicator["death"][timestep],
        compartment=death_t,
        sync_coef=sync_coef)

    # Infected
    infected_d_t = model_utils.sync_compartment(
        gt_list=(gt_list["confirmed"][timestep] - recovered_gt_est -
                 hospitalized_gt_est - death_gt_est),
        gt_indicator=gt_indicator["confirmed"][timestep],
        compartment=infected_d_t,
        sync_coef=sync_coef)

    # Recovered
    recovered_d_t = model_utils.sync_compartment(
        gt_list=gt_list["recovered"][timestep],
        gt_indicator=gt_indicator["recovered"][timestep],
        compartment=recovered_d_t + reinfectable_d_t,
        sync_coef=sync_coef)
    recovered_d_t = recovered_d_t - reinfectable_d_t

    # Hospitalized
    hospitalized_t = model_utils.sync_compartment(
        gt_list=gt_list["hospitalized"][timestep] - icu_gt_est,
        gt_indicator=gt_indicator["hospitalized"][timestep],
        compartment=hospitalized_t,
        sync_coef=sync_coef)

    # Hospitalized Cumulative
    hospitalized_cumulative_t = model_utils.sync_compartment(
        gt_list=gt_list["hospitalized_cumulative"][timestep],
        gt_indicator=gt_indicator["hospitalized_cumulative"][timestep],
        compartment=hospitalized_cumulative_t,
        sync_coef=sync_coef)

    # ICU
    icu_t = model_utils.sync_compartment(
        gt_list=gt_list["icu"][timestep] - ventilator_gt_est,
        gt_indicator=gt_indicator["icu"][timestep],
        compartment=icu_t,
        sync_coef=sync_coef)

    # Ventilator
    ventilator_t = model_utils.sync_compartment(
        gt_list=gt_list["ventilator"][timestep],
        gt_indicator=gt_indicator["ventilator"][timestep],
        compartment=ventilator_t,
        sync_coef=sync_coef)

    # Death
    death_t = model_utils.sync_compartment(
        gt_list=gt_list["death"][timestep],
        gt_indicator=gt_indicator["death"][timestep],
        compartment=death_t,
        sync_coef=sync_coef)

    # Update the last state
    last_state = tf.stack([
        exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
        hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
        icu_t, ventilator_t, death_t, population_t, reinfectable_d_t,
        reinfectable_ud_t, reinfectable_vaccine_t, vaccine_immuned_t,
        infected_ud_increase_t
    ])

    return last_state

  def sync_undoc(self, hparams, last_state,
                 ground_truth_timeseries,
                 last_variable, timestep,
                 is_training):
    """Synchronize the undocumented infected counts using confirmed increment.

    Args:
      hparams: Model's hyper-parameters. Usually contains sync_coef to define
        the amount of teacher forcing.
      last_state: The model's previous state.
      ground_truth_timeseries: The ground truth values to sync with.
      last_variable: The model's variables from the previous step. Should
        include the diagnosis rate.
      timestep: The current time step.
      is_training: True if the model is being trained.

    Returns:
      The updated values for the last_state.
    """
    # Divided by hparams["reduced_sync_undoc"] to synchronize with
    # less stable GT (GT times predicted values)
    sync_coef = hparams["sync_coef"] / hparams["reduced_sync_undoc"]
    # At inference, we use sync_coef = 1.0
    if not is_training:
      sync_coef = 1.0 / hparams["reduced_sync_undoc"]
    # Unstack the last state (for updating)
    (exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
     hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t, icu_t,
     ventilator_t, death_t, population_t, reinfectable_d_t, reinfectable_ud_t,
     reinfectable_vaccine_t, vaccine_immuned_t,
     infected_ud_increase_t) = tf.unstack(last_state)

    # Get the previous diagnosis rate
    diagnosis_rate = last_variable[6]

    (_, gt_list, gt_indicator, _, _) = ground_truth_timeseries

    confirmed_increase_gt = (
        gt_list["confirmed"][timestep] - gt_list["confirmed"][timestep - 1])
    confirmed_increase_gt_indicator = (
        gt_indicator["confirmed"][timestep] *
        gt_indicator["confirmed"][timestep - 1])

    infected_ud_t = model_utils.sync_compartment(
        gt_list=confirmed_increase_gt * (1.0 / diagnosis_rate),
        gt_indicator=confirmed_increase_gt_indicator,
        compartment=infected_ud_t,
        sync_coef=sync_coef)

    # Update the last state
    last_state = tf.stack([
        exposed_t, infected_d_t, infected_ud_t, recovered_d_t, recovered_ud_t,
        hospitalized_t, hospitalized_cumulative_t, hospitalized_increase_t,
        icu_t, ventilator_t, death_t, population_t, reinfectable_d_t,
        reinfectable_ud_t, reinfectable_vaccine_t, vaccine_immuned_t,
        infected_ud_increase_t
    ])

    return last_state

  def seir_dynamics(self, current_state,
                    seir_variables):
    raise NotImplementedError("To be ported from the model_constructor class")

  def compute_losses(
      self, hparams, propagated_states,
      ground_truth_timeseries
  ):
    raise NotImplementedError("To be ported from the model_constructor class")


def _add_vaccine_features(example_feature,
                          transformed_features,
                          feature_scalers):
  """Adds VACCINATED_RATIO and VACCINE_EFFECTIVENESS to the features.

    Updates transformed_features and feature_scalers in place.

  Args:
    example_feature: An example feature whose shape should be copied.
    transformed_features: A dictionary with transformed features.
    feature_scalers: A dictionary with feature scalers.
  """
  # Make vaccines have the same dimensions as deaths for now.
  first_dose_vaccine_ratio_per_day = {}
  second_dose_vaccine_ratio_per_day = {}
  first_dose_vaccine_effectiveness = {}
  second_dose_vaccine_effectiveness = {}
  for location, ex_data in example_feature.items():
    first_dose_vaccine_ratio_per_day[location] = np.zeros(
        ex_data.shape, dtype="float32")
    second_dose_vaccine_ratio_per_day[location] = np.zeros(
        ex_data.shape, dtype="float32")
    first_dose_vaccine_effectiveness[location] = np.ones(
        ex_data.shape,
        dtype="float32") * constants.FIRST_DOSE_VACCINE_MAX_EFFECT
    second_dose_vaccine_effectiveness[location] = np.ones(
        ex_data.shape,
        dtype="float32") * constants.SECOND_DOSE_VACCINE_MAX_EFFECT

  transformed_features.update({
      constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED:
          first_dose_vaccine_ratio_per_day,
      constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED:
          second_dose_vaccine_ratio_per_day,
      constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE:
          first_dose_vaccine_effectiveness,
      constants.VACCINATED_EFFECTIVENESS_SECOND_DOSE:
          second_dose_vaccine_effectiveness,
  })

  feature_scalers.update({
      constants.VACCINATED_RATIO_FIRST_DOSE_PER_DAY_PREPROCESSED: None,
      constants.VACCINATED_RATIO_SECOND_DOSE_PER_DAY_PREPROCESSED: None,
      constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE: None,
      constants.VACCINATED_EFFECTIVENESS_SECOND_DOSE: None,
  })

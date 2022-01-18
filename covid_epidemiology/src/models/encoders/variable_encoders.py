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

# Lint as: python3
"""Encoders that use time series inputs to predict SEIR terms."""
import abc
import logging
import sys
from typing import List
import numpy as np
import tensorflow as tf
from covid_epidemiology.src import constants
from covid_epidemiology.src.models.shared import feature_utils


class VariableEncoder(abc.ABC):
  """Encodes time series variables."""

  @abc.abstractmethod
  def encode(self,
             time_series_input,
             timestep,
             is_training = False):
    """Encodes the time series variables.

    Args:
      time_series_input: List of tf.tensors of size [currnet timestep]. Each
        element of time_series_input is a tf.tensor of shape [num_locations].
      timestep: Point in time relative to the beginning of time_series_input to
        predict from. Has to be <= len(time_series_input).
      is_training: Whether the model is currently training or performing
        inference.

    Returns:
      tf.tensor of shape [num_locations]
    """
    raise NotImplementedError

  @abc.abstractproperty
  def trainable_variables(self):
    raise NotImplementedError

  @property
  def lasso_loss(self):
    return 0.


class StaticEncoder(VariableEncoder):
  """Static encoder with fixed parameters."""

  def encode(self,
             time_series_input,
             timestep,
             is_training = False):
    del timestep, is_training  # Unused.
    return time_series_input[-1]

  @property
  def trainable_variables(self):
    return []


class PassThroughEncoder(VariableEncoder):
  """Encoder that outputs the input without changing.

  PassThroughEncoder is required to have a single covariate input, and that
  input is required to be categorical (i.e., it must have absolute overrides,
  not relative).
  """

  def __init__(self,
               chosen_locations,
               num_known_timesteps,
               forecast_window_size,
               covariates,
               forecasted_covariates,
               covariate_overrides,
               covariate_feature_specs,
               ts_categorical_features = None,
               name = ""):
    self.num_known_timesteps = num_known_timesteps
    covariates_over_time = []
    for timestep in range(num_known_timesteps):
      covariates_this_timestep = np.zeros((len(chosen_locations), 1),
                                          dtype="float32")
      if len(covariate_feature_specs) > 1:
        raise ValueError(
            "Only one covariate is supported for PassThroughEncoder.")
      feature_spec = covariate_feature_specs[0]
      if covariate_overrides is not None:
        if (ts_categorical_features is None or
            feature_spec.name not in ts_categorical_features):
          raise ValueError(
              ("Only categorical features are supported by PassThroughEncoder. "
               f"{feature_spec.name} not in {ts_categorical_features}"))
      if feature_spec.name in covariates:
        for location_index, location in enumerate(chosen_locations):
          if location in covariates[feature_spec.name]:
            covariates_this_timestep[location_index] = (
                covariates[feature_spec.name][location][timestep])
      else:
        raise ValueError(
            "Wrong feature name specified in covariate_feature_specs.")
      covariates_over_time.append(covariates_this_timestep)
    self.covariates_over_time = covariates_over_time

    self.name = name
    # Extract covariates name for each encoder
    self.forecasted_feature_values = (
        feature_utils.extract_forecasted_features(forecasted_covariates,
                                                  covariate_feature_specs))
    self.forecasted_feature_values = tf.constant(
        np.array(self.forecasted_feature_values), dtype=tf.float32)

    if covariate_overrides is not None:
      self.covariate_feature_overrides = feature_utils.covariate_overrides_to_dense(
          covariate_overrides, covariate_feature_specs or {}, chosen_locations,
          num_known_timesteps + forecast_window_size)
    else:
      self.covariate_feature_overrides = None

  def encode(self,
             time_series_input,
             timestep,
             is_training = False):
    potential_timestep = timestep - 1
    if potential_timestep >= 0:
      output_values = self.get_overriden_covariate_values_passthrough(
          potential_timestep)
    else:
      output_values = tf.zeros(tf.shape(self.covariates_over_time[0]))
    # output_values is of shape (num locations, 1). we must convert it into
    # the shape (num_locations) before returning.  E.g.,
    # output_values = [[0.3]
    #                  [0.3]] becomes [0.3 0.3].
    assert output_values.shape[1] == 1
    return output_values[:, -1]

  def _get_overriden_covariate_values(self, potential_timestep):
    """Overrides the covariates depending on the type of covariate and timestep.

    Calculate the overridden covariate values.

    Args:
      potential_timestep: integer. Timestep to compute overriden covariates.

    Returns:
      Tensor with the computed overriden covariate at that timestep.
    """
    # Note that if we move to potential_timestamp being a tensor we will have
    # to move from the chained comparison to an and statement.
    if 0 <= potential_timestep < self.num_known_timesteps:
      current_covariate_values = self.covariates_over_time[potential_timestep]
    else:
      value_timestep = potential_timestep - self.num_known_timesteps

      current_covariate_values = (
          self.forecasted_feature_values[value_timestep])

    if self.covariate_feature_overrides is not None:
      covariate_values_merged = np.where(
          self.covariate_feature_overrides[potential_timestep] == -1,
          current_covariate_values,
          self.covariate_feature_overrides[potential_timestep])

      # Log the overrides for debugging purposes.
      change_mask = np.where(
          self.covariate_feature_overrides[potential_timestep] != -1, 1, 0)

      change_ixs = np.nonzero(change_mask)
      indices = np.dstack(change_ixs)

      if change_ixs[0].size != 0:
        np.set_printoptions(threshold=sys.maxsize)
        current_covariate_values_np = np.asarray(current_covariate_values)
        covariate_values_merged_np = np.asarray(covariate_values_merged)

        logging.debug(
            "Pass-through covariate changes at t=%d for %s\n"
            "for [location index, covariate index] =\n %s:\n"
            "%s overridden by %s results in\n%s", potential_timestep, self.name,
            indices, current_covariate_values_np[change_ixs],
            self.covariate_feature_overrides[potential_timestep][change_ixs],
            covariate_values_merged_np[change_ixs])
    else:
      covariate_values_merged = current_covariate_values

    return covariate_values_merged

  def get_overriden_covariate_values_passthrough(self, potential_timestep):
    """This wrapper makes mocking this function possible."""
    return self._get_overriden_covariate_values(potential_timestep)

  @property
  def trainable_variables(self):
    return []


class VaccineEncoder(VariableEncoder):
  """Encoder for Vaccinations.

  VaccineEncoder is required to have a single covariate input.
  """

  def __init__(self,
               chosen_locations,
               num_known_timesteps,
               forecast_window_size,
               covariates,
               forecasted_covariates,
               covariate_overrides,
               covariate_feature_specs,
               ts_categorical_features = None,
               name = "",
               vaccine_type = "first_dose",
               trend_following = True):
    self.num_known_timesteps = num_known_timesteps

    covariates_over_time_all = dict()
    for feature_spec in covariate_feature_specs:
      covariates_over_time = []
      for timestep in range(num_known_timesteps):
        covariates_this_timestep = np.zeros((len(chosen_locations), 1),
                                            dtype="float32")
        if feature_spec.name in covariates:
          for location_index, location in enumerate(chosen_locations):
            if location in covariates[feature_spec.name]:
              covariates_this_timestep[location_index] = (
                  covariates[feature_spec.name][location][timestep])
        else:
          raise ValueError(
              "Wrong feature name specified in covariate_feature_specs.")
        covariates_over_time.append(covariates_this_timestep)
      covariates_over_time_all[feature_spec.name] = covariates_over_time.copy()

    self.covariates_over_time_all = covariates_over_time_all

    self.name = name
    self.vaccine_type = vaccine_type
    self.covariate_feature_specs = covariate_feature_specs

    self.forecasted_feature_values_all = dict()
    for feature_spec in covariate_feature_specs:
      # Extract covariates name for each encoder
      if trend_following:
        # Same daily vaccinated ratio for future.
        # Will disable if XGBoost properly forecast.
        # Currently, it forecasted 0 for all.
        # Note that averaging is another option.
        self.forecasted_feature_values_all[
            feature_spec.name] = covariates_over_time_all[
                feature_spec.name].copy()[-forecast_window_size:]
      else:
        self.forecasted_feature_values_all[feature_spec.name] = (
            feature_utils.extract_forecasted_features(
                forecasted_covariates[feature_spec.name], [feature_spec]))
        self.forecasted_feature_values_all[feature_spec.name] = tf.constant(
            np.array(self.forecasted_feature_values_all[feature_spec.name]),
            dtype=tf.float32)

    self.covariate_feature_overrides_all = dict()
    for feature_spec in covariate_feature_specs:
      if covariate_overrides is not None:
        self.covariate_feature_overrides_all[
            feature_spec.name] = feature_utils.covariate_overrides_to_dense(
                covariate_overrides, [feature_spec] or {}, chosen_locations,
                num_known_timesteps + forecast_window_size)
      else:
        self.covariate_feature_overrides_all[feature_spec.name] = None

  def encode(self,
             time_series_input,
             timestep,
             is_training = False):
    potential_timestep = timestep - 1
    if potential_timestep >= 0:
      output_values = self.compute_immuned_patients(potential_timestep)
    else:
      output_values = tf.zeros(
          tf.shape(self.covariates_over_time_all[
              self.covariate_feature_specs[0].name][0]))
    # output_values is of shape (num locations, 1). we must convert it into
    # the shape (num_locations) before returning.  E.g.,
    # output_values = [[0.3]
    #                  [0.3]] becomes [0.3 0.3].
    assert output_values.shape[1] == 1
    return output_values[:, -1]

  def compute_immuned_patients(self, potential_timestep):
    """Compute immuned patients for first and second dosage vaccination.

    Args:
      potential_timestep: Timestep to compute immuned patients.

    Returns:
      immuned_patient_count: Number of immuned patient no via certain dosage.
    """

    if self.vaccine_type == "first_dose":
      vaccine_effect_diff = self.get_overriden_covariate_values_passthrough(
          0, constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE)
    if self.vaccine_type == "second_dose":
      vaccine_effect_diff = (
          self.get_overriden_covariate_values_passthrough(
              0, constants.VACCINATED_EFFECTIVENESS_SECOND_DOSE) -
          self.get_overriden_covariate_values_passthrough(
              0, constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE))

    immuned_patient_count = (
        vaccine_effect_diff * self.get_overriden_covariate_values_passthrough(
            0, self.covariate_feature_specs[0].name))

    for time_index in range(potential_timestep):

      if self.vaccine_type == "first_dose":
        vaccine_effect_diff = self.get_overriden_covariate_values_passthrough(
            potential_timestep - time_index,
            constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE)
      if self.vaccine_type == "second_dose":
        vaccine_effect_diff = (
            self.get_overriden_covariate_values_passthrough(
                potential_timestep - time_index,
                constants.VACCINATED_EFFECTIVENESS_SECOND_DOSE) -
            self.get_overriden_covariate_values_passthrough(
                potential_timestep - time_index,
                constants.VACCINATED_EFFECTIVENESS_FIRST_DOSE))

      current_vaccine_effect = np.minimum(
          (vaccine_effect_diff / constants.VACCINE_EFFECTIVENESS_CHANGE_PERIOD)
          * time_index, vaccine_effect_diff)

      current_vaccinated_count = (
          self.get_overriden_covariate_values_passthrough(
              potential_timestep - time_index,
              self.covariate_feature_specs[0].name))
      immuned_patient_count += current_vaccinated_count * current_vaccine_effect

    return immuned_patient_count

  def _get_overriden_covariate_values(self, potential_timestep,
                                      feature_spec_name):
    """Overrides the covariates depending on the type of covariate and timestep.

    Calculate the overridden covariate values.

    Args:
      potential_timestep: integer. Timestep to compute overriden covariates.
      feature_spec_name: Name of the covariate

    Returns:
      Tensor with the computed overriden covariate at that timestep.
    """
    if 0 <= potential_timestep < self.num_known_timesteps:
      current_covariate_values = self.covariates_over_time_all[
          feature_spec_name][potential_timestep]
    else:
      value_timestep = potential_timestep - self.num_known_timesteps

      current_covariate_values = (
          self.forecasted_feature_values_all[feature_spec_name][value_timestep])

    if self.covariate_feature_overrides_all[feature_spec_name] is not None:
      covariate_values_merged = np.where(
          self.covariate_feature_overrides_all[feature_spec_name]
          [potential_timestep] == -1, current_covariate_values,
          self.covariate_feature_overrides_all[feature_spec_name]
          [potential_timestep])

      # Log the overrides for debugging purposes.
      change_mask = np.where(
          self.covariate_feature_overrides_all[feature_spec_name]
          [potential_timestep] != -1, 1, 0)

      change_ixs = np.nonzero(change_mask)
      indices = np.dstack(change_ixs)

      if change_ixs[0].size != 0:
        np.set_printoptions(threshold=sys.maxsize)
        current_covariate_values_np = np.asarray(current_covariate_values)
        covariate_values_merged_np = np.asarray(covariate_values_merged)

        logging.debug(
            "Pass-through covariate changes at t=%d for %s\n"
            "for [location index, covariate index] =\n %s:\n"
            "%s overridden by %s results in\n%s", potential_timestep, self.name,
            indices, current_covariate_values_np[change_ixs],
            self.covariate_feature_overrides_all[feature_spec_name]
            [potential_timestep][change_ixs],
            covariate_values_merged_np[change_ixs])
    else:
      covariate_values_merged = current_covariate_values

    return covariate_values_merged

  def get_overriden_covariate_values_passthrough(self, potential_timestep,
                                                 feature_spec_name):
    """This wrapper makes mocking this function possible."""

    # The function name must be unique among {GamEncoder, PassThroughEncoder}
    # functions so that both GamEncoder and PassThroughEncider functions can be
    # mocked in the same test function.
    return self._get_overriden_covariate_values(potential_timestep,
                                                feature_spec_name)

  @property
  def trainable_variables(self):
    return []

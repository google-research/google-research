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

"""The base compartmental model definition.

This base class can be extended to fully define a customized compartmental
disease model that incorporates static and dynamic covariates by learning
adaptive encoders that modify the model's compartmental transitions.
"""

import abc
import collections
import functools
import logging
from typing import List, Union
# from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union


import numpy as np
import pandas as pd
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src import feature_preprocessing as preprocessing
from covid_epidemiology.src.models import generic_seir_model_constructor
from covid_epidemiology.src.models.encoders import gam_encoder
from covid_epidemiology.src.models.encoders import variable_encoder_builder
from covid_epidemiology.src.models.encoders import variable_encoders
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib
from covid_epidemiology.src.models.shared import typedefs

# pylint: disable=invalid-name
_ENCODER_TYPES = Union[gam_encoder.GamEncoder, variable_encoders.StaticEncoder,
                       variable_encoders.PassThroughEncoder,
                       variable_encoders.VaccineEncoder]


# noinspection PyMethodMayBeStatic
class BaseModelDefinition(abc.ABC):
  """Defines the structure and dynamics of the compartmental model.

  Attributes:
    ts_preprocessing_config: The default configuration to use for pre-processing
      time-series features.
    static_preprocessing_config: The default configuration to use for
      pre-processing static features.
    random_seed: A number to be used as a random seed for the model.
  """

  # The name of the column in the static and ts dataframe that has the location.
  _LOCATION_COLUMN_NAME: str = constants.GEO_ID_COLUMN

  # The list of rates that the encoders will predict
  # This should be implemented by the sub-class
  _ENCODER_RATE_LIST: List[str] = []

  def __init__(
      self,
      ts_preprocessing_config = None,
      static_preprocessing_config = None,
      random_seed = 0,
      **kwargs,  # pylint:disable=unused-argument
  ):
    """Creates the compartmental model.

    Args:
      ts_preprocessing_config: The default configuration to use for
        pre-processing time-series features.
      static_preprocessing_config: The default configuration to use for
        pre-processing static features.
      random_seed: A random seed to use for the model.
      **kwargs: Model specific keyword arguments.
    """
    #   FeatureConfig.
    self.ts_preprocessing_config: preprocessing.FeaturePreprocessingConfig = (
        ts_preprocessing_config or preprocessing.FeaturePreprocessingConfig())
    self.static_preprocessing_config: preprocessing.FeaturePreprocessingConfig = (
        static_preprocessing_config or
        preprocessing.FeaturePreprocessingConfig())
    self.random_seed = random_seed

  @abc.abstractmethod
  def get_ts_features(self):
    """Gets mapping of feature aliases to feature names for time series.

    The feature aliases must match the associated "feature_name" in the
    input_ts_table_id BiqQuery table.

    Returns:
      Mapping from feature names to the "feature_name" column value in the
      BigQuery table.
    """

  @abc.abstractmethod
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

  @abc.abstractmethod
  def get_ts_features_to_preprocess(self):
    """Get a list of time series features to pre-process.

    Returns:
      A list of feature aliases.
    """

  @abc.abstractmethod
  def get_static_features(self):
    """"Gets mapping of feature aliases to feature names for time series.

    The feature aliases must match the associated "feature_name" in the
    input_static_table_id BiqQuery table.

    Returns:
      Mapping from feature names to the "feature_name" column value in the
      BigQuery table.
    """
    #  feature configuration in one location.

  @abc.abstractmethod
  def get_model_spec(self, *args, **kwargs):
    """Returns the model spec.

    This defines the encoders and hyper-parameters for the model.

    Args:
      *args: model spec related args.
      **kwargs: model spec related kwargs.

    Returns:
      The ModelSpec object for this model
    """

  @abc.abstractmethod
  def seir_dynamics(self, current_state,
                    seir_variables):
    """Returns the derivatives of each state for SEIR dynamics.

    Args:
      current_state: The values of the model's compartments.
      seir_variables: The values of the model's transition variables

    Returns:
      The derivative of the state so that the next state is the current state
        plus the output of this function.
    """

  @abc.abstractmethod
  def compute_losses(
      self, hparams, propagated_states,
      ground_truth_timeseries
  ):
    """Calculates the loss between the propagates states and the ground truth.

    Args:
      hparams: Dictionary of the models hyperparameters.
      propagated_states: The model's current states.
      ground_truth_timeseries: The ground truth time series

    Returns:
      The model's loss
    """

  @abc.abstractmethod
  def transform_static_features(
      self, static_features
  ):
    """Transforms static features (scales them, removes NaNs, etc).

       Can also create new features (e.g., ratios).

    Args:
      static_features: A mapping from the feature name to its value, the value
        of each feature is a map from location to a value .

    Returns:
      A mapping from the feature name to its value and its fitted scaler after
      being prepared for the encoders, the value of each feature is a map from
      location to a float.
    """

  @abc.abstractmethod
  def bound_variables(
      self,
      seir_timeseries_variables,
  ):
    """Maps the encoded SEIR variables into realistic bounds for the model.

    This is called at the beginning of the propagation loop.

    Args:
      seir_timeseries_variables: The model's SEIR variables.

    Returns:
      The model's SEIR variables after their have been bounded approriately.
    """

  @abc.abstractmethod
  def initialize_ground_truth_timeseries(
      self,
      static_features,
      ts_features,
      chosen_locations,
      num_observed_timesteps,
      infected_threshold,
  ):
    """Creates the ground truth data structure from the features.

    Args:
      static_features: The static features as a dictionary of dictionaries.
      ts_features: The time series data as a dictionary of dictionaries.
      chosen_locations: A list of the locations that will be processed.
      num_observed_timesteps: The total number of observed data points.
      infected_threshold: The minimum number of infections to consider the virus
        to be active.

    Returns:
      The ground truth data as a Tuple of:
        1. A tensor with populations for each of the locations.
        2. A dictionary where the keys are features and the values are tensors
          (n_locations x num_observed_timesteps) of the feature's values.
        3. A dictionary where the keys are features and the values are tensors
          (n_locations x num_observed_timesteps) that are when if the feature is
           valid.
        4. A list of all the ground truth feature names.
        5. A dictionary where the keys are ground truth compartment names and
           the values are the unmodified original ground truth values.
      A mask of if the infection is active which is determined when the number
        of confirmed cases at a given location is above the infected threshold.
        This float32 tensor is of size (num_locations x num_observed_timesteps.
    """

  def initialize_components(
      self,
      model_spec,
      ground_truth_timeseries,
      infected_threshold,
      hparams,
      num_locations,
      location_dependent_init,
      chosen_locations,
      num_observed_timesteps,
      forecast_window_size,
      output_window_size,
      static_features,
      static_overrides,
      ts_categorical_features,
      covariates,
      forecasted_covariates,
      covariate_overrides,
      static_scalers = None,
      ts_scalers = None,
      trainable = True,
  ):
    """Initializes states, variables, and encoders.

    Args:
      model_spec: The model specification used for extracting the encoder specs.
      ground_truth_timeseries: A tuple of the populations of the chosen
        locations, the ground truth values for each compartment, the indicator
        for each ground truth value, and the names of the ground truth elements.
      infected_threshold: The minimum number of infections to consider the
        infection to be active in each location.
      hparams: Model hyper-parameters.
      num_locations: The number of locations that will be predicted for.
      location_dependent_init: If true different locations will have different
        bias terms.
      chosen_locations: A list of all the locations to use in the model.
      num_observed_timesteps: The total number of observed time steps.
      forecast_window_size: The number of training time steps to use in the
        encoder.
      output_window_size: The number of time steps to forecast from the trained
        encoder.
      static_features: A dictionary of static feature values where each value is
        a map of location to value for that location.
      static_overrides: A dictionary of over-rides for the static features.
      ts_categorical_features: Which features are categorical.
      covariates: A dictionary of covariate values where each value is a map of
        location to array of time values.
      forecasted_covariates: A map from covariate names to numpy arrays that
        have n_forecast_timesteps x n_locations.
      covariate_overrides: Overrides of the covariate values.
      static_scalers: fitted scalers for static featutes.
      ts_scalers: fitted scalers for timeseries featutes.
      trainable: If False the variables will not be trainable.

    Returns:
      init_state: The initial SEIR states as a Tensor.
      init_variables: The initial SEIR rates as taken from the hyper-parameters.
      seir_encoders: The initialized encoders.

    """
    init_state = self.initialize_seir_state(
        ground_truth_timeseries=ground_truth_timeseries,
        infected_threshold=infected_threshold,
        trainable=trainable,
    )

    init_variables = self.initialize_seir_variables(
        hparams=hparams,
        num_locations=num_locations,
        location_dependent_init=location_dependent_init,
        trainable=trainable,
    )

    seir_encoders = self.initialize_encoders(
        model_spec=model_spec,
        chosen_locations=chosen_locations,
        num_observed_timesteps=num_observed_timesteps,
        forecast_window_size=forecast_window_size,
        output_window_size=output_window_size,
        static_features=static_features,
        static_overrides=static_overrides,
        covariates=covariates,
        forecasted_covariates=forecasted_covariates,
        covariate_overrides=covariate_overrides,
        ts_categorical_features=ts_categorical_features,
        static_scalers=static_scalers,
        ts_scalers=ts_scalers,
        trainable=trainable,
    )

    return init_state, init_variables, seir_encoders

  @abc.abstractmethod
  def initialize_seir_state(
      self, ground_truth_timeseries,
      infected_threshold, trainable):
    """Returns initialized states for seir dynamics."""

  @abc.abstractmethod
  def sync_values(
      self,
      hparams,
      last_state,
      ground_truth_timeseries,
      timestep,
      is_training,
  ):
    """Syncs values with ground truth.

    This is used to implement partial teacher forcing and is used to update the
    last state prior to calling `seir_dynamics`.

    Args:
      hparams: Model's hyper-parameters. Usually contains sync_coef to define
        the amount of teacher forcing.
      last_state: The model's previous state
      ground_truth_timeseries: The ground truth values to sync with.
      timestep: The current timestep
      is_training: A boolean scalar Tensor. True if the model is being trained.

    Returns:
      The updated values for the last_state.
    """

  @abc.abstractmethod
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
      is_training: A boolean scalar Tensor. True if the model is being trained.

    Returns:
      The updated values for the last_state.
    """

  @abc.abstractmethod
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
      The model constructor instance for the model.
    """

  def initialize_seir_variables(self, hparams, num_locations,
                                location_dependent_init,
                                trainable):
    """Returns initialized variables for SEIR terms."""
    np.random.seed(self.random_seed)

    degrees_of_freedom = num_locations if location_dependent_init else 1
    init_rates = []
    for rate_name in self._ENCODER_RATE_LIST:
      init_rates.append(hparams[rate_name + '_init'] *
                        np.ones(degrees_of_freedom))

    variable_list = np.asarray(init_rates)

    seir_variables = tf.Variable(
        variable_list, dtype=tf.float32, trainable=trainable)
    return seir_variables

  def initialize_encoders(
      self,
      model_spec,
      chosen_locations,
      num_observed_timesteps,
      forecast_window_size,
      output_window_size,
      static_features,
      static_overrides,
      covariates,
      forecasted_covariates,
      covariate_overrides,
      ts_categorical_features,
      static_scalers = None,
      ts_scalers = None,
      trainable = True,
  ):
    """Returns a set of initialized encoders for updating SEIR variables.

    Args:
      model_spec: The specification of the model and it's encoders.
      chosen_locations: The locations to use data from.
      num_observed_timesteps: The number of total time steps to use.
      forecast_window_size: The number of time points to use in creating the
        forecast.
      output_window_size: The number of time points to forecast into the future.
      static_features: Static features that will be used for the encoder.
      static_overrides: Overrides of the static features.
      covariates: Time-varying covariates for the encoders.
      forecasted_covariates: Forecast of time points.
      covariate_overrides: Overrides for time-varying covariates
      ts_categorical_features: Features that are categorical.
      static_scalers: fitted scalers for static featutes.
      ts_scalers: fitted scalers for timeseries featutes.
      trainable: If False the encoder variables will not be trainable.

    Returns:
      A tuple of all the initialized encoders.
    """
    encoders = list()
    for rate_name in self._ENCODER_RATE_LIST:
      encoders.append(
          variable_encoder_builder.encoder_from_encoder_spec(
              self.get_encoder_by_name(model_spec.encoder_specs, rate_name),
              chosen_locations=chosen_locations,
              num_known_timesteps=num_observed_timesteps,
              forecast_window_size=forecast_window_size,
              output_window_size=output_window_size,
              static_features=static_features,
              static_overrides=static_overrides,
              covariates=covariates,
              covariate_overrides=covariate_overrides,
              ts_categorical_features=ts_categorical_features,
              forecasted_covariates=forecasted_covariates,
              random_seed=self.random_seed,
              static_scalers=static_scalers,
              ts_scalers=ts_scalers,
              trainable=trainable,
          ))

    return tuple(encoders)

  def extract_all_features(
      self,
      static_data,
      ts_data,
      locations,
      training_window_size,
  ):
    """Creates time-series and static feature dictionaries from data frames.

    Args:
      static_data: Static data.
      ts_data: Time series data.
      locations: Locations to be extracted.
      training_window_size: Time-series data points to use for training.

    Returns:
      The static series dictionary mapping features to values where the values
        are a mapping of locations to a single numeric value.
      The time series dictionary mapping features to values where the values are
        a mapping of locations to time series values.
    """
    if static_data is None:
      static_features_and_scaler = (None, None)
    else:
      static_features_and_scaler = self._extract_static_features(
          static_data, locations)

    if ts_data is None or static_data is None:
      ts_features_and_scaler = (None, None)
    else:
      (static_features, _) = static_features_and_scaler
      ts_features_and_scaler = self._extract_ts_features(
          ts_data, static_features, locations, training_window_size)
    return static_features_and_scaler, ts_features_and_scaler

  def _extract_ts_features(
      self, ts_data, static_features,
      locations, training_window_size
  ):
    """Creates time-series feature dictionaries from data frame.

    This is an internal function to allow for feature engineering using both
    static and time series features.

    Args:
      ts_data: Time series DataFrame with columns constants.FEATURE_NAME_COLUMN,
        `ModelDefinition._LOCATION_COLUMN_NAME`, constants.DATE_COLUMN, and
        constants.FEATURE_VALUE_COLUMN.
      static_features: Static features.
      locations: Locations to be extracted.
      training_window_size: Time-series data points to use for training.

    Returns:
      The time series dictionary mapping features to values where the values are
        a mapping of locations to time series values, and the fitted scalers.
    """
    all_dates = preprocessing.get_all_valid_dates(ts_data)

    ts_features = preprocessing.ts_feature_df_to_nested_dict(
        ts_data,
        locations,
        all_dates,
        self.get_ts_features(),
        self._LOCATION_COLUMN_NAME,
    )
    proc_features, feature_scalers = self.transform_ts_features(
        ts_features=ts_features,
        static_features=static_features,
        initial_train_window_size=training_window_size)
    return proc_features, feature_scalers

  def _extract_static_features(
      self, static_data, locations
  ):
    """Creates a static feature dictionary from a data frame.

    This is an internal function to allow for feature engineering using both
    static and time series features.

    Args:
      static_data: Static DataFrame with columns constants.FEATURE_NAME_COLUMN,
        `ModelDefinition._LOCATION_COLUMN_NAME`, and
        constants.FEATURE_VALUE_COLUMN.
      locations: List of locations to extract.

    Returns:
      The static dictionary mapping features to values where the values are
        a mapping of locations to static values, and the fitted scalers.
    """
    static_features = collections.defaultdict(
        functools.partial(collections.defaultdict, lambda: None))
    static_feature_map = self.get_static_features()
    for feature_alias, feature_name in static_feature_map.items():
      feature_data = static_data[static_data[constants.FEATURE_NAME_COLUMN] ==
                                 feature_name]
      for location in locations:
        static_features[feature_alias][
            location] = preprocessing.static_covariate_value_or_none_for_location(
                feature_data, location, self._LOCATION_COLUMN_NAME)

    proc_features, feature_scalers = self.transform_static_features(
        static_features)
    return proc_features, feature_scalers

  def get_all_locations(self, input_df):
    """Gets a set of locations in the input data frame.

    Args:
      input_df: DataFrame with the column `_LOCATION_COLUMN_NAME`

    Returns:
      The set of all locations
    """
    return set(pd.unique(input_df[self._LOCATION_COLUMN_NAME]))

  def get_encoder_by_name(self, encoder_specs, name):
    for encoder_spec in encoder_specs:
      if encoder_spec.encoder_name == name:
        return encoder_spec
    raise ValueError(f'No encoder spec for requested encoder with name: {name}')

  def encode_variables(
      self,
      encoders,
      seir_timeseries_variables,
      global_biases,
      timestep,
      prediction,
      scaler,
      is_training,
  ):
    """Encodes the input variables to create an output time-series.

    Args:
      encoders: The encoders used to encode the time-series of the SEIR
        variables.
      seir_timeseries_variables: The variables to be encoded.
      global_biases: The global biases for each of the encoders. Is only used
        for GAM encoders.
      timestep: The time point being encoded.
      prediction: A dictionary of predictions from the model.
      scaler: A dictionary to transform predictions the same way that the
        variables have already been transformed (e.g. scaling to 0-1).
      is_training: True if the model is training.
    """
    for variable_index in range(len(encoders)):
      variable_encoder = encoders[variable_index]
      variable_list = seir_timeseries_variables[variable_index]
      if isinstance(variable_encoder, gam_encoder.GamEncoder):
        variable_bias = global_biases[variable_index]
        variable_list.append(
            variable_encoder.encode(variable_list, variable_bias, timestep,
                                    prediction, scaler, is_training))
      else:
        variable_list.append(
            variable_encoder.encode(variable_list, timestep, is_training))


class BaseCovidModelDefinition(BaseModelDefinition, abc.ABC):
  """Extends the base class with some common helper methods.

  This class takes advantage of additional assumptions about the model to
  consolidate the code for common tasks like feature pre-processing.
  """

  # The number of features that will be used for quantile estimation in the
  # method apply_quantile_transform.
  _NUM_QUANTILE_FEATURES: int = 7

  def get_static_features_to_preprocess(self):
    static_features_to_not_preprocess = {constants.POPULATION}
    return {
        feature_alias for feature_alias in self.get_static_features().keys()
        if feature_alias not in static_features_to_not_preprocess
    }

  def transform_static_features(
      self, static_features
  ):
    """Transforms static features (scales them, removes NaNs, etc).

       Can also create new features (e.g., ratios).

    Args:
      static_features: A mapping from the feature name to its value, the value
        of each feature is a map from location to a value .

    Returns:
      A mapping from the feature name to its value after being prepared for the
      encoders, the value of each feature is a map from location to a float.
    """
    # The static data must have population data for the compartments
    if constants.POPULATION not in static_features:
      raise ValueError(f'Static features must include {constants.POPULATION}')

    transformed_features = {}
    feature_scalers = {}
    self._standard_static_preprocessing(static_features, transformed_features,
                                        feature_scalers)
    return transformed_features, feature_scalers

  def _standard_static_preprocessing(
      self,
      raw_features,
      transformed_features,
      feature_scalers,
  ):
    """Pre-processes static features into an output dictionary.

    Args:
      raw_features: A dictionary with the features to be processed.
      transformed_features: The output dictionary.
      feature_scalers: dict of fitted scalers used to transform each feature.
    """

    to_preprocess = self.get_static_features_to_preprocess()
    for feature_name in raw_features:
      # Don't overwrite existing data
      if feature_name in transformed_features:
        continue

      if feature_name in to_preprocess:
        transformed_features[feature_name], feature_scalers[
            feature_name] = self._preprocess_static_feature(
                raw_features[feature_name])
      else:
        transformed_features[feature_name] = raw_features[feature_name]
        feature_scalers[feature_name] = None  # this feature was not scaled

  def _preprocess_static_feature(
      self, feature_data
  ):
    """Pre-process a single static feature.

    Args:
      feature_data: A single static feature dictionary.

    Returns:
       The input data after being pre-processed.
    """
    preprocessed_feature, scaler = preprocessing.preprocess_static_feature(
        feature_data, self.static_preprocessing_config.imputation_strategy,
        self.static_preprocessing_config.standardize)
    return preprocessed_feature, scaler

  def get_ts_features_to_preprocess(self):
    return set()

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
    transformed_features = {}
    feature_scalers = {}
    self._standard_ts_preprocessing(ts_features, initial_train_window_size,
                                    transformed_features, feature_scalers)
    return transformed_features, feature_scalers

  def _standard_ts_preprocessing(
      self,
      ts_features,
      initial_train_window_size,
      transformed_features,
      feature_scalers,
  ):
    """Do the normal time series pre-processing.

    This transfers over death and confirmed features and creates four new
    features that are named according to the constants:
      `constants.DEATH_PREPROCESSED`, `constants.CONFIRMED_PREPROCESSED`,
      `constants.CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO`, and
      `constants.DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO`.
    The features that are returned by `get_ts_features_to_preprocess` are
      pre-processed accordingly.

    Args:
      ts_features: A mapping from the feature name to its value, the value of
        each feature is a map from location to np.ndarray.
      initial_train_window_size: Size of initial training window.
      transformed_features: The output feature dictionary.
      feature_scalers: dict of fitted scalers used to transform each feature.
    """
    if constants.DEATH not in ts_features:
      raise ValueError(f'{constants.DEATH} must be in the input features')
    if constants.CONFIRMED not in ts_features:
      raise ValueError(f'{constants.CONFIRMED} must be in the input features')

    transformed_features.update({
        constants.DEATH: ts_features[constants.DEATH],
        constants.CONFIRMED: ts_features[constants.CONFIRMED],
    })

    # Need to pre-process our two new added fields
    features_to_preprocess = (
        self.get_ts_features_to_preprocess()
        | {constants.DEATH_PREPROCESSED, constants.CONFIRMED_PREPROCESSED})
    for feature_name, feature_location_dictionary in ts_features.items():
      if feature_name == constants.DEATH:
        feature_name = constants.DEATH_PREPROCESSED
      elif feature_name == constants.CONFIRMED:
        feature_name = constants.CONFIRMED_PREPROCESSED

        # Don't overwrite existing data
        if feature_name in transformed_features:
          continue

      if feature_name in features_to_preprocess:
        logging.info('Preprocessing feature: %s', feature_name)
        transformed_features[feature_name], feature_scalers[
            feature_name] = self._preprocess_ts_feature(
                feature_location_dictionary, initial_train_window_size)
      else:
        transformed_features[feature_name] = ts_features[feature_name]
        feature_scalers[feature_name] = None  # this feature was not scaled

    transformed_features[
        constants.
        CONFIRMED_PREPROCESSED_MEAN_TO_SUM_RATIO] = preprocessing.construct_feature_ratios(
            transformed_features[constants.CONFIRMED_PREPROCESSED])

    transformed_features[
        constants.
        DEATH_PREPROCESSED_MEAN_TO_SUM_RATIO] = preprocessing.construct_feature_ratios(
            transformed_features[constants.DEATH_PREPROCESSED])

  def _preprocess_ts_feature(
      self,
      feature_data,
      initial_train_window_size,
      bfill_features = None,
      imputation_strategy = None,
      standardize = None,
      initial_value = None,
  ):
    """Pre-process a single time-series feature.

    Args:
      feature_data: A single time-series feature dictionary.
      initial_train_window_size: The training window size.
      bfill_features: Backward fill imputation for time-series data.
      imputation_strategy: Additional imputation after ffill and bfill
      standardize: Flag to indicate whether this feature is standardized.
      initial_value: If None no actions will be taken. Otherwise, the first
        value for each location will be set to this value if it is null.

    Returns:
       The input data after being pre-processed and the fitted scaler.
    """

    preprocessed_feature, scaler = preprocessing.preprocess_ts_feature(
        feature_data,
        ffill_features=self.ts_preprocessing_config.ffill_features,
        bfill_features=self.ts_preprocessing_config.bfill_features
        if bfill_features is None else bfill_features,
        imputation_strategy=self.ts_preprocessing_config.imputation_strategy
        if imputation_strategy is None else imputation_strategy,
        standardize=self.ts_preprocessing_config.standardize
        if standardize is None else standardize,
        fitting_window=initial_train_window_size,
        initial_value=initial_value,
    )
    return preprocessed_feature, scaler

  @abc.abstractmethod
  def get_model_spec(self,
                     model_type,
                     covariate_delay = 0,
                     **kwargs):
    f"""Return the model spec.

    Args:
      model_type: The type of the model. Currently supports:
        {constants.MODEL_TYPE_STATIC_SEIR}
        {constants.MODEL_TYPE_TIME_VARYING_WITH_COVARIATES}
        {constants.MODEL_TYPE_TREND_FOLLOWING}
      covariate_delay: The amount to delay the covariates. Defaults to 0.
      **kwargs: Additional kwargs.

    Returns:
      The corresponding model spec.
    """  # pylint: disable=pointless-statement

  def initialize_quantile_variables(
      self,
      hparams,
      num_quantiles,
  ):
    """Creates the trainable tensors for quantile regression.

    Args:
      hparams: The hyperparameters including the quantile_encoding_window.
      num_quantiles: The number of output quantiles (e.g. 23).

    Returns:
      The 3D trainable kernel for the quantile estimation. Of size:
        quantile_encoding_window * number of features x number of states x
        number of quantiles.
      The 2D trainable biases for the quantile estimation. Of size:
        number of states x number of quantiles.
    """
    quantile_encoding_window = hparams['quantile_encoding_window']

    initial_kernel = np.zeros(
        (quantile_encoding_window * self._NUM_QUANTILE_FEATURES,
         self._NUM_STATES, num_quantiles))
    quantile_kernel = tf.Variable(initial_kernel, dtype=tf.float32)

    initial_biases = (0.1 / num_quantiles) * np.ones(
        (self._NUM_STATES, num_quantiles))
    quantile_biases = tf.Variable(initial_biases, dtype=tf.float32)

    return quantile_kernel, quantile_biases

  def gt_scaler(
      self,
      ground_truth_timeseries,
      num_time_steps,
  ):
    """Get min/max values of each covariate.

    These we be used to scale the predictions so that they match the
    preprocessed features created from the ground truth.

    Args:
      ground_truth_timeseries: The ground truth data including the GT confirmed
        and death values.
      num_time_steps: The number of time steps over which to compute the maximum
        and minimum values.

    Returns:
      A dictionary with keys of confirmed and death where each value is a
        dictionary of the minimum and maximum values in the time range.
    """
    (_, gt_list, _, _, _) = ground_truth_timeseries

    confirmed_scaler = {
        'min': np.min(gt_list['confirmed'][:num_time_steps]),
        'max': np.max(gt_list['confirmed'][:num_time_steps]),
    }

    death_scaler = {
        'min': np.min(gt_list['death'][:num_time_steps]),
        'max': np.max(gt_list['death'][:num_time_steps]),
    }

    return {'confirmed': confirmed_scaler, 'death': death_scaler}

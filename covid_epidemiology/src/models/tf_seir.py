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
"""Tensorflow model for multi-horizon forecasts with SEIR dynamics."""

import logging
import pickle
import string
import traceback
from typing import Dict, Tuple

from google.cloud import storage
import numpy as np
from pandas_gbq import gbq
import tensorflow as tf

from covid_epidemiology.src import constants
from covid_epidemiology.src.models import tf_model
from covid_epidemiology.src.models.definitions import base_model_definition
from covid_epidemiology.src.models.encoders import variable_encoders
from covid_epidemiology.src.models.shared import context_timer
from covid_epidemiology.src.models.shared import feature_utils
from covid_epidemiology.src.models.shared import output_utils
from covid_epidemiology.src.models.shared import typedefs


class TfSeir(tf_model.TfSeir):
  """Tensorflow SEIR Model for modeling compartments (states)."""

  def __init__(self,
               model_type,
               location_granularity,
               model_definition,
               covariate_delay=0,
               random_seed=0,
               num_xgboost_threads=1,
               tensorboard_config=None,
               **hparams_overrides):

    self.random_seed = random_seed
    self.model_type = model_type
    self.model_definition = model_definition

    self.model_spec = self.model_definition.get_model_spec(
        model_type, covariate_delay)
    self.model_constructor = self.model_definition.get_model_constructor(
        self.model_spec, random_seed)

    self.covariate_feature_specs = self.model_spec.covariate_feature_specs
    self.hparams = dict(self.model_spec.hparams)  # Load default hyperparameters
    for hparam_override in hparams_overrides:
      if hparam_override in self.hparams:
        self.hparams[hparam_override] = hparams_overrides[hparam_override]
      else:
        raise ValueError(
            f"Tried to override hparam: {hparam_override}, however this is not "
            "a valid hparam")

    for hparam_name, hparam_value in self.hparams.items():
      self.__dict__[hparam_name] = hparam_value

    self._metrics = {}
    self.location_granularity = location_granularity
    self.num_xgboost_threads = num_xgboost_threads

    # Default to no Tensorboard if not specified
    self.tensorboard_config = (
        tensorboard_config or output_utils.TensorboardConfig("OFF"))
    if self.tensorboard_config.enabled:
      self.tensorboard_writer = tf.summary.create_file_writer(
          self.tensorboard_config.log_location)
    else:
      self.tensorboard_writer = None

  def export_trained_model(self,
                           output_bucket,
                           output_filename,
                           chosen_location_list,
                           num_known_time_steps,
                           num_forecast_steps,
                           num_train_forecast_steps,
                           optimal_state,
                           optimal_variables,
                           optimal_encoders,
                           quantile_regression,
                           quantile_kernel_optimal,
                           quantile_biases_optimal,
                           static_scalers,
                           ts_scalers,
                           ts_state_scalers=None):
    """Exports the trained model to a file specified by output_filename.

    Args:
      output_bucket: Bucket of the output.
      output_filename: The name of the file in which to package values.
      chosen_location_list: As fit_forecast below.
      num_known_time_steps: Number of known timesteps.
      num_forecast_steps: Number of forecasting steps for inference.
      num_train_forecast_steps: Number of forecasting steps for training.
      optimal_state: Optimized state values.
      optimal_variables: Optimized vaariable values.
      optimal_encoders: Optimized encoder weights.
      quantile_regression: Bool representing if quantile variable were trained.
      quantile_kernel_optimal: Optimized quantile regression kernels.
      quantile_biases_optimal: Optimized quantile regression biases.
      static_scalers: Scalers for static variables.
      ts_scalers: Scalers for time-series variables.
      ts_state_scalers: Scalers for time-series variables at state level.

    Returns:
      GCS uploading.
    """
    output_dict = dict()
    output_dict["num_known_time_steps"] = num_known_time_steps
    output_dict["num_forecast_steps"] = num_forecast_steps
    output_dict["num_train_forecast_steps"] = num_train_forecast_steps
    output_dict["chosen_location_list"] = chosen_location_list
    output_dict["state_optimal"] = optimal_state.numpy()
    output_dict["variables_optimal"] = optimal_variables.numpy()
    output_dict["seir_encoders"] = []
    if quantile_regression:
      output_dict["quantile_kernel_optimal"] = quantile_kernel_optimal.numpy()
      output_dict["quantile_biases_optimal"] = quantile_biases_optimal.numpy()
    output_dict["quantile_flag"] = quantile_regression
    output_dict["location_dependent"] = self.location_dependent_init
    output_dict["static_scalers"] = static_scalers
    output_dict["ts_scalers"] = ts_scalers
    output_dict["ts_state_scalers"] = ts_state_scalers
    output_dict["model_params"] = self.hparams
    # Save model spec for downstream analysis of model outputs.
    output_dict["model_spec"] = self.model_spec

    for encoder in optimal_encoders:
      encoder_var_dict = {}
      for var in encoder.trainable_variables:
        encoder_var_dict[var.name] = var.numpy()
      output_dict["seir_encoders"].append(encoder_var_dict)

    return output_utils.upload_string_to_gcs(
        pickle.dumps(output_dict), output_bucket, output_filename)

  def _download_trained_model(self, gcs_bucket, input_filename):
    """Downloads trained model from GCS."""

    what_if_client = storage.Client(project=constants.PROJECT_ID_MODEL_TRAINING)
    model_bucket = what_if_client.get_bucket(gcs_bucket)
    model_blob = model_bucket.get_blob(input_filename)
    model_string = model_blob.download_as_string()
    input_dict = pickle.loads(model_string)
    return input_dict

  def import_trained_model_params(self, gcs_bucket, input_filename):
    """Imports model params from trained model."""

    input_dict = self._download_trained_model(gcs_bucket, input_filename)
    self.hparams = input_dict["model_params"]
    print("Imported model_params:")
    print(input_dict["model_params"])
    for hparam_name, hparam_value in self.hparams.items():
      # Copies model params to the this instance
      self.__dict__[hparam_name] = hparam_value
    return

  def import_trained_model(
      self,
      gcs_bucket,
      input_filename,
      ts_features=None,
      ts_forecasted_features=None,
      ts_name=None,
      ts_overrides=None,
      ts_categorical_features=None,
      static_features=None,
      static_overrides=None,
      num_forecast_steps=None,
  ):
    """Imports a model previously trained and exported with export_trained_model."""
    input_dict = self._download_trained_model(gcs_bucket, input_filename)

    num_known_time_steps = input_dict["num_known_time_steps"]
    num_forecast_steps = num_forecast_steps or input_dict["num_forecast_steps"]
    chosen_location_list = input_dict["chosen_location_list"]
    state_optimal = tf.convert_to_tensor(input_dict["state_optimal"])
    variables_optimal = tf.convert_to_tensor(input_dict["variables_optimal"])
    static_scalers = input_dict["static_scalers"]
    ts_scalers = input_dict["ts_scalers"]
    # Here we do not reload any previously saved model spec.

    if "quantile_kernel_optimal" in input_dict:
      quantile_kernel_optimal = tf.convert_to_tensor(
          input_dict["quantile_kernel_optimal"])
    else:
      quantile_kernel_optimal = None
    if "quantile_biases_optimal" in input_dict:
      quantile_biases_optimal = tf.convert_to_tensor(
          input_dict["quantile_biases_optimal"])
    else:
      quantile_biases_optimal = None

    # Construct new SEIR encoders to load trained SEIR encoders into.
    # For now, 'ts_state_scalers' is not used or passed into the encoders.

    loaded_encoders = self.model_definition.initialize_encoders(
        model_spec=self.model_spec,
        chosen_locations=chosen_location_list,
        num_observed_timesteps=num_known_time_steps,
        forecast_window_size=num_forecast_steps,
        output_window_size=num_forecast_steps,
        static_features=static_features,
        static_overrides=static_overrides,
        covariates=ts_features,
        forecasted_covariates=ts_forecasted_features,
        covariate_overrides=ts_overrides,
        ts_categorical_features=ts_categorical_features,
        static_scalers=static_scalers,
        ts_scalers=ts_scalers,
        trainable=False,
    )

    for encoder_index in range(len(loaded_encoders)):
      encoder = loaded_encoders[encoder_index]
      encoder_trainable_vars = encoder.trainable_variables
      loaded_data = input_dict["seir_encoders"][encoder_index]
      for var in encoder_trainable_vars:
        var.assign(tf.convert_to_tensor(loaded_data[var.name]))

    seir_encoders_optimal = loaded_encoders

    return (state_optimal, variables_optimal, seir_encoders_optimal,
            quantile_kernel_optimal, quantile_biases_optimal)

  def log_iteration(self, iteration):
    return (self.tensorboard_config.enabled and
            iteration % self.tensorboard_config.log_iterations == 0)

  def start_profiler_iteration(self, iteration):
    if (self.tensorboard_config.use_profiler and
        self.tensorboard_config.enabled and
        iteration == self.tensorboard_config.profiler_start):
      tf.profiler.experimental.start(self.tensorboard_config.log_location)

  def stop_profiler_iteration(self, iteration):
    if (self.tensorboard_config.use_profiler and
        self.tensorboard_config.enabled and
        iteration == self.tensorboard_config.profiler_end):
      tf.profiler.experimental.stop()

  def fit_forecast(self,
                   num_known_time_steps,
                   num_forecast_steps,
                   num_train_forecast_steps,
                   chosen_location_list,
                   static_features,
                   ts_features,
                   training_data_generator,
                   ts_state_features=None,
                   static_overrides=None,
                   ts_overrides=None,
                   ts_categorical_features=None,
                   num_iterations=5000,
                   display_iterations=1000,
                   optimization="RMSprop",
                   quantile_regression=False,
                   quantiles=(),
                   perform_training=True,
                   saved_model_bucket=None,
                   saved_model_name=None,
                   static_scalers=None,
                   ts_scalers=None,
                   ts_state_scalers=None,
                   xgboost_train_only=False):
    """The function for fitting and forecasting.

    Modeled compartments include Susceptible, Exposed, Infected (reported),
    Infected (undocumented), Recovered (reported), Recovered (undocumented),
    Hospitalized, ICU, Ventilator and Death for the chosen locations for
    num_known_time_steps+num_forecast_steps time steps.

    Args:
      num_known_time_steps: Number of known timesteps.
      num_forecast_steps: Number of forecasting steps for inference.
      num_train_forecast_steps: Number of forecasting steps for training.
      chosen_location_list: List of locations to show.
      static_features: Dictionary mapping feature keys to a dictionary of
        location strings to feature value mappings.
      ts_features: Dictionary mapping feature key to a dictionary of location
        strings to a list of that features values over time.
      training_data_generator: Whether Synthetic data generator is training.
        This is used to train the model in the simulator which generates
        synthetic data, and the effect is that the loss is computed from t=0 and
        there is no syncing to the ground truth.
      ts_state_features: Dictionary mapping feature key to a dictionary of state
        location strings to a list of that features values over time.
      static_overrides: Dictionary mapping feature key to a dictionary of
        location strings to a list of that feature's overrides.
      ts_overrides: Dictionary mapping feature key to a dictionary of location
        strings to a list of that feature's overrides over time.
      ts_categorical_features: List of categorical features to apply overrides
        correctly, only needed if ts_overrides is not None.
      num_iterations: Number of iterations to fit.
      display_iterations: Display period.
      optimization: Which optimizer to use.
      quantile_regression: Whether to use quantile regression and forecast
        quantile predictions.
      quantiles: Quantile values used for quantile predictions.
      perform_training: if False, saved_model_name must be not None. In this
        case, only inference will be performed on a previously trained model
        which will be restored from $saved_model_name.
      saved_model_bucket: GCS bucket to save model in.
      saved_model_name: if not None and perform_training is True, the trained
        model will be saved to this filename. If perform_training is False, the
        trained model will be restored from this filename.
      static_scalers: Dictionary of fitted scalers for each feature.
      ts_scalers: Dictionary of fitted scalers for each feature.
      ts_state_scalers: Scalers for time-series variables at state level.
      xgboost_train_only: Return without further training after XGBoost train.

    Returns:
      Predictions from the fitted model.
    """

    if not perform_training:
      # Loads saved model params at the beginning of fit_forecast
      self.import_trained_model_params(saved_model_bucket, saved_model_name)

    def propagate(seir_state,
                  seir_variables,
                  encoders,
                  num_steps,
                  num_train_steps,
                  num_future_steps,
                  infection_active_mask,
                  location_dependent_init,
                  num_locations,
                  training_data_generator,
                  gt_list = None,
                  scaler = None,
                  is_training = tf.constant(False, dtype=tf.bool)):
      """Tensorflow function to model state-space dynamics propagation.

      Args:
        seir_state: A tensor of shape [num_states, num_loc] representing the
          seir states.
        seir_variables: A tensor of shape [num_variables, num_loc] representing
          the seir equation terms.
        encoders: A list of encoders of size [num_variables] used to encode the
          seir variables.
        num_steps: The number of steps to propagate the SEIR dynamics.
        num_train_steps: The number of training steps.  Int cast to tensor by
          tf.function to avoid recompilation.
        num_future_steps: The number of future propagate steps.
        infection_active_mask: A Tensor mask indicating from which point own
          each location is active.
        location_dependent_init: Whether the initialization is location
          dependent for SEIR variables.
        num_locations: Number of locations.
        training_data_generator: Whether Synthetic data generator is training.
        gt_list: ground truth timeseries
        scaler: normalization (scaling) dictionary.
        is_training: Parameter indicating if this is running training as opposed
          to validation or inference. Boolean cast to tensor by tf.function to
          avoid recompilation.

      Returns:
        2-Tuple of:
          - states over time at the end of prediction,
          - variables over time at the end of prediction
      """
      logging.info("tracing propagate")
      all_states = [seir_state]

      # If the same variables are used to initialize all locations, we tile
      # them to match the dimensionality of the subsequent encoders.
      if not location_dependent_init:
        seir_variables = tf.tile(seir_variables, [1, num_locations])

      all_variables = [seir_variables]

      seir_timeseries_variables = tuple(
          [seir_variable] for seir_variable in tf.unstack(seir_variables))

      global_biases = seir_variables

      # If the model is in training stage for the synthetic data generation,
      # we do not synchronize with GT
      if training_data_generator:
        sync_steps = tf.constant(0, dtype=tf.int32)
      # If the model is in training stage, we synchronize with GT until number
      # of training steps
      elif is_training:
        sync_steps = num_train_steps + 1
      # If the model is in inference stage, we synchronize with GT until number
      # of known steps
      else:
        sync_steps = tf.constant(
            (num_steps + 1) - num_future_steps, dtype=tf.int32)

      for timestep in range(num_steps):

        last_state = all_states[-1]

        # We sync the propagated values with ground truth to mitigate exposure
        # bias and error propagation.

        if timestep < sync_steps:

          last_state = self.model_definition.sync_values(
              self.hparams, last_state, gt_list, timestep, is_training)
          last_state = tf.nn.relu(last_state)

          if timestep > 0:
            last_state = self.model_definition.sync_undoc(
                self.hparams, last_state, gt_list, updated_variables, timestep,
                is_training)
            last_state = tf.nn.relu(last_state)

        updated_variables = self.model_definition.bound_variables(
            seir_timeseries_variables=seir_timeseries_variables)

        state_deltas = self.model_constructor.seir_dynamics(
            current_state=last_state, seir_variables=updated_variables)

        # We introduce a binary mask over time, infection_active_mask, to
        # distinguish the timesteps when the disease has started.
        if (infection_active_mask is not None and
            len(infection_active_mask) > timestep):
          state_deltas *= infection_active_mask[timestep]
        propagated_state = last_state + state_deltas

        # Convert NaN states to 0
        propagated_state = tf.where(
            tf.math.is_nan(propagated_state), tf.zeros_like(propagated_state),
            propagated_state)
        # Convert state with negative values to 0
        propagated_state = tf.nn.relu(propagated_state)

        all_states.append(propagated_state)
        all_variables.append(updated_variables)

        # Extract death and confirmed predictions
        prediction = self.model_constructor.extract_prediction(all_states)

        self.model_definition.encode_variables(
            encoders=encoders,
            seir_timeseries_variables=seir_timeseries_variables,
            global_biases=global_biases,
            timestep=(timestep + 1),
            prediction=prediction,
            scaler=scaler,
            is_training=is_training)

      logging.info("finished tracing propagate")
      return tf.stack(all_states), tf.stack(all_variables)

    # Arrange given values in tensor form.
    num_locations = len(chosen_location_list)
    num_train_steps = num_known_time_steps - num_train_forecast_steps
    if training_data_generator:
      losses_start_index = 0
    else:
      losses_start_index = num_train_steps

    ## For training data generator
    if training_data_generator:
      # (1) Increase the learning rate
      self.initial_learning_rate = self.initial_learning_rate * 10
      # (2) Increase fine tuning steps
      self.fine_tuning_steps = self.fine_tuning_steps * 10
      # (3) Exclude early stopping
      self.restart_threshold = num_iterations
      # (4) More uniformly distribute the coefs
      if self.location_granularity == "STATE":
        self.hparams["train_loss_coef"] = [
            0, 0.1, 0.1, 1.0, 0.1, 0.01, 0.01, 0.01
        ]
        self.hparams["valid_loss_coef"] = [
            0, 0.1, 0.1, 1.0, 0.1, 0.01, 0.01, 0.01
        ]

    with context_timer.Timer("Initialize GT Timeseries"):
      ground_truth_timeseries, infection_active_mask = (
          self.model_definition.initialize_ground_truth_timeseries(
              static_features=static_features,
              ts_features=ts_features,
              chosen_locations=chosen_location_list,
              num_observed_timesteps=num_known_time_steps,
              infected_threshold=self.infected_threshold))
    with context_timer.Timer("Smooth GT Timeseries"):
      ground_truth_timeseries = (
          self.model_constructor.smooth_ground_truth_timeseries(
              num_train_steps, ground_truth_timeseries, self.smooth_coef))

    if self.location_granularity == "COUNTY":
      # Preprocess ground truth state
      ground_truth_state = (
          self.model_constructor.preprocess_state_ground_truth_timeseries(
              ts_state_features, num_known_time_steps, num_train_steps,
              self.smooth_coef))
    elif self.location_granularity == "JAPAN_PREFECTURE":
      # a different geographic resolution, so this is a placeholder.
      # Preprocess ground truth state.

      ground_truth_state = (
          self.model_constructor.preprocess_state_ground_truth_timeseries(
              ts_state_features, num_known_time_steps, num_train_steps,
              self.smooth_coef))
    else:
      ground_truth_state = None

    # Get normalization scaler
    scaler = self.model_definition.gt_scaler(ground_truth_timeseries,
                                             num_known_time_steps)

    # Covariate forecasting (Before training the model)

    with context_timer.Timer("Forecast train covariates"):
      ts_covariates_train = feature_utils.cov_pred(
          ts_features,
          num_train_steps,
          num_train_forecast_steps,
          max(num_forecast_steps, num_train_forecast_steps),
          self.covariate_feature_specs,
          chosen_location_list,
          num_threads=self.num_xgboost_threads,
          covariate_training_mixing_coef=self
          .hparams["covariate_training_mixing_coef"])

    # Set file name for saving xgboost forecast output
    num_covariate_forecast_steps = max(num_forecast_steps,
                                       num_train_forecast_steps)
    if not saved_model_name:
      xgboost_file_name = None
    else:
      name_no_run_number = saved_model_name.rstrip(string.digits)
      xgboost_file_name = (
          f"xgboost.{name_no_run_number}{num_covariate_forecast_steps}")

    # Try to load forecasted covariates from BigQuery:
    try:
      if not xgboost_file_name:
        raise gbq.NotFoundException("No XGBoost file to load")

      with context_timer.Timer("Load inference covariates"):
        ts_covariates_inference = feature_utils.read_covariates_from_bigquery(
            xgboost_file_name)
    except Exception as e:  # pylint: disable=broad-except
      if not isinstance(e, gbq.NotFoundException):
        logging.info("Loading inference covariates failed with:\n%s",
                     traceback.format_exc())
      with context_timer.Timer("Forecast inference covariates"):
        ts_covariates_inference = feature_utils.cov_pred(
            ts_features,
            num_known_time_steps,
            num_forecast_steps,
            num_covariate_forecast_steps,
            self.covariate_feature_specs,
            chosen_location_list,
            num_threads=self.num_xgboost_threads,
            covariate_training_mixing_coef=0.0)
      if xgboost_file_name is not None:
        with context_timer.Timer("Write forecasted covariates"):
          # This may return different covariates if an existing covariate table
          # for this run already exists.
          ts_covariates_inference = feature_utils.write_covariates_to_bigquery(
              ts_covariates_inference, chosen_location_list, xgboost_file_name)

    if xgboost_train_only:
      self._metrics["xgboost_training_complete"] = True
      return []

    if not perform_training:
      # Load model from file.
      (loaded_state, loaded_variables, loaded_encoders, loaded_quantile_kernel,
       loaded_quantile_biases) = self.import_trained_model(
           gcs_bucket=saved_model_bucket,
           input_filename=saved_model_name,
           ts_features=ts_features,
           ts_forecasted_features=ts_covariates_inference,
           ts_overrides=ts_overrides,
           static_features=static_features,
           static_overrides=static_overrides,
           ts_categorical_features=ts_categorical_features,
           num_forecast_steps=num_forecast_steps,
       )

      propagated_states, propagated_variables = propagate(
          loaded_state,
          loaded_variables,
          encoders=loaded_encoders,
          num_steps=num_known_time_steps + num_forecast_steps - 1,
          num_train_steps=tf.constant(num_known_time_steps - 1, dtype=tf.int32),
          num_future_steps=num_forecast_steps,
          infection_active_mask=infection_active_mask,
          location_dependent_init=self.location_dependent_init,
          num_locations=num_locations,
          training_data_generator=training_data_generator,
          gt_list=ground_truth_timeseries,
          scaler=scaler,
          is_training=tf.constant(False, dtype=tf.bool))

      # Convert NaN states to 0
      propagated_states = tf.where(
          tf.math.is_nan(propagated_states), tf.zeros_like(propagated_states),
          propagated_states)
      # Convert state with negative values to 0
      propagated_states = tf.nn.relu(propagated_states)

      if quantile_regression:
        propagated_states = self.model_constructor.apply_quantile_transform(
            self.hparams,
            propagated_states,
            loaded_quantile_kernel,
            loaded_quantile_biases,
            ground_truth_timeseries,
            num_known_time_steps,
            num_forecast_steps,
            num_quantiles=len(quantiles),
            is_training=False,
            initial_quantile_step=0)
      else:
        propagated_states = tf.expand_dims(propagated_states, axis=-1)

      return self.model_constructor.generate_compartment_predictions(
          chosen_location_list=chosen_location_list,
          propagated_states=propagated_states,
          propagated_variables=propagated_variables,
          num_forecast_steps=num_forecast_steps,
          ground_truth_timeseries=ground_truth_timeseries,
          quantile_regression=quantile_regression)

    train_coefs, valid_coefs = self.model_constructor.compute_coef(
        ground_truth_timeseries,
        ground_truth_state if self.location_granularity == "COUNTY" else None,
        num_train_steps,
        num_known_time_steps,
        power=2.0)

    # Initial values of states.

    (init_state, init_variables,
     seir_encoders) = self.model_definition.initialize_components(
         model_spec=self.model_spec,
         ground_truth_timeseries=ground_truth_timeseries,
         infected_threshold=self.infected_threshold,
         hparams=self.hparams,
         num_locations=num_locations,
         location_dependent_init=self.location_dependent_init,
         chosen_locations=chosen_location_list,
         num_observed_timesteps=num_train_steps,
         forecast_window_size=num_train_forecast_steps,
         output_window_size=num_train_forecast_steps,
         static_features=static_features,
         static_scalers=static_scalers,
         static_overrides=None,
         covariates=ts_features,
         ts_scalers=ts_scalers,
         forecasted_covariates=ts_covariates_train,
         covariate_overrides=None,
         ts_categorical_features=None)

    # We define a second set of states, variables, and encoders to keep track of
    # the optimal set of hyperparameters that minimize the validation loss.
    (init_state_optimal, init_variables_optimal,
     seir_encoders_optimal) = self.model_definition.initialize_components(
         model_spec=self.model_spec,
         ground_truth_timeseries=ground_truth_timeseries,
         infected_threshold=self.infected_threshold,
         hparams=self.hparams,
         num_locations=num_locations,
         location_dependent_init=self.location_dependent_init,
         chosen_locations=chosen_location_list,
         num_observed_timesteps=num_known_time_steps,
         forecast_window_size=num_forecast_steps,
         output_window_size=num_forecast_steps,
         static_features=static_features,
         static_scalers=static_scalers,
         static_overrides=None,
         covariates=ts_features,
         ts_scalers=ts_scalers,
         forecasted_covariates=ts_covariates_inference,
         covariate_overrides=None,
         ts_categorical_features=None,
         trainable=False,
     )

    logging.info("Variables are initialized")

    if quantile_regression:

      # Define quantile variables
      (quantile_kernel,
       quantile_biases) = self.model_definition.initialize_quantile_variables(
           self.hparams, num_quantiles=len(quantiles))

      # We define a third set of states, variables, and encoders for quantile
      # regression by copying the optimal encoder values from the point-wise
      # forecast training for use during quantile optimization.
      (init_state_quantile, init_variables_quantile,
       seir_encoders_quantile) = self.model_definition.initialize_components(
           model_spec=self.model_spec,
           ground_truth_timeseries=ground_truth_timeseries,
           infected_threshold=self.infected_threshold,
           hparams=self.hparams,
           num_locations=num_locations,
           location_dependent_init=self.location_dependent_init,
           chosen_locations=chosen_location_list,
           num_observed_timesteps=num_train_steps,
           forecast_window_size=num_train_forecast_steps,
           output_window_size=num_train_forecast_steps,
           static_features=static_features,
           static_scalers=static_scalers,
           static_overrides=None,
           covariates=ts_features,
           ts_scalers=ts_scalers,
           forecasted_covariates=ts_covariates_train,
           covariate_overrides=None,
           ts_categorical_features=None,
           trainable=False,
       )

      # Compute train/valid coefs
      quantile_train_coefs, quantile_valid_coefs = self.model_constructor.compute_coef(
          ground_truth_timeseries,
          ground_truth_state if self.location_granularity == "COUNTY" else None,
          num_train_steps,
          num_known_time_steps,
          power=1.0)

    # Initialize states for the optimizer at the optimality
    optimizer_state_optimal = None

    # Training loop comprised of propagates state updates and
    # corresponding loss computation.
    best_train_loss = 1e128
    best_valid_loss = 1e128
    best_valid_loss_q = 1e128
    worse_count = 0

    # Train on train, optimize hparams on valid, then retrain on train+valid.

    # pylint: disable=g-complex-comprehension
    encoder_variables = [
        trainable_variable for encoder in seir_encoders
        for trainable_variable in encoder.trainable_variables
    ]
    variable_list = [init_state, init_variables] + encoder_variables
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        self.initial_learning_rate,
        self.decay_steps,
        self.decay_rate,
        staircase=False,
    )
    learning_rate_q = tf.keras.optimizers.schedules.ExponentialDecay(
        self.initial_learning_rate,
        self.decay_steps,
        self.decay_rate,
        staircase=False,
    )
    if optimization == "Adagrad":
      optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
      optimizer_q = tf.keras.optimizers.Adagrad(learning_rate=learning_rate_q)
    elif optimization == "RMSprop":
      optimizer = tf.keras.optimizers.RMSprop(
          learning_rate=learning_rate, rho=0.99, momentum=self.momentum)
      optimizer_q = tf.keras.optimizers.RMSprop(
          learning_rate=learning_rate_q, rho=0.99, momentum=self.momentum)

    # Create a tf.function-wrapped propagate function for using in train_step
    # and fine_tune_step. Tracing propagate is expensive, and we only want to
    # trace it once for both training and fine tuning, but not for prediction or
    # what-if scenarios, where the cost of tracing/compilation cannot be
    # amortized over multiple calls.

    propagate_tf_fn = tf.function(propagate)

    def create_train_step_fn():
      """Creates a training step function."""

      @tf.function
      def train_step(num_train_steps_effective):
        logging.info("tracing train_step_fn")
        if training_data_generator:
          losses_start_index_effective = tf.constant(0, dtype=tf.int32)
        else:
          losses_start_index_effective = num_train_steps_effective

        with tf.GradientTape() as tape:

          # In training stage, the train loss is computed in the range of
          # [randomly selected training steps, randomly selected training steps
          # + number of training forecasting steps]
          propagated_states_train, propagated_variables_train = propagate_tf_fn(
              init_state,
              init_variables,
              encoders=seir_encoders,
              num_steps=num_known_time_steps - 1,
              # num_train_steps_effective is already a tensor.
              num_train_steps=num_train_steps_effective - 1,
              num_future_steps=num_train_forecast_steps,
              infection_active_mask=infection_active_mask,
              location_dependent_init=self.location_dependent_init,
              num_locations=num_locations,
              training_data_generator=training_data_generator,
              gt_list=ground_truth_timeseries,
              scaler=scaler,
              is_training=tf.constant(True, dtype=tf.bool))

          # In validation stage, the validation loss is computed in the range of
          # [number of training steps, number of training steps +
          # number of training forecasting steps]
          propagated_states_valid, _ = propagate_tf_fn(
              init_state,
              init_variables,
              encoders=seir_encoders,
              num_steps=num_known_time_steps - 1,
              num_train_steps=tf.constant(num_train_steps - 1, dtype=tf.int32),
              num_future_steps=num_train_forecast_steps,
              infection_active_mask=infection_active_mask,
              location_dependent_init=self.location_dependent_init,
              num_locations=num_locations,
              training_data_generator=training_data_generator,
              gt_list=ground_truth_timeseries,
              scaler=scaler,
              is_training=tf.constant(False, dtype=tf.bool))

          r_eff = self.model_constructor.calculate_r_eff(
              propagated_variables=propagated_variables_train)

          train_loss_overall, _ = self.model_constructor.compute_losses(
              hparams=self.hparams,
              train_coefs=train_coefs,
              valid_coefs=valid_coefs,
              propagated_states=propagated_states_train,
              ground_truth_timeseries=ground_truth_timeseries,
              r_eff=r_eff,
              train_start_index=losses_start_index_effective,
              train_end_index=num_train_steps_effective +
              num_train_forecast_steps,
              valid_start_index=losses_start_index_effective,
              valid_end_index=num_train_steps_effective +
              num_train_forecast_steps,
              num_forecast_steps=num_train_forecast_steps)

          _, valid_loss_overall = self.model_constructor.compute_losses(
              hparams=self.hparams,
              train_coefs=train_coefs,
              valid_coefs=valid_coefs,
              propagated_states=propagated_states_valid,
              ground_truth_timeseries=ground_truth_timeseries,
              r_eff=None,
              train_start_index=losses_start_index,
              train_end_index=num_train_steps + num_train_forecast_steps,
              valid_start_index=losses_start_index,
              valid_end_index=num_train_steps + num_train_forecast_steps,
              num_forecast_steps=num_train_forecast_steps)

          direction_loss = self.model_constructor.direction_losses(
              hparams=self.hparams, seir_encoders=seir_encoders)
          if direction_loss is not None:
            train_loss_overall += direction_loss
            valid_loss_overall += direction_loss

          # Local bias penalty
          bias_penalty = 0
          for encoder in seir_encoders:
            encoder_trainable_vars = encoder.trainable_variables
            for var in encoder_trainable_vars:
              if "LocationBias" in var.name:
                bias_penalty += tf.reduce_mean(tf.square(var))
          bias_penalty *= self.hparams["bias_penalty_coef"]
          train_loss_overall += bias_penalty

          lasso_loss = 0
          for encoder in seir_encoders:
            lasso_loss += encoder.lasso_loss
          lasso_loss *= self.hparams["lasso_penalty_coef"]
          train_loss_overall += lasso_loss

          # Add aggregation loss for county model
          if self.location_granularity == "COUNTY":
            train_aggregate_loss, _ = self.model_constructor.aggregation_penalty(
                hparams=self.hparams,
                train_coefs=train_coefs,
                valid_coefs=valid_coefs,
                propagated_states=propagated_states_train,
                chosen_location_list=chosen_location_list,
                ground_truth_state=ground_truth_state,
                train_start_index=losses_start_index_effective,
                train_end_index=(num_train_steps_effective +
                                 num_train_forecast_steps),
                valid_start_index=losses_start_index_effective,
                valid_end_index=(num_train_steps_effective +
                                 num_train_forecast_steps),
                num_forecast_steps=num_train_forecast_steps,
            )

            _, valid_aggregate_loss = self.model_constructor.aggregation_penalty(
                hparams=self.hparams,
                train_coefs=train_coefs,
                valid_coefs=valid_coefs,
                propagated_states=propagated_states_valid,
                chosen_location_list=chosen_location_list,
                ground_truth_state=ground_truth_state,
                train_start_index=losses_start_index,
                train_end_index=num_train_steps + num_train_forecast_steps,
                valid_start_index=losses_start_index,
                valid_end_index=num_train_steps + num_train_forecast_steps,
                num_forecast_steps=num_train_forecast_steps,
            )

            train_loss_overall += train_aggregate_loss
            valid_loss_overall += valid_aggregate_loss
          else:
            train_aggregate_loss = None
            valid_aggregate_loss = None

        grads = tape.gradient(train_loss_overall, variable_list)
        logging.info("finished tracing train_step_fn")
        return (grads, train_loss_overall, valid_loss_overall, bias_penalty,
                train_aggregate_loss, valid_aggregate_loss, lasso_loss)

      return train_step

    if quantile_regression:

      @tf.function
      def quantile_train_step():
        logging.info("tracing quantile_train_step")
        with tf.GradientTape() as quantile_tape:
          # This seems to get retraced which may be due to the different
          # variables but is hard to isolate.
          propagated_states_point, _ = propagate_tf_fn(
              init_state_quantile,
              init_variables_quantile,
              encoders=seir_encoders_quantile,
              num_steps=num_known_time_steps - 1,
              num_train_steps=tf.constant(num_train_steps - 1, dtype=tf.int32),
              num_future_steps=num_train_forecast_steps,
              infection_active_mask=infection_active_mask,
              location_dependent_init=self.location_dependent_init,
              num_locations=num_locations,
              training_data_generator=training_data_generator,
              gt_list=ground_truth_timeseries,
              scaler=scaler,
              is_training=tf.constant(False, dtype=tf.bool))

          quantile_propagated_states = self.model_constructor.apply_quantile_transform(
              self.hparams,
              propagated_states_point,
              quantile_kernel,
              quantile_biases,
              ground_truth_timeseries,
              num_train_steps,
              num_train_forecast_steps,
              num_quantiles=len(quantiles),
              initial_quantile_step=num_train_steps)

          _, valid_loss_overall_q = self.model_constructor.compute_losses(
              hparams=self.hparams,
              train_coefs=quantile_train_coefs,
              valid_coefs=quantile_valid_coefs,
              propagated_states=quantile_propagated_states,
              ground_truth_timeseries=ground_truth_timeseries,
              r_eff=None,
              train_start_index=losses_start_index,
              train_end_index=num_train_steps + num_train_forecast_steps,
              valid_start_index=losses_start_index,
              valid_end_index=num_train_steps + num_train_forecast_steps,
              quantiles=quantiles,
              num_forecast_steps=num_train_forecast_steps)

        grads_q = quantile_tape.gradient(valid_loss_overall_q,
                                         [quantile_kernel, quantile_biases])
        logging.info("finished tracing quantile_train_step")

        return grads_q, valid_loss_overall_q

    train_step_fn = create_train_step_fn()

    for iteration_index in range(num_iterations):
      self.start_profiler_iteration(iteration_index)

      num_train_steps_random = np.random.randint(
          int(
              tf.cast(num_train_steps, dtype=tf.float32) /
              self.hparams["train_window_range"]), num_train_steps,
          1)[0] - num_train_forecast_steps

      # For training generator, disable random sampling
      if training_data_generator:
        num_train_steps_random = num_train_steps - num_train_forecast_steps

      # We must have at least num_train_forecast_steps for incremental loss
      num_train_steps_random = max(num_train_steps_random,
                                   num_train_forecast_steps)

      (grads, train_loss_overall, valid_loss_overall, bias_penalty,
       train_aggregate_loss, valid_aggregate_loss, lasso_loss) = train_step_fn(
           tf.constant(num_train_steps_random, dtype=tf.int32))

      if self.location_granularity == "COUNTY":
        if self.log_iteration(iteration_index):
          with self.tensorboard_writer.as_default():
            tf.summary.scalar(
                "train_aggregate_loss",
                train_aggregate_loss,
                step=iteration_index)
            tf.summary.scalar(
                "valid_aggregate_loss",
                valid_aggregate_loss,
                step=iteration_index)

      if valid_loss_overall.numpy() < best_valid_loss:
        best_train_loss = train_loss_overall.numpy()
        best_valid_loss = valid_loss_overall.numpy()
        init_state_optimal.assign(init_state)
        init_variables_optimal.assign(init_variables)

        for encoder_index in range(len(seir_encoders_optimal)):
          encoder_optimal = seir_encoders_optimal[encoder_index]
          encoder = seir_encoders[encoder_index]
          encoder_optimal_trainable_vars = encoder_optimal.trainable_variables
          encoder_trainable_vars = encoder.trainable_variables
          for var in encoder_trainable_vars:
            for var_optimal in encoder_optimal_trainable_vars:
              if var.name == var_optimal.name:
                var_optimal.assign(var)

        # Save the states of the optimizer
        if iteration_index > 0:
          opt_state = optimizer.get_weights()
          optimizer_state_optimal = [opt_state[0]]
          for i in range(1, len(opt_state)):
            optimizer_state_optimal.append(np.ndarray.copy(opt_state[i]))

        worse_count = 0

      else:
        # If the optimization is not improving after restart_threshold
        # iterations or training/valid loss is nan at the beginning of the
        # training, we break from the loop to early terminate the training.
        worse_count += 1
        if ((worse_count > self.restart_threshold) or
            (np.isnan(valid_loss_overall.numpy()) and
             best_valid_loss == 1e128)):
          break

      optimizer.apply_gradients(zip(grads, variable_list))

      if self.log_iteration(iteration_index):
        with self.tensorboard_writer.as_default():
          tf.summary.scalar(
              "train_loss", train_loss_overall, step=iteration_index)
          tf.summary.scalar(
              "valid_loss", valid_loss_overall, step=iteration_index)
          tf.summary.scalar("bias_penalty", bias_penalty, step=iteration_index)
          tf.summary.scalar("lasso_loss", lasso_loss, step=iteration_index)
          for idx, v in enumerate(variable_list):
            tf.summary.histogram(
                f"point_variable_{idx}", v, step=iteration_index)
          for idx, g in enumerate(grads):
            tf.summary.histogram(f"point_grad_{idx}", g, step=iteration_index)

      # Quantile regression
      if (quantile_regression and
          iteration_index % int(1 / self.quantile_training_iteration_ratio)
          == 0):

        # Assign the optimal parameters
        init_state_quantile.assign(init_state_optimal)
        init_variables_quantile.assign(init_variables_optimal)

        for encoder_index in range(len(seir_encoders_quantile)):
          encoder_quantile = seir_encoders_quantile[encoder_index]
          encoder_optimal = seir_encoders_optimal[encoder_index]
          encoder_quantile_trainable_vars = encoder_quantile.trainable_variables
          encoder_optimal_trainable_vars = encoder_optimal.trainable_variables
          for var in encoder_optimal_trainable_vars:
            for var_quantile in encoder_quantile_trainable_vars:
              if var.name == var_quantile.name:
                var_quantile.assign(var)

        grads_q, valid_loss_overall_q = quantile_train_step()

        if valid_loss_overall_q.numpy() < best_valid_loss_q:
          quantile_kernel_optimal = tf.convert_to_tensor(quantile_kernel)
          quantile_biases_optimal = tf.convert_to_tensor(quantile_biases)
          best_valid_loss_q = valid_loss_overall_q.numpy()

        optimizer_q.apply_gradients(
            zip(grads_q, [quantile_kernel, quantile_biases]))

        if self.log_iteration(iteration_index):
          with self.tensorboard_writer.as_default():
            tf.summary.scalar(
                "quantile_valid_loss",
                valid_loss_overall_q,
                step=iteration_index)
            tf.summary.histogram(
                "quantile_kernel", quantile_kernel, step=iteration_index)
            tf.summary.histogram(
                "quantile_biases", quantile_biases, step=iteration_index)
            for idx, g in enumerate(grads_q):
              tf.summary.histogram(
                  f"quantile_grad_{idx}", g, step=iteration_index)

      self.stop_profiler_iteration(iteration_index)

      if iteration_index % display_iterations == 0:
        print("-----------------------------------------------------")

        print("Optimal initial state:")
        print(init_state_optimal)
        print("Optimal initial variables:")
        print(init_variables_optimal)
        print("Optimal encoder variables:")
        for encoder_index in range(len(seir_encoders_optimal)):
          encoder_optimal = seir_encoders_optimal[encoder_index]
          encoder_optimal_trainable_vars = encoder_optimal.trainable_variables
          print(encoder_optimal_trainable_vars)

        print("Step: " + str(iteration_index))
        print("Train loss: " + str(train_loss_overall.numpy()))
        print("Valid loss: " + str(valid_loss_overall.numpy()))

        print("Best train loss: " + str(best_train_loss))
        print("Best valid loss: " + str(best_valid_loss))

        if quantile_regression:
          print("Valid loss quantile: " + str(valid_loss_overall_q.numpy()))
          print("Best valid loss quantile: " + str(best_valid_loss_q))

      # If the current training loss is NaN and we have the saved best model
      if (np.isnan(train_loss_overall.numpy()) and best_valid_loss < 1e128):
        break

    self._metrics["train_loss"] = best_train_loss.item()
    self._metrics["validation_loss"] = best_valid_loss.item()
    self._metrics["step"] = iteration_index + 1

    # Fine-tune on training and validation.

    # Assign the optimal parameters
    init_state.assign(init_state_optimal)
    init_variables.assign(init_variables_optimal)
    for encoder_index in range(len(seir_encoders_optimal)):
      encoder_optimal = seir_encoders_optimal[encoder_index]
      encoder = seir_encoders[encoder_index]
      encoder_optimal_trainable_vars = encoder_optimal.trainable_variables
      encoder_trainable_vars = encoder.trainable_variables
      for var in encoder_trainable_vars:
        for var_optimal in encoder_optimal_trainable_vars:
          if var.name == var_optimal.name:
            var.assign(0 * var + var_optimal)

    # Reset the optimizer
    if optimizer_state_optimal is not None:
      optimizer.set_weights(optimizer_state_optimal)
      optimizer.lr.initial_learning_rate = (
          self.fine_tuning_decay * self.initial_learning_rate)

    for iteration_index in range(self.fine_tuning_steps):

      (grads_ft, train_loss_overall, valid_loss_overall, bias_penalty,
       train_aggregate_loss, valid_aggregate_loss, lasso_loss) = train_step_fn(
           tf.constant(num_train_steps, dtype=tf.int32))

      if self.location_granularity == "COUNTY":
        if self.log_iteration(iteration_index):
          with self.tensorboard_writer.as_default():
            tf.summary.scalar(
                "fine_tuning_train_aggregate_loss",
                train_aggregate_loss,
                step=iteration_index)
            tf.summary.scalar(
                "fine_tuning_valid_aggregate_loss",
                valid_aggregate_loss,
                step=iteration_index)

      if valid_loss_overall.numpy() < best_valid_loss:
        best_train_loss = train_loss_overall.numpy()
        best_valid_loss = valid_loss_overall.numpy()
        init_state_optimal.assign(init_state)
        init_variables_optimal.assign(init_variables)
        for encoder_index in range(len(seir_encoders_optimal)):
          encoder_optimal = seir_encoders_optimal[encoder_index]
          encoder = seir_encoders[encoder_index]
          encoder_optimal_trainable_vars = encoder_optimal.trainable_variables
          encoder_trainable_vars = encoder.trainable_variables
          for var in encoder_trainable_vars:
            for var_optimal in encoder_optimal_trainable_vars:
              if var.name == var_optimal.name:
                var_optimal.assign(var)

          worse_count = 0

      optimizer.apply_gradients(zip(grads_ft, variable_list))

      if self.log_iteration(iteration_index):
        # Log the fine-tuning data
        with self.tensorboard_writer.as_default():
          tf.summary.scalar(
              "fine_tuning_train_loss",
              train_loss_overall,
              step=iteration_index)
          tf.summary.scalar(
              "fine_tuning_valid_loss",
              valid_loss_overall,
              step=iteration_index)
          tf.summary.scalar(
              "fine_tuning_lasso_loss", lasso_loss, step=iteration_index)
          tf.summary.scalar(
              "fine_tuning_bias_penalty", bias_penalty, step=iteration_index)
          for idx, v in enumerate(variable_list):
            tf.summary.histogram(
                f"fine_tuning_point_variable_{idx}", v, step=iteration_index)
          for idx, g in enumerate(grads_ft):
            tf.summary.histogram(
                f"fine_tuning_point_grad_{idx}", g, step=iteration_index)

      if iteration_index % display_iterations == 0:
        print("-----------------------------------------------------")

        print("Optimal initial state:")
        print(init_state_optimal)
        print("Optimal initial variables:")
        print(init_variables_optimal)
        print("Optimal encoder variables:")
        for encoder_index in range(len(seir_encoders_optimal)):
          encoder_optimal = seir_encoders_optimal[encoder_index]
          encoder_optimal_trainable_vars = encoder_optimal.trainable_variables
          print(encoder_optimal_trainable_vars)

        print("Step: " + str(iteration_index))
        print("Train loss: " + str(train_loss_overall.numpy()))
        print("Valid loss: " + str(valid_loss_overall.numpy()))

        print("Best train loss: " + str(best_train_loss))
        print("Best valid loss: " + str(best_valid_loss))

        if quantile_regression:
          print("Valid loss quantile: " + str(valid_loss_overall_q.numpy()))
          print("Best valid loss quantile: " + str(best_valid_loss_q))

    if saved_model_name:
      if not quantile_regression:
        quantile_kernel_optimal = None
        quantile_biases_optimal = None

      self.export_trained_model(
          saved_model_bucket, saved_model_name, chosen_location_list,
          num_known_time_steps, num_forecast_steps, num_train_forecast_steps,
          init_state_optimal, init_variables_optimal, seir_encoders_optimal,
          quantile_regression, quantile_kernel_optimal, quantile_biases_optimal,
          static_scalers, ts_scalers, ts_state_scalers)

    # Inference of future values.
    def predict():
      propagated_states, propagated_variables = propagate(
          init_state_optimal,
          init_variables_optimal,
          encoders=seir_encoders_optimal,
          num_steps=num_known_time_steps + num_forecast_steps - 1,
          num_train_steps=tf.constant(num_known_time_steps - 1, dtype=tf.int32),
          num_future_steps=num_forecast_steps,
          infection_active_mask=infection_active_mask,
          location_dependent_init=self.location_dependent_init,
          num_locations=num_locations,
          training_data_generator=training_data_generator,
          gt_list=ground_truth_timeseries,
          scaler=scaler,
          is_training=tf.constant(False, dtype=tf.bool))

      # Convert NaN states to 0
      propagated_states = tf.where(
          tf.math.is_nan(propagated_states), tf.zeros_like(propagated_states),
          propagated_states)
      # Convert state with negative values to 0
      propagated_states = tf.nn.relu(propagated_states)

      if quantile_regression:
        propagated_states = self.model_constructor.apply_quantile_transform(
            self.hparams,
            propagated_states,
            quantile_kernel_optimal,
            quantile_biases_optimal,
            ground_truth_timeseries,
            num_known_time_steps,
            num_forecast_steps,
            num_quantiles=len(quantiles),
            is_training=False,
            initial_quantile_step=0)
      else:
        propagated_states = tf.expand_dims(propagated_states, axis=-1)

      return propagated_states, propagated_variables

    propagated_states, propagated_variables = predict()

    return self.model_constructor.generate_compartment_predictions(
        chosen_location_list=chosen_location_list,
        propagated_states=propagated_states,
        propagated_variables=propagated_variables,
        num_forecast_steps=num_forecast_steps,
        ground_truth_timeseries=ground_truth_timeseries,
        quantile_regression=quantile_regression)

  @property
  def metrics(self):
    return self._metrics

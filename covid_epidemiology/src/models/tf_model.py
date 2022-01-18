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

"""Tensorflow model for time-varying SEIR dynamics.

Constructs a Tensorflow program to model multi-horizon forecasts with
time-varying SEIR dynamics, where fitting is based on gradient descent based
optimization.
"""

import abc
import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from covid_epidemiology.src import constants
from covid_epidemiology.src import model
from covid_epidemiology.src.models.shared import model_utils
from covid_epidemiology.src.models.shared import typedefs


class TfSeir(model.Model, abc.ABC):
  """Tensorflow Model class."""

  def fit_forecast_fixed(
      self,
      train_window_end_index,
      train_window_end_date,
      num_forecast_steps,
      num_train_forecast_steps,
      static_features,
      static_overrides,
      ts_features,
      ts_overrides,
      ts_categorical_features,
      ts_state_features,
      locations,
      num_iterations,
      display_iterations,
      optimization,
      training_data_generator,
      quantile_regression,
      static_scalers = None,
      ts_scalers = None,
      ts_state_scalers = None,
      perform_training=True,
      saved_model_bucket=None,
      saved_model_name=None,
      xgboost_train_only=False,
      reduced_output=False,
      reduced_quantile_list=True
  ):
    """Returns a model forecast."""
    location_to_window_predictions_all = {}
    location_to_window_predictions_forecast = {}
    for location in locations:
      location_to_window_predictions_all[location] = []
      location_to_window_predictions_forecast[location] = []

    # Specify quantile values in regression if quantile forecast is needed
    # Also, define the string format of metric in quantile forecast
    if quantile_regression:
      if reduced_quantile_list:
        quantiles_output = constants.QUANTILE_LIST_REDUCED
      else:
        quantiles_output = constants.QUANTILE_LIST
      quantiles = constants.QUANTILE_LIST
      metric_string_format = '{metric}_{quantile}_quantile'
    else:
      quantiles = None
      quantiles_output = None
      metric_string_format = None

    # Returns the forecasts for the chosen locations.
    compartments = self.fit_forecast(
        num_known_time_steps=train_window_end_index,
        num_forecast_steps=num_forecast_steps,
        num_train_forecast_steps=num_train_forecast_steps,
        chosen_location_list=locations,
        static_features=static_features,
        static_scalers=static_scalers,
        ts_features=ts_features,
        training_data_generator=training_data_generator,
        ts_scalers=ts_scalers,
        ts_state_features=ts_state_features,
        ts_state_scalers=ts_state_scalers,
        static_overrides=static_overrides,
        ts_overrides=ts_overrides,
        ts_categorical_features=ts_categorical_features,
        num_iterations=num_iterations,
        display_iterations=display_iterations,
        optimization=optimization,
        quantile_regression=quantile_regression,
        quantiles=quantiles,
        perform_training=perform_training,
        saved_model_bucket=saved_model_bucket,
        saved_model_name=saved_model_name,
        xgboost_train_only=xgboost_train_only)
    if reduced_output:
      eval_train_start_idx = train_window_end_index - num_forecast_steps
      time_horizon_offset = num_forecast_steps
    else:
      eval_train_start_idx = 0
      time_horizon_offset = train_window_end_index

    def update_metrics(metric_to_predictions_all,
                       metric_to_predictions_forecast, location, name,
                       predictions, ground_truth, num_forecast_steps,
                       use_quantiles):
      gt_data = ground_truth[location] if ground_truth is not None else None
      metric_to_predictions_all = model_utils.update_metric_to_predictions(
          metric=name,
          values=(predictions[location]
                  [eval_train_start_idx:train_window_end_index +
                   num_forecast_steps]),
          metric_to_predictions=metric_to_predictions_all,
          train_end_of_window=eval_train_start_idx,
          gt_data=gt_data,
          time_horizon_offset=time_horizon_offset,
          quantiles=quantiles if use_quantiles else None,
          quantiles_output=quantiles_output if use_quantiles else None,
          metric_string_format=metric_string_format)
      metric_to_predictions_forecast = model_utils.update_metric_to_predictions(
          metric=name,
          values=(predictions[location]
                  [train_window_end_index:train_window_end_index +
                   num_forecast_steps]),
          metric_to_predictions=metric_to_predictions_forecast,
          train_end_of_window=train_window_end_index,
          gt_data=gt_data,
          quantiles=quantiles if use_quantiles else None,
          quantiles_output=quantiles_output if use_quantiles else None,
          metric_string_format=metric_string_format)
      return metric_to_predictions_all, metric_to_predictions_forecast

    for location in locations:
      metric_to_predictions_all = {}
      metric_to_predictions_forecast = {}

      for compartment in compartments:
        (metric_to_predictions_all,
         metric_to_predictions_forecast) = update_metrics(
             metric_to_predictions_all=metric_to_predictions_all,
             metric_to_predictions_forecast=metric_to_predictions_forecast,
             location=location,
             name=compartment.name,
             predictions=compartment.predictions,
             ground_truth=compartment.ground_truth,
             num_forecast_steps=compartment.num_forecast_steps,
             use_quantiles=compartment.use_quantiles)

      location_to_window_predictions_all[location].append(
          model.WindowPrediction(train_window_end_date,
                                 metric_to_predictions_all))
      location_to_window_predictions_forecast[location].append(
          model.WindowPrediction(train_window_end_date,
                                 metric_to_predictions_forecast))

    model_output_all = model.ModelOutput(location_to_window_predictions_all)
    model_output_forecast = model.ModelOutput(
        location_to_window_predictions_forecast)
    return (model_output_forecast, model_output_all)

  @abc.abstractmethod
  def fit_forecast(self, num_known_time_steps, num_forecast_steps,
                   num_train_forecast_steps, chosen_location_list,
                   static_features, static_scalers, ts_features,
                   training_data_generator, ts_scalers, ts_state_features,
                   ts_state_scalers, static_overrides, ts_overrides,
                   ts_categorical_features, num_iterations, display_iterations,
                   optimization, quantile_regression, quantiles,
                   perform_training, saved_model_bucket, saved_model_name,
                   xgboost_train_only):
    """Returns the estimated compartment values for the chosen locations.

    In particular, this returns the estimated Susceptible, Exposed, Infected
    (reported), Infected (undocumented), Recovered (reported), Recovered
    (undocumented), and Death for the chosen locations.

    Args:
      num_known_time_steps: Number of known timesteps.
      num_forecast_steps: Number of forecasting steps at inference.
      num_train_forecast_steps: Number of forecasting steps at training.
      chosen_location_list: List of locations to show.
      static_features: Dictionary mapping feature keys to a dictionary of
        location strings to feature value mappings.
      static_scalers: Dictionary of fitted scalers for each feature.
      ts_features: Dictionary mapping feature key to a dictionary of location
        strings to a list of that features values over time.
      training_data_generator: Training data generator.
      ts_scalers: Dictionary of fitted scalers for each feature.
      ts_state_features: Dictionary mapping feature key to a dictionary of state
        location strings to a list of that features values over time.
      ts_state_scalers: Dictionary of fitted scalers for each feature.
      static_overrides: Dictionary mapping feature key to a dictionary of
        location strings to a list of that feature's overrides.
      ts_overrides: Dictionary mapping feature key to a dictionary of location
        strings to a list of that feature's overrides over time.
      ts_categorical_features: List of categorical features to apply overrides
        correctly, only needed if ts_overrides is not None.
      num_iterations: Number of iterations to fit.
      display_iterations: Display period.
      optimization: Which optimizer to use.
      quantile_regression: Whether to output quantile predictions.
      quantiles: Quantile values used for quantile predictions.
      perform_training: Flag to indicate if the function should perform training
        or load a saved model.
      saved_model_bucket: GCS bucket to save model in.
      saved_model_name: Name of saved model to either load from if training is
        not performed or save to if training is performed.
      xgboost_train_only: Return without further training after XGBoost train.
    """

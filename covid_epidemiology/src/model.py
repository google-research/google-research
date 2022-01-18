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

"""The model interface that should be implemented by all the models."""
import datetime
from typing import Dict, List, Optional, Tuple

import constants
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytz

from covid_epidemiology.src.models.shared import typedefs


@dataclass
class Prediction:
  # The offset of the day for which the prediction is made. 1 means the day
  # after the training_window_end.
  time_horizon: int
  predicted_value: float
  ground_truth: float


@dataclass
class WindowPrediction:
  # The last day of the training window.
  training_window_end: datetime.date
  # A map from a metric (e.g., death, ...) to a list of predictions. The size
  # of the list equals num_forecast_steps.
  metric_to_predictions: Dict[str, List[Prediction]]


@dataclass
class ModelOutput:
  # A map from the location to a list of window predictions. The number of
  # window predictions equals the number of training windows used.
  location_to_window_predictions: Dict[str, List[WindowPrediction]]


def _extract_first_and_last_days(
    ts_data):
  # pylint: disable=g-tzinfo-replace
  return ts_data[constants.DATE_COLUMN].min().replace(
      tzinfo=pytz.UTC), ts_data[constants.DATE_COLUMN].max().replace(
          tzinfo=pytz.UTC)


class Model:
  """Model class."""

  # TODO(b/153386569): Implement another moving window function based on
  # sufficient data points in the training window.

  def fit_forecast_moving_window(
      self,
      init_window_size,
      window_size_step,
      num_forecast_steps,
      num_train_forecast_steps,
      static_features,
      ts_features,
      ts_data,
      ts_state_features,
      locations,
      num_iterations,
      display_iterations,
      optimization,
      quantile_regression,
      training_data_generator,
      perform_training = True,
      saved_model_bucket = None,
      saved_model_name = None,
      xgboost_train_only = False,
      first_day = pd.to_datetime(constants.STARTING_DATE, utc=True),
      ts_overrides = None,
      static_overrides = None,
      ts_categorical_features = None,
      reduced_output = False,
      reduced_quantile_list = True,
      static_scalers = None,
      ts_scalers = None,
      ts_state_scalers = None
  ):
    """Fits the model using a moving window of training.

    Args:
      init_window_size: The initial training window size.
      window_size_step: The number of days added to the training window after
        each iteration. If 0, do one iteration only.
      num_forecast_steps: Number of forecasting steps at inference.
      num_train_forecast_steps: Number of forecasting steps at training.
      static_features: A panda dataframe containing all static input data.
      ts_features: A panda dataframe containing all time-series data.
      ts_data: A panda dataframe containing the timestamp column named
        constants.DATE_COLUMN. The dataframes static_data, ts_features,
        ts_overrides, ts_data must have rows sorted by date (with no missing
        days), and each row in static_data,ts_features, and ts_overrides must
        correspond to the date in ts_data in the same row.
      ts_state_features: A panda dataframe containing all state time-series.
      locations: List of locations to fit the model.
      num_iterations: Number of iterations to fit.
      display_iterations: Display period.
      optimization: Optimization algorithm.
      quantile_regression: Whether to use quantile regression.
      training_data_generator: Indicates whether Synthetic data generator is
        training.
      perform_training: Whether to perform training.
      saved_model_bucket: Bucked of the saved model.
      saved_model_name: Named of the saved model.
      xgboost_train_only: Whether to use it for covariate training only.
      first_day: The first date for modeling.
      ts_overrides: A panda dataframe containing all time-series feature
        overrides.
      static_overrides:  A panda dataframe containing all static feature
        overrides.
      ts_categorical_features: List of categorical features to apply overrides
        correctly, only needed if ts_overrides is not None.
      reduced_output: Reduced eval_training output for reducing BQ issue.
      reduced_quantile_list: Reduced qualtile list.
      static_scalers: Dictionary of fitted scalers for each feature.
      ts_scalers: Dictionary of fitted scalers for each feature.
      ts_state_scalers: Dictionary of fitted scalers for each feature.

    Returns:
      The model output in a format ready to be written to BQ.

    """
    if not perform_training:
      if not saved_model_name:
        raise ValueError(
            "saved_model_name is required when perform_training=False")
      if window_size_step != 0:
        raise ValueError(
            "window_size_step must be 0 when perform_training=False")

    first_day_per_data, last_day = _extract_first_and_last_days(ts_data)
    first_day = first_day or first_day_per_data

    train_window_end_index = init_window_size
    location_to_window_predictions_forecast = {}
    location_to_window_predictions_all = {}
    for location in locations:
      location_to_window_predictions_forecast[location] = []
      location_to_window_predictions_all[location] = []
    train_window_end_date = first_day + np.timedelta64(train_window_end_index,
                                                       "D")

    # train_window_end_date is the first date of the forecast window.
    # We can train up to and including the last_day in the training dataset.
    if train_window_end_date > (last_day + np.timedelta64(1, "D")):
      raise ValueError(
          "Training window of size {window_size} extends beyond "
          "available data. The last available date is "
          "{last_available}, the last date in the requested "
          "training window is {last_in_train_window}. "
          "Use window size less than or equal to the training data size."
          .format(
              window_size=init_window_size,
              last_available=last_day,
              last_in_train_window=(train_window_end_date -
                                    np.timedelta64(1, "D"))))

    while train_window_end_date <= (last_day + np.timedelta64(1, "D")):
      # pylint: disable=assignment-from-no-return
      (model_output_forecast, model_output_all) = self.fit_forecast_fixed(
          train_window_end_index=train_window_end_index,
          train_window_end_date=train_window_end_date,
          num_forecast_steps=num_forecast_steps,
          num_train_forecast_steps=num_train_forecast_steps,
          static_features=static_features,
          static_scalers=static_scalers,
          static_overrides=static_overrides,
          ts_features=ts_features,
          training_data_generator=training_data_generator,
          ts_scalers=ts_scalers,
          ts_overrides=ts_overrides,
          ts_categorical_features=ts_categorical_features,
          ts_state_features=ts_state_features,
          ts_state_scalers=ts_state_scalers,
          locations=locations,
          num_iterations=num_iterations,
          display_iterations=display_iterations,
          optimization=optimization,
          quantile_regression=quantile_regression,
          perform_training=perform_training,
          saved_model_bucket=saved_model_bucket,
          saved_model_name=saved_model_name,
          xgboost_train_only=xgboost_train_only,
          reduced_output=reduced_output,
          reduced_quantile_list=reduced_quantile_list)
      for location in locations:
        location_to_window_predictions_forecast[location].extend(
            model_output_forecast.location_to_window_predictions[location])
        location_to_window_predictions_all[location].extend(
            model_output_all.location_to_window_predictions[location])
      if window_size_step == 0:
        break
      train_window_end_index += window_size_step
      train_window_end_date = first_day + np.timedelta64(
          train_window_end_index, "D")
    return (ModelOutput(location_to_window_predictions_forecast),
            ModelOutput(location_to_window_predictions_all))

  def fit_forecast_fixed(
      self,
      train_window_end_index,
      train_window_end_date,
      num_forecast_steps,
      num_train_forecast_steps,
      static_features,
      ts_features,
      ts_state_features,
      locations,
      num_iterations,
      display_iterations,
      optimization,
      quantile_regression,
      training_data_generator = False,
      perform_training = True,
      saved_model_bucket = None,
      saved_model_name = None,
      xgboost_train_only = True,
      ts_overrides = None,
      static_overrides = None,
      ts_categorical_features = None,
      reduced_output = False,
      reduced_quantile_list = True,
      static_scalers = None,
      ts_scalers = None,
      ts_state_scalers = None,
  ):
    """Fits the model and returns a ModelOutput.

    Args:
      train_window_end_index: Index of the training window end.
      train_window_end_date: Date of the training window end.
      num_forecast_steps: Number of forecasting steps at inference.
      num_train_forecast_steps: Number of forecasting steps at training.
      static_features: A panda dataframe containing all static input data.
      ts_features: A panda dataframe containing all time-series input data.
      ts_state_features: A panda dataframe containing all state time-series
        input data.
      locations: List of locations to fit the model.
      num_iterations: Number of iterations to fit.
      display_iterations: Display period.
      optimization: Optimization method.
      quantile_regression: Whether to apply quantile regression.
      training_data_generator: Whether to generate training data.
      perform_training: If False, saved_model_name must be not None. In this
        case, only inference will be performed on a previously trained model
        which will be restored from $saved_model_name.
      saved_model_bucket: GCS bucket to store models in.
      saved_model_name: if not None and perform_training is True, the trained
        model will be saved to this filename. If perform_training is False, the
        trained model will be restored from this filename.
      xgboost_train_only: Return without further training after XGBoost train.
      ts_overrides: A panda dataframe containing all time-series feature
        overrides.
      static_overrides: A panda dataframe containing all static feature
        overrides.
      ts_categorical_features: List of categorical features to apply overrides
        correctly, only needed if ts_overrides is not None.
      reduced_output: reduced eval_training output for reducing BQ issue.
      reduced_quantile_list: Whether to reduce the list of quantiles to output.
      static_scalers: Dictionary of fitted scalers for each feature.
      ts_scalers: Dictionary of fitted scalers for each feature.
      ts_state_scalers: Dictionary of fitted scalers for each feature.

    Returns:
      An instance of ModelOutput.
    """
    pass

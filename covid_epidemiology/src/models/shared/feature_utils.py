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
"""Utility functions to extract features from the data."""

import logging
import multiprocessing
import os
import random
import re
import time
# from typing import Dict, Iterable, List, Tuple
import warnings

import numpy as np
import pandas as pd
from pandas_gbq import gbq
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
import xgboost

from covid_epidemiology.src import constants
from covid_epidemiology.src.models.shared import model_spec as model_spec_lib


def filter_data_based_on_location(
    static_data, ts_data,
    locations):
  return static_data[static_data[constants.COUNTRY_COLUMN].isin(
      locations)], ts_data[ts_data[constants.COUNTRY_COLUMN].isin(locations)]


def static_feature_map_for_locations(
    static_features,
    locations):
  """Returns the static features for a specific location."""
  resulting_features = {}
  for feature_name in static_features or {}:
    resulting_features[feature_name] = np.zeros((len(locations),))
    for location_index, location in enumerate(locations):
      if location in static_features[feature_name]:
        resulting_features[feature_name][location_index] = (
            static_features[feature_name][location])
  return resulting_features


def static_feature_to_dense(static_features, static_feature_specs, locations):
  """Returns a tensor of static features of [num_locations, num_features].

  Args:
    static_features: A dictionary of
      {feature_name: {location: feature_value}}
    static_feature_specs: FeatureSpecs for covariate feautres
    locations: A list of location keys to train for.
  """
  static_feature_values = np.zeros((len(locations), len(static_feature_specs)),
                                   dtype=np.float32)
  for feature_index, feature_spec in enumerate(static_feature_specs):
    if feature_spec.name in static_features:
      for location_index, location in enumerate(locations):
        if location in static_features[feature_spec.name]:
          static_feature_values[location_index, feature_index] = (
              static_features[feature_spec.name][location])
        else:
          raise ValueError(f"feature: {feature_spec.name} was not populated "
                           f"for location: {location}.")
    else:
      raise ValueError(f"Feature spec passed for feature: {feature_spec.name}, "
                       f"however this features is not populated.")

  return static_feature_values


def covariate_features_to_dense(covariates, covariate_feature_specs, locations,
                                num_observed_timesteps):
  """Returns a list of tensors with features for training.

  Args:
    covariates: A default_dict
      {feature_name: {location: [feature_value_t0, ..., feature_value_tn]}}
    covariate_feature_specs: FeatureSpecs for covariate feautres
    locations: A list of location keys to train for.
    num_observed_timesteps: The number of timesteps we have data for.

  Returns:
    A list of size [num_timesteps] with arrays of size
      [num_locations, num_features tensors] or an empty list if there are no
      covariates.
  """
  columns_in_order = [
      feature_spec.name for feature_spec in covariate_feature_specs
  ]
  if not columns_in_order:
    # Return an empty list
    return []

  location_set = set(locations)
  column_set = set(columns_in_order)
  # Only create the dataframe with the features we care about.
  # pylint: disable=g-complex-comprehension
  df = pd.DataFrame({(feature_name, location): arr
                     for feature_name, location_dict in covariates.items()
                     if feature_name in column_set and location_dict
                     for location, arr in location_dict.items()
                     if location in location_set})

  # Make the index compound with (timestep, location) x feature and make sure
  # the second level is indexed by location for all timesteps
  stacked = df.stack(1).reindex(locations, level=1)
  # This returns a list n_timesteps long of arrays n_locations x n_features
  return [
      stacked.loc[i, columns_in_order].to_numpy(dtype="float32")
      for i in range(num_observed_timesteps)
  ]


def covariate_overrides_to_dense(covariate_overrides, covariate_feature_specs,
                                 locations, num_observed_timesteps):
  """Returns a list of tensors with feature overrides for inference.

  Args:
    covariate_overrides: A dictionary of
      {feature_name: {location: [feature_override_t0, ...,
        feature_override_tn]}}
    covariate_feature_specs: FeatureSpecs for covariate feautres
    locations: A list of location keys to train for.
    num_observed_timesteps: The number of timesteps we have data for.

  Returns:
    A list of size [num_timesteps] with tensors of size
      [num_locations, num_features tensors].
  """
  covariate_overrides_over_time = []
  # some of these timeseries features can be categorical. categorical features
  # must be handled separately as it does semantically incorrect to merely
  # multiply them by and override multiplier.
  # we handle this in the calling function, by filtering by feature name. This
  # method is not ideal, but is expedient for now.
  for timestep in range(num_observed_timesteps):
    covariate_overrides_this_timestep = np.ones(
        (len(locations), len(covariate_feature_specs)), dtype=np.float32)

    covariate_overrides_over_time.append(covariate_overrides_this_timestep)
    for feature_index, feature_spec in enumerate(covariate_feature_specs):
      if feature_spec.name in covariate_overrides:
        for location_index, location in enumerate(locations):
          if location in covariate_overrides[feature_spec.name]:

            covariate_overrides_this_timestep[location_index, feature_index] = (
                covariate_overrides[feature_spec.name][location][timestep])
          else:
            raise ValueError(f"feature: {feature_spec.name} was not populated "
                             f"for location: {location}.")

  return covariate_overrides_over_time


def get_categorical_features_mask(
    feature_specs,
    categorical_features, num_locations,
    is_static):  # pylint: disable=unused-argument
  """Get a mask of shape locationsXfeatures that identifies categorical features.

  Generate a mask of shape (num_locations)X(num_total_features) that has a '1'
  where a categorical feature is present or a '0' where it is absent.

  Args:
    feature_specs: List of model_spec_lib.FeatureSpec. List of available
      features.
    categorical_features: List of string. List of categorical features to mask.
    num_locations: int. Number of geographic locations for mask.
    is_static: boolean. Flag to indicate if these features are static or ts.

  Returns:
    Tensor of shape (num_locations)X(num_all_features) containing the mask.

  Raises:
    AssertionError if the provided categorical features are not a subset of
    the available features.
    ValueError if number of locations is not positive.
  """

  # for the first version of the What-If Tool, we are filtering categorical
  # features by hardcoding a substring pattern as a filter.
  # for this we need to know the *positional* index of each feature override.
  # we get this from the positional index of features in the
  # covariate_feature_specs.
  if num_locations <= 0:
    raise ValueError(("Number of locations must be positive, received ",
                      "{}".format(num_locations)))
  # get positional indices of categorical features
  categorical_idxs = [
      int(f.name in categorical_features) for f in feature_specs
  ]
  # construct mask of shape (num_locations)X(num_all_features)
  categorical_mask_shape = (num_locations, len(feature_specs))
  categorical_mask = np.broadcast_to(categorical_idxs, categorical_mask_shape)
  return categorical_mask


def generate_prediction_features(features_all, features_all_smoothed,
                                 feature_end_indices, output_window_size,
                                 is_training, lags, averages, max_windows):
  """Generates derived input features derived to be used for forecasting."""

  num_locations = features_all.shape[0]
  num_features = len(lags) + len(averages) + len(max_windows)
  num_indices = len(feature_end_indices)

  all_features = np.zeros((num_locations * num_indices, num_features))

  if is_training:
    labels = np.zeros((num_locations * num_indices, output_window_size))
  else:
    labels = None

  for ind, feature_end_index in enumerate(feature_end_indices):

    feature_ind = 0

    # Add lag features
    for lag in lags:
      all_features[ind * num_locations:(ind + 1) * num_locations,
                   feature_ind] = features_all_smoothed[:,
                                                        feature_end_index - lag]
      feature_ind += 1

    # Add average features
    for average in averages:
      all_features[ind * num_locations:(ind + 1) * num_locations,
                   feature_ind] = np.mean(
                       features_all_smoothed[:, feature_end_index - average:],
                       axis=1)
      feature_ind += 1

    # Add max ratio features
    for max_window in max_windows:
      all_features[ind * num_locations:(ind + 1) * num_locations,
                   feature_ind] = np.max(
                       features_all_smoothed[:,
                                             feature_end_index - max_window:],
                       axis=1)
      feature_ind += 1

    if is_training:
      labels[
          ind * num_locations:(ind + 1) *
          num_locations, :] = features_all[:,
                                           feature_end_index:feature_end_index +
                                           output_window_size]

  return all_features, labels


def forecast_features(
    covariates,
    coviariate_feature_specs,
    num_forecast_steps = 14,
    smooth_coef = 1.0,
    max_num_training_samples = 50000,
    max_num_val_samples = 10000,
    num_threads = 1):
  """Forecasts future values of covariates.

  Args:
    covariates: Input time series covariates.
    coviariate_feature_specs: The feature specs for the covariates.
    num_forecast_steps: The number of steps to forecast.
    smooth_coef: The smoothing coefficent. Should be between 0 and 1.
    max_num_training_samples: The maximum number of training samples.
    max_num_val_samples: The maximum number of validation samples.
    num_threads: The number of parallel threads with which XGBoost should run.

  Returns:
    Forecasted covariates of size
      [num_forecast_steps x num_locations x num_features], same as covariates.

  """
  if num_threads == 0:
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is not None:
      logging_num_threads_str = str(omp_num_threads)
    else:
      logging_num_threads_str = str(multiprocessing.cpu_count())
  else:
    logging_num_threads_str = f"{num_threads}"

  print("Number of threads:")
  print(logging_num_threads_str)

  # Disable warnings
  # pylint: disable=unused-argument
  def warn(*args, **kwargs):
    pass

  warnings.warn = warn

  stacked_covariates = np.transpose(np.stack(covariates), [2, 1, 0])
  num_features, num_locations, training_window_end = stacked_covariates.shape
  forecasted_features = np.zeros(
      (num_features, num_locations, num_forecast_steps))

  max_train_timepoints = stacked_covariates.shape[-1]

  # Iterate over all the features except the first 4 as they are directly
  # inferred from the compartmental model.
  for i, covariate_spec in enumerate(coviariate_feature_specs.values()):
    input_covariates = stacked_covariates[i]
    if covariate_spec.forecast_method == model_spec_lib.ForecastMethod.NONE:
      continue
    # Use constant extrapolation on very short time series that are used for
    # testing
    elif (covariate_spec.forecast_method
          == model_spec_lib.ForecastMethod.CONSTANT or
          max_train_timepoints < 7):
      forecasted_features[i, Ellipsis] = input_covariates[:, -1:]
    elif covariate_spec.forecast_method == model_spec_lib.ForecastMethod.PERIODIC_WEEKLY:
      periodic_forecast(input_covariates, forecasted_features[i, Ellipsis], period=7)
    elif covariate_spec.forecast_method == model_spec_lib.ForecastMethod.XGBOOST:
      # XGBoost based forecasting of the feature values of the covariates.
      predicted = xgboost_forecast(input_covariates, training_window_end,
                                   num_forecast_steps, smooth_coef,
                                   max_num_training_samples,
                                   max_num_val_samples, num_threads)
      forecasted_features[
          i, :, :num_forecast_steps] = predicted[:, :num_forecast_steps]
    else:
      raise ValueError(
          f"Forecasting method {covariate_spec.forecast_method} is not supported."
      )

  forecasted_features = np.transpose(forecasted_features, [2, 1, 0])

  return forecasted_features


def xgboost_forecast(input_covariates,
                     training_window_end,
                     num_forecast_steps,
                     smooth_coef = 1.0,
                     max_num_training_samples = 100000,
                     max_num_val_samples = 10000,
                     num_threads = 1):
  """Forecast covariates using XGBoost classification or regression.

  Args:
    input_covariates:
    training_window_end: The number of points in the training data.
    num_forecast_steps: Number of forecasting steps.
    smooth_coef: Smoothing coefficient.
    max_num_training_samples: Maximum number of training samples.
    max_num_val_samples: Maximum number of validation samples.
    num_threads: The number of parallel threads with which XGBoost should run.

  Returns:
    An array of covariates for the input feature.
  """
  # Hyperparameter search space
  lr_candidate = [0.05, 0.1]
  n_estimators_candidate = [20, 100]
  max_depth_candidate = [5]
  subsample_candidate = [1.0]
  lambda_candidate = [1.0, 0.01]

  # Decide whether to treat the problem as classification or
  # regression based on the number of categories for labels.
  unique = np.unique(input_covariates.flatten())
  num_unique_values = unique.size
  if num_unique_values > 10:
    problem_type = "regression"
    # Note that the range may
    # not be in [0, 1] and with increasing and decreasing trends, covariate
    # values may go beyond the training data range.
    min_value = 0.0
    max_value = 1.0
  elif num_unique_values > 1:
    problem_type = "classification"
    le = preprocessing.LabelEncoder()
    le.fit(input_covariates.flatten())
  else:
    # If there is only one value return that value
    return (np.ones(
        (input_covariates.shape[0], num_forecast_steps)) * unique[0])

  if problem_type == "regression":
    # Smoothing
    input_covariates_smoothed = np.copy(input_covariates)
    for t in range(1, input_covariates.shape[1] - 1):
      input_covariates_smoothed[:, t] = (
          smooth_coef * input_covariates_smoothed[:, t] + (1.0 - smooth_coef) *
          (input_covariates_smoothed[:, t - 1]))
  else:
    input_covariates_smoothed = input_covariates

  # Construct datasets.

  # Actual case vs. for short-ter integration tests.
  if training_window_end > 5 * num_forecast_steps:
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    averages = [3, 5, 7, 14, 21, 28]
    max_windows = [3, 5, 7, 14, 21, 28]
    if problem_type == "classification":
      training_window_beginning = num_forecast_steps
    else:
      training_window_beginning = max(
          training_window_end - 5 * num_forecast_steps, 0)
  else:
    lags = [1, 2, 3, 4, 5, 6, 7]
    averages = [3, 5, 7]
    max_windows = [3, 5, 7]
    training_window_beginning = 8

  # Train
  train_range_end = max(training_window_end - 2 * num_forecast_steps,
                        training_window_beginning + 1)
  if input_covariates.shape[1] - train_range_end <= num_forecast_steps:
    raise ValueError(
        f"Cannot forecast covariates {num_forecast_steps} days with only "
        f"{input_covariates.shape[1]} days of data")
  train_features, train_labels = generate_prediction_features(
      input_covariates,
      input_covariates_smoothed,
      range(training_window_beginning, train_range_end),
      num_forecast_steps,
      is_training=True,
      lags=lags,
      averages=averages,
      max_windows=max_windows)

  num_training_points, label_dim = train_labels.shape
  if num_training_points > max_num_training_samples:
    np.random.seed(seed=1)
    indices = np.random.choice(train_features.shape[0],
                               max_num_training_samples)
    train_features = train_features[indices, :]
    train_labels = train_labels[indices, :]
    num_training_points = max_num_training_samples

  # Validation
  val_features, val_labels = generate_prediction_features(
      input_covariates,
      input_covariates_smoothed,
      range(train_range_end,
            max(training_window_end - num_forecast_steps, train_range_end + 1)),
      num_forecast_steps,
      is_training=True,
      lags=lags,
      averages=averages,
      max_windows=max_windows)

  num_val_points = val_features.shape[0]
  if num_val_points > max_num_val_samples:
    np.random.seed(seed=1)
    indices = np.random.choice(val_features.shape[0], max_num_val_samples)
    val_features = val_features[indices, :]
    val_labels = val_labels[indices, :]
    num_val_points = max_num_val_samples

  # Test
  test_features, _ = generate_prediction_features(
      input_covariates,
      input_covariates_smoothed,
      range(training_window_end, training_window_end + 1),
      num_forecast_steps,
      is_training=False,
      lags=lags,
      averages=averages,
      max_windows=max_windows)
  num_test_points = test_features.shape[0]

  optimal_err = 1e128
  for lr in lr_candidate:
    for n_estimators in n_estimators_candidate:
      for max_depth in max_depth_candidate:
        for subsample in subsample_candidate:
          for lambda_v in lambda_candidate:

            if problem_type == "regression":
              multioutputpredictor = MultiOutputRegressor(
                  xgboost.XGBRegressor(
                      learning_rate=lr,
                      n_estimators=n_estimators,
                      max_depth=max_depth,
                      subsample=subsample,
                      reg_lambda=lambda_v,
                      objective="reg:squarederror",
                      n_jobs=num_threads)).fit(train_features, train_labels)
            elif problem_type == "classification":
              multioutputpredictor = MultiOutputClassifier(
                  xgboost.XGBClassifier(
                      learning_rate=lr,
                      n_estimators=n_estimators,
                      max_depth=max_depth,
                      subsample=subsample,
                      reg_lambda=lambda_v,
                      n_jobs=num_threads)).fit(
                          train_features,
                          le.transform(
                              train_labels.reshape(
                                  (num_training_points * label_dim))).reshape(
                                      (num_training_points, label_dim)))

            val_predicted = multioutputpredictor.predict(val_features)
            if problem_type == "classification":
              val_predicted = le.inverse_transform(
                  val_predicted.reshape((num_val_points * label_dim))).reshape(
                      (num_val_points, label_dim))

            val_err = np.mean((val_predicted - val_labels)**2)

            if val_err < optimal_err:
              multioutputpredictor_opt = multioutputpredictor
              optimal_err = val_err

  # Generate forecasts
  test_predicted = multioutputpredictor_opt.predict(test_features)
  if problem_type == "classification":
    test_predicted = le.inverse_transform(
        test_predicted.reshape((num_test_points * label_dim))).reshape(
            (num_test_points, label_dim))

  else:
    # Clip to be in the range.
    test_predicted = np.clip(test_predicted, min_value, max_value)

  return test_predicted


def periodic_forecast(
    covariate,
    output_array,
    period,
):
  """Repeats the covariate with a periodic schedule.

  Args:
    covariate: Assumed to have dimensions [# locations x # days].
    output_array: The output array where the results will be written.
    period: The period with which to repeat the data.
  """
  num_forecast_steps = output_array.shape[1]
  n_periods = int(np.ceil(num_forecast_steps / float(period)))
  num_days = covariate.shape[1]
  last_week = covariate[:, -period:]
  if num_days < period:
    n_blocks = int(np.ceil(period / num_days))
    last_week = np.tile(last_week, (1, n_blocks))[:, -period:]

  repeated_weeks = np.tile(last_week, (1, n_periods))
  output_array[:, :] = repeated_weeks[:, :num_forecast_steps]


def cov_pred(ts_features, num_train_steps, num_forecast_steps_sliced,
             num_forecast_steps, feature_specs, chosen_location_list,
             num_threads, covariate_training_mixing_coef):
  """Covariate forecasting.

  Args:
    ts_features: Time-series features (Dictionary with num_features elements of
      size [num_locations, num_known_timesteps]).
    num_train_steps: Number of training timesteps.
    num_forecast_steps_sliced: Number of forecast steps sliced to be used in
      training.
    num_forecast_steps: Number of forecast steps for output.
    feature_specs: Covariate feature specs names for forecasting.
    chosen_location_list: List of chosen locations.
    num_threads: The number of parallel threads with which XGBoost should run.
    covariate_training_mixing_coef: Coefficient for mixture of actual vs.
      forecasted covariate ratio. 0 = use forecasted values, 1 = use actual gt
      values.

  Returns:
    forecasted_covariates: Forecasted future time-series features (Dictionary
      with num_features elements of size
      [num_forecast_steps_sliced, num_locations].
  """

  # Transforms ts_features to numpy matrix
  covariates_names = feature_specs.keys()
  cov_array = None
  for cov_idx, cov_name in enumerate(covariates_names):
    df = pd.DataFrame.from_dict(ts_features[cov_name])
    df = df[chosen_location_list]
    if cov_array is None:
      cov_array = np.empty(
          (df.shape[0], df.shape[1], len(feature_specs)),
          dtype=np.float32,
      )
    cov_array[:, :, cov_idx] = df.values

  # Forecast future covariates
  if cov_array.shape[0] >= num_train_steps + num_forecast_steps_sliced:
    gt_covariates_array = cov_array[num_train_steps:num_train_steps +
                                    num_forecast_steps_sliced, :, :].copy()
  else:
    gt_covariates_array = None

  if covariate_training_mixing_coef < 1.0 or gt_covariates_array is None:
    train_cov_array = cov_array[:num_train_steps, :, :].copy()
    forecasted_cov = forecast_features(
        train_cov_array,
        feature_specs,
        num_forecast_steps,
        num_threads=num_threads)
    forecasted_cov = np.stack(forecasted_cov)

  ts_mixed_covariates = dict()
  for i, cov_name in enumerate(covariates_names):
    if covariate_training_mixing_coef == 0.0 or gt_covariates_array is None:
      ts_mixed_covariates[
          cov_name] = forecasted_cov[:num_forecast_steps_sliced, :, i]
    elif covariate_training_mixing_coef == 1.0:
      ts_mixed_covariates[cov_name] = gt_covariates_array[:, :, i]
    else:
      ts_mixed_covariates[cov_name] = (
          covariate_training_mixing_coef * gt_covariates_array[:, :, i]) + (
              (1.0 - covariate_training_mixing_coef) *
              forecasted_cov[:num_forecast_steps_sliced, :, i])

  return ts_mixed_covariates


def extract_forecasted_features(forecasted_covariates, covariate_feature_specs):
  """Extracts forecasted values of selected covariates.

  Args:
    forecasted_covariates: Forecasted time-series covariates
    covariate_feature_specs: Selected features to be extracted

  Returns:
    extracted_covariates: Extracted time-series covariates with shape
      output window size x #num_locations x #num_features
  """

  extracted_covariates = list()
  for cov in covariate_feature_specs:
    extracted_covariates.append(forecasted_covariates[cov.name])
  extracted_covariates = np.stack(extracted_covariates, axis=2)

  return extracted_covariates


def get_lasso_feature_mask(
    feature_specs):
  """Gets a mask of features to apply lasso to.

  Args:
    feature_specs: List of model_spec_lib.FeatureSpec. List of available
      features.

  Returns:
    Tensor of shape (1 x num_all_features) containing the mask.
  """
  return tf.constant([1.0 if fs.apply_lasso else 0.0 for fs in feature_specs])


def read_from_a_project(bq_query,
                        all_projects = (
                            constants.PROJECT_ID_WHAT_IF_INFERENCE,
                            constants.PROJECT_ID_MODEL_TRAINING),
                        verbose = False):
  """Attempts to load a BigQuery table from both projects.

  The query will be executed and results returned from the first project that
  has the specified table.

  Args:
    bq_query: The query to be executed.
    all_projects: The projects where the query should be run.
    verbose: If True messages about where the table was found will be logged.

  Returns:
    The DataFrame resulting from the query.

  Raises:
      ValueError: If no projects are specified.
      NotFoundException: If the query could not be completed anywhere.
  """
  output_exception = ValueError("A project must be specified.")
  attempted_projects = set()
  for current_project in all_projects:
    if current_project in attempted_projects:
      continue
    try:
      downloaded_df = pd.read_gbq(bq_query, project_id=current_project)
      if verbose:
        logging.info("Executed query %s in %s", bq_query, current_project)
      return downloaded_df
    except (gbq.NotFoundException, gbq.GenericGBQException) as e:
      if verbose:
        logging.info("Did not find %s in %s: %s", bq_query, current_project,
                     repr(e))
      # Handle 4XX
      if (isinstance(e, gbq.GenericGBQException) and
          not re.search(r"Reason: 4\d{2}", repr(e))):
        logging.info("Unhandled GenericGBQException for %s in %s: %s", bq_query,
                     current_project, repr(e))
        raise
      attempted_projects.add(current_project)
      output_exception = gbq.NotFoundException(f"Not Found: {repr(e)}")

  raise output_exception


def write_covariates_to_bigquery(
    covariates_dict,
    location_list,
    table_name,
    max_attempts = 5,
    initial_retry = 30,
    max_retry_delay = 600,
):
  """Uploads a covariates dictionary to bigquery.

  Args:
    covariates_dict: The dictionary of covariates. This dictionary maps
      covariates to a list of lists. The outer list corresponds to time horizons
      and the inner list corresponds to location ids.
    location_list: The list of locations for which covariates were forecasted.
    table_name: Name of the target BigQuery table.
    max_attempts: The maximum number of times to try to either write or read the
      covariates from BigQuery before raising an exception
    initial_retry: The number of seconds to wait for the first retry. Retries
      after the first are exponentially backed off with jitter.
    max_retry_delay: Maximum number of seconds to wait before retrying.

  Returns:
    A potentially updated covariates_dict. If a table already exists it will be
      read in order to ensure that all covariates for a run are equal.

  Raises:
    IOError: If the covariates could not be written to BiqQuery after the
      specified number of attempts.
  """
  from utils import get_evaluation_tables  # pylint: disable=g-import-not-at-top

  def _wait_time(attempt):
    return min(random.random() * initial_retry * 2**(max_attempts - attempt),
               max_retry_delay)

  ts_covariate_column_names = [
      "location_id", "time_horizon", "predicted_metric", "point_prediction"
  ]
  ts_covariate_rows = []
  for (metric, values) in covariates_dict.items():
    for time_ix in range(len(values)):
      loc_values = values[time_ix]
      if len(loc_values) != len(location_list):
        raise ValueError(
            "Number of locations in covariate forecast does not match expected location list."
        )
      for ix in range(len(loc_values)):
        location = location_list[ix]
        prediction = loc_values[ix]
        ts_covariate_rows.append([location, time_ix, metric, prediction])
  ts_covariate_pd = pd.DataFrame(
      ts_covariate_rows, columns=ts_covariate_column_names)

  while max_attempts > 0:
    if get_evaluation_tables.predicted_metric_table_completed(
        table_name,
        constants.PROJECT_ID_WHAT_IF_INFERENCE,
        remove_incomplete_table=True):
      logging.info("Found an existing covariate table %s. Reading covariates.",
                   table_name)
      return read_covariates_from_bigquery(table_name)

    try:
      ts_covariate_pd.to_gbq(
          table_name,
          project_id=constants.PROJECT_ID_WHAT_IF_INFERENCE,
          if_exists="fail")
      logging.info("Successfully wrote covariates to %s in %s", table_name,
                   constants.PROJECT_ID_WHAT_IF_INFERENCE)
      return covariates_dict
    except (gbq.GenericGBQException, gbq.TableCreationError) as e:
      seconds_to_sleep = _wait_time(max_attempts)
      logging.error(
          "Failed to write covariates to %s due to %s. %s attempts remaining. "
          "Waiting %s seconds before retrying.", table_name, repr(e),
          max_attempts - 1, seconds_to_sleep)
      time.sleep(seconds_to_sleep)

    max_attempts -= 1

  raise IOError("Could not successfully write covariates to BigQuery")


def read_covariates_from_bigquery(
    table_name):
  """Downloads a set of forecasted covariates from bigquery.

  Args:
    table_name: Name of bigquery table to download from.

  Returns:
    A dictionary mapping metric names to a list of lists. The inner list
    contains covariate values sorted by location id. The outer list contains
    inner lists sorted by time horizon.
  """
  ts_covariates_pd = read_from_a_project(
      f"select * from {table_name}", verbose=True)

  ts_covariates_dict = {}
  metrics = list(ts_covariates_pd.predicted_metric.unique())
  metrics.sort()
  for metric in metrics:
    ts_covariates_dict[metric] = []
    ts_covariates_metric = ts_covariates_pd.loc[
        ts_covariates_pd["predicted_metric"] == metric]
    time_values = list(ts_covariates_metric.time_horizon.unique().astype(int))
    time_values.sort()
    for current_time in time_values:
      ts_covariates_time = ts_covariates_metric.loc[
          ts_covariates_metric["time_horizon"] == current_time]
      ts_covariates_time_sorted = ts_covariates_time.sort_values(
          by=["location_id"])
      ts_covariates_dict[metric].append(
          list(ts_covariates_time_sorted["point_prediction"]))
  return ts_covariates_dict

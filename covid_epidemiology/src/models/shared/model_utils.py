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

"""Utility functions to extract features from the data."""
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from covid_epidemiology import model
from covid_epidemiology.src import constants


def apply_sigmoid_bounds(variable, lower_bound, upper_bound):
  """Applies soft bounding using sigmoid nonlinearity.

  Args:
    variable: Input tensor.
    lower_bound: Lower bound.
    upper_bound: Upper bound.

  Returns:
    Bounded tensor.
  """
  return lower_bound + (upper_bound - lower_bound) * tf.nn.sigmoid(variable)


def apply_relu_bounds(variable, lower_bound, upper_bound, replace_nan=True):
  """Applies hard bounding using ReLU nonlinearity.

  Args:
    variable: Input tensor.
    lower_bound: Lower bound.
    upper_bound: Upper bound.
    replace_nan: Whether to replace NaNs.

  Returns:
    Bounded tensor.
  """

  bounded_variable = tf.nn.relu(variable - lower_bound) + lower_bound

  bounded_variable = upper_bound - tf.nn.relu(upper_bound - bounded_variable)

  if replace_nan:
    bounded_variable = tf.where(
        tf.math.is_nan(bounded_variable), tf.zeros_like(bounded_variable),
        bounded_variable)

  return bounded_variable


def populate_gt_list(index, location_to_gt,
                     location, num_observed_timesteps,
                     gt_list, gt_indicator):
  """Copies the ground truth of a location to a corresponding index of gt_list.

  Args:
    index: The index of the location.
    location_to_gt: A map from location to ground truth time-series.
    location: The location
    num_observed_timesteps: Number of observed time-series.
    gt_list: The ground truth list.
    gt_indicator: The binary indicator whether the particular value exists.

  Returns:
    The ground truth list and the indicator.
  """

  if location_to_gt and location_to_gt[location].any():
    observed_gt = location_to_gt[location][:num_observed_timesteps]
    observed_indicator = 1.0 - np.isnan(observed_gt)
    fill_values = observed_gt * observed_indicator
    fill_values[np.isnan(fill_values)] = 0
    gt_list[index, :observed_gt.shape[0]] = fill_values
    gt_indicator[index, :observed_gt.shape[0]] = observed_indicator
  return gt_list, gt_indicator


def search_county_indice(chosen_location_list, location):
  idx1 = np.where(
      np.asarray(chosen_location_list).astype(int) > int(location))[0]
  idx2 = np.where(
      np.asarray(chosen_location_list).astype(int) < int(location) + 1000)[0]
  idx = list(np.intersect1d(idx1, idx2))
  return idx


def filter_data_based_on_location(
    static_data,
    ts_data,
    locations  # pylint: disable=g-bare-generic
):
  return static_data[static_data[constants.COUNTRY_COLUMN].isin(
      locations)], ts_data[ts_data[constants.COUNTRY_COLUMN].isin(locations)]


def update_metric_to_predictions(
    metric,
    values,
    metric_to_predictions,
    train_end_of_window,
    gt_data,
    time_horizon_offset = 0,
    quantiles = None,
    quantiles_output = None,
    metric_string_format = None
):
  """Updates a given metric_to_predictions dictionary using given values.

  Args:
    metric: The metric for which the new predictions are added.
    values: The predictions by the model. This is a numpy array of
      single-element lists. Shape:(num_forecast_steps, 1)
    metric_to_predictions: The dictionary that will be updated.
    train_end_of_window: The train_end_window used for training the model.
    gt_data: The ground truth data for the same metric.  Shape:(N), where N is
      the total number of time steps in the entire time series.  Note that
      gt_data contains the entire time series, while values contains only
      predicted future values in the forecasting window, starting with
      train_end_of_window. Therefore, we will compare values with
    gt_data[train_end_of_window:train_end_of_window + num_forecast_steps]
    time_horizon_offset: Optional integer offset to be subtracted from
      the time horizon value of the prediction. It can be used to ensure the
      first prediction data point starts from time_horizon=1 when training
      window is also in values.
    quantiles: Defines quantile values used in the quantile forecast. None if
      quantile forecast is not used.
    quantiles_output: If defined, only export quantile values in this list.
    metric_string_format: Defines an optional metric string pattern in returned
    dict. It will be formatted with metric (and quantile if quantile forecast is
      used).

  Returns:
    The updated dictionary mapping metrics to model predictions.
  """
  if quantiles is None:
    # Provide an one element list for this corner case
    # in order to output value[0]
    quantiles = [None]
    quantiles_output = [None]
  elif quantiles_output is None:
    quantiles_output = quantiles

  for index, quantile in enumerate(quantiles):
    if quantile not in quantiles_output:
      continue
    predictions = []
    for i, value in enumerate(values):
      time_horizon = i + train_end_of_window
      # TODO(aepshtey): Add more tests to test the data reading and evaluation.
      if gt_data is not None and len(gt_data) > time_horizon:
        predictions.append(
            model.Prediction(i + 1 - time_horizon_offset, value[index],
                             gt_data[time_horizon]))
      else:
        predictions.append(
            model.Prediction(i + 1 - time_horizon_offset, value[index], None))
    if metric_string_format is not None:
      metric_string = metric_string_format.format(
          metric=metric, quantile=quantile)
    else:
      metric_string = metric
    metric_to_predictions[metric_string] = predictions

  return metric_to_predictions


def inv_sig(x, lb=0, ub=1):
  """Inverse of sigmoid function given the bounds."""

  assert x > lb
  return np.log((x-lb)/np.max([ub-x, 1e-6]))


def inv_softplus(x):
  """Inverse of softplus function."""
  assert x >= 0
  if x > 15:
    return x
  else:
    return np.log(np.exp(x) - 1 + 1e-6)


def set_random_seeds(random_seed):
  """Set all random seeds.

  Set the random seeds for the calling environment, Numpy, Tensorflow and
  Python.

  Args:
    random_seed: int. Value for seed.

  Returns:
    None.
  """

  os.environ['PYTHONHASHSEED'] = str(random_seed)
  tf.random.set_seed(random_seed)
  np.random.seed(random_seed)
  random.seed(random_seed)


def compartment_base(gt_list, gt_indicator, num_train_steps, num_known_steps):
  """Computes base of compartment coefficients.

  Args:
    gt_list: ground truth list
    gt_indicator: ground truth indicator
    num_train_steps: training window
    num_known_steps: number of known timesteps

  Returns:
    train_compartment_base: base of train_coef for each compartment
    valid_compartment_base: base of valid_coef for each compartment
  """
  train_compartment_base = (
      np.sum(gt_list[:num_train_steps] * gt_indicator[:num_train_steps]) /
      (np.sum(gt_indicator[:num_train_steps])))
  valid_compartment_base = (
      np.sum(gt_list[num_train_steps:num_known_steps] *
             gt_indicator[num_train_steps:num_known_steps]) /
      (np.sum(gt_indicator[num_train_steps:num_known_steps])))
  return train_compartment_base, valid_compartment_base


def increment_compartment_base(gt_list, gt_indicator, num_train_steps,
                               num_known_steps):
  """Computes base of compartment coefficients for increment loss.

  Args:
    gt_list: ground truth list
    gt_indicator: ground truth indicator
    num_train_steps: training window
    num_known_steps: number of known timesteps

  Returns:
    train_compartment_increment_base: base of train_coef for each compartment
      increment
    valid_compartment_increment_base: base of valid_coef for each compartment
      increment
  """

  num_forecast_steps = num_known_steps - num_train_steps

  gt_list_increment = (
      gt_list[num_forecast_steps:num_known_steps] -
      gt_list[:num_known_steps - num_forecast_steps])
  gt_indicator_increment = (
      gt_indicator[num_forecast_steps:num_known_steps] *
      gt_indicator[:num_known_steps - num_forecast_steps])

  train_compartment_increment_base, valid_compartment_increment_base = compartment_base(
      gt_list_increment, gt_indicator_increment,
      num_train_steps - num_forecast_steps, num_train_steps)

  return train_compartment_increment_base, valid_compartment_increment_base


def sync_compartment(gt_list, gt_indicator, compartment, sync_coef):
  """Synchronizes the ground truth and forecast.

  Args:
    gt_list: Ground truth time-series gt_indicator ground truth time-series
      indicator.
    gt_indicator: Indicator to denote availability of ground truth.
    compartment: Forecasted values for a certain compartment.
    sync_coef: Synchronization coefficient.

  Returns:
    synced_compartment: synchronized compartment value
  """
  synced_compartment_tf = tf.nn.relu(gt_list * sync_coef * gt_indicator)
  synced_compartment_pred = compartment * (1 - sync_coef * gt_indicator)
  synced_compartment = synced_compartment_tf + synced_compartment_pred
  return synced_compartment


def construct_infection_active_mask(
    confirmed_gt_list,
    num_locations,
    num_observed_timesteps,
    infected_threshold,
):
  """Creates a array that is 1 when the infection is active.

  Args:
    confirmed_gt_list: The locations are the keys and the values are the
      confirmed cases across time.
    num_locations: The number of locations.
    num_observed_timesteps: The number of observed time steps.
    infected_threshold: The minimum number of cases a location must have to be
      considered active

  Returns:
    An array of size num_observed_timesteps x num_locations that is 0.0 when
    the infection is not active and 1.0 when it is.
  """
  # We define infection_active_mask[ni, ti] such that at ni^th location,
  # at time ti, the infection is likely to be active. We will only start SEIR
  # dynamics if infection_active_mask[ni, ti]=1 for the location ni at
  # time ti. The motivation is that different locations start the infection
  # behavior at different timestep. A simple preprocessing step to
  # determine their relative position in time allows
  # more efficient fitting (i.e. the differential equations are not fit before
  # the disease is started).
  infection_active_mask = np.zeros((num_observed_timesteps, num_locations),
                                   dtype=np.float32)

  for location_index in range(num_locations):
    for timestep in range(num_observed_timesteps):
      if confirmed_gt_list[timestep, location_index] >= infected_threshold:
        infection_active_mask[timestep:, location_index] = 1.0
        break

  return infection_active_mask

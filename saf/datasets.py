# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Datasets for use in time series modeling."""

from data_utils import prepare_datasets_from_tables
from data_utils import tabular_functions
import numpy as np


def synthetic_autoregressive(synthetic_data_option=1,
                             normalization="standard",
                             len_total=3000):
  """Creates synthetic autoregressive datasets.

  Args:
    synthetic_data_option: Selection between the options: 1 - Scenario with
      abrupt changes in the data generating mechanism 2 - Scenario where the
      parameters of the data generating process smoothly drift 3 - Scenario
      where the changes can occur at random times 4 - Stationary random process
      scenario.
    normalization: Normalization option: None, "standard" or "min_max"..
    len_total: Number of data samples

  Returns:
    A tuple of (train tf.data.Dataset, val tf.data.Dataset, test
    tf.data.Dataset, config dictionary). Each dataset yields a
    (time_series_input, static_input, labels) tuple per example containing data
    for one item at one timestep, where time_series_input is a tensor of shape
    (len_max_lookback, num_features), static_input is a tensor of shape
    (num_static,), and labels is a tensor of shape (forecast_horizon,). All
    shape values are represented in the returned config dictionary.
  """

  if synthetic_data_option == 1:
    y = np.zeros(len_total)
    for ni in range(len_total - 1):
      if ni >= 1000 and ni <= 2000:
        alpha = -0.9
      else:
        alpha = 0.9
      epsilon = np.random.normal(0, 0.05)
      y[ni + 1] = alpha * y[ni] + epsilon
  elif synthetic_data_option == 2:
    y = np.zeros(len_total)
    for ni in range(len_total - 1):
      alpha = (1 - ni / 1500.0)
      epsilon = np.random.normal(0, 0.05)
      y[ni + 1] = alpha * y[ni] + epsilon
  elif synthetic_data_option == 3:
    y = np.zeros(len_total)
    last_same_count = 1
    alpha = -0.5
    for ni in range(len_total - 1):
      if np.random.uniform() > 0.99995**last_same_count:
        last_same_count = 1
        # Change alpha based on the last observed quantity.
        if alpha == -0.5:
          alpha = 0.9
        elif alpha == 0.9:
          alpha = -0.5
      else:
        last_same_count += 1
      epsilon = np.random.normal(0, 0.05)
      y[ni + 1] = alpha * y[ni] + epsilon
  elif synthetic_data_option == 4:
    y = np.zeros(len_total)
    for ni in range(len_total - 1):
      alpha = -0.5
      epsilon = np.random.normal(0, 0.05)
      y[ni + 1] = alpha * y[ni] + epsilon
  else:
    raise ValueError("Option not defined.")

  # Below train-validation-test split are chosen to match the paper:
  # https://cs.nyu.edu/~mohri/pub/tsj.pdf by ranging len_total from 775 to 3020.

  dataset_params = {
      "train_start": 0,
      "test_end": len_total - 1,
      "len_val": 25,
      "len_test": 25,
      "num_static": 1,
      "num_items": 1,
      "num_features": 1,
      "forecast_horizon": 25,
      "len_max_lookback": 100,
      "normalization": normalization,
      "target_index": 0
  }

  # Conditions to make sure the experimental settings make sense so that the
  # last prediction date of training is followed by the first prediction date
  # of validation and the last prediction date of validation is followed by the
  # first prediction date of test.

  dataset_params["test_start"] = (
      dataset_params["test_end"] - dataset_params["len_test"] + 2 -
      dataset_params["forecast_horizon"] - dataset_params["len_max_lookback"])

  dataset_params["val_end"] = dataset_params["test_start"] + dataset_params[
      "len_max_lookback"] + 1

  dataset_params["val_start"] = (
      dataset_params["val_end"] - dataset_params["len_val"] + 2 -
      dataset_params["forecast_horizon"] - dataset_params["len_max_lookback"])

  dataset_params["train_end"] = dataset_params["val_start"] + dataset_params[
      "len_max_lookback"] + 1

  # The input data time_sequences has dimensions of [num_items, len_total,
  # num_features] The input data static has dimensions of [num_items,
  # num_static]
  synthetic_index = np.zeros(
      (dataset_params["num_items"], dataset_params["num_static"]))
  time_sequences_data = np.float32(np.reshape(y, [1, len_total, 1]))
  static_data = np.float32(synthetic_index)

  (time_sequences_data, output_shift,
   output_scale) = tabular_functions.time_series_normalization(
       time_sequences_data,
       dataset_params["target_index"],
       per_item=True,
       type=dataset_params["normalization"],
       num_items=dataset_params["num_items"])
  static_data = tabular_functions.static_normalization(
      static_data, type=dataset_params["normalization"])

  # Append the unnormalization parameters to static_data for batching.
  static_data = np.concatenate([
      static_data,
      np.expand_dims(output_shift, -1),
      np.expand_dims(output_scale, -1)
  ],
                               axis=1)
  dataset_params["num_static"] += 2
  dataset_params["static_index_cutoff"] = 2

  input_tables = {"time_sequences": time_sequences_data, "static": static_data}

  (train_dataset, valid_dataset,
   test_dataset) = prepare_datasets_from_tables.return_datasets(
       input_tables=input_tables, shuffle_train_items=True, **dataset_params)

  return train_dataset, valid_dataset, test_dataset, dataset_params

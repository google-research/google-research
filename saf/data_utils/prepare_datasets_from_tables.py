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

"""Prepares training, validation and tests datasets from given tables."""

import gc
from typing import Optional

import numpy as np
import tensorflow as tf


def create_windowed_dataset(
    dataset,
    len_max_lookback,
    forecast_horizon,
    num_items = None,
    num_parallel_calls=None,
):
  """Creates a dataset with lookback windows given a dataset of timesteps.

  Args:
    dataset: A tf.data.Dataset where each example is a tensor of shape
      (num_timesteps, ...).
    len_max_lookback: The length of each lookback window.
    forecast_horizon: The length of the future forecast window.
    num_items: The number of items in the data. If equal to the number of items
      the data will cycle by item and then by time step (e.g. X1[0], X2[0], ...,
      X1[1], x2[1], ...). If 0 or None the opposite will occur (e.g. X1[0],
      X1[1], ..., x2[0], x2[1], ...).
    num_parallel_calls: Number of threads to use for processing and interleaving
      results. None is no parallelization, while tf.data.experimental.AUTOTUNE
      sets it automatically based on the number of CPUs. If we have a large
      number of items this should be set to None to avoid creating too many
      threads.

  Returns:
    A tf.data.Dataset where each example is a tensor of shape (len_max_lookback
    + forecast_horizon, ...), and the dataset iterates over all lookback windows
    for all examples (moving forward one step at a time within num_timesteps),
    with the windows from each example interleaved. If cycle_by_item_first is
    True the same time step for all items will be returned first and then then
    the time step will increment. If it is false all the data for item 0 will be
    returned first, followed by the data for item 2, etc.
  """

  def create_windows(x):
    # x is a tensor of shape (num_timesteps, ...). We create a dataset from this
    # of length num_timesteps, and the window method then yields a dataset of
    # sub datasets each of length len_max_lookback + forecast_horizon for all
    # lookback windows. Those sub datasets are batched such that there is a
    # single example of shape (len_max_lookback + forecast_horizon, ...) per sub
    # dataset, and then the dataset of sub datasets is flat mapped to yield a
    # single dataset with a length equal to the number of lookback windows.
    len_window = len_max_lookback + forecast_horizon
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.window(len_window, shift=1)
    dataset = dataset.flat_map(
        lambda sub_ds: sub_ds.batch(len_window, drop_remainder=True))
    return dataset

  # Each example in the original dataset is mapped to a dataset of lookback
  # windows. The order in which these are returned depends on the cycle length.
  # If the cycle length is 1 all of the timesteps for each item will be returned
  # followed by the next item. If the cycle length is equal to the number of
  # items then each of the items for one time step will be returned followed by
  # other timesteps.
  cycle_length = num_items or 1
  return dataset.interleave(
      create_windows,
      cycle_length=cycle_length,
      num_parallel_calls=num_parallel_calls)


def return_datasets(
    input_tables,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end,
    num_static,
    forecast_horizon,
    len_max_lookback,
    target_index,
    shuffle_train_items=False,
    shuffle_after_windowing=None,
    max_train_examples=None,
    cycle_items_first=True,
    **unused_kwargs,
):
  """Prepares the datasets for training, validation, and testing.

  Args:
    input_tables: A dictionary of NumPy arrays containing a `time_sequences`
      array of shape (num_items, len_labeled_timesteps, num_features), and a
      `static` array of shape (num_items, num_static).
    train_start: The start index for the training data.
    train_end: The end index for the training data.
    val_start: The start index for the validation data.
    val_end: The end index for the validation data.
    test_start: The start index for the test data.
    test_end: The end index for the test data.
    num_static: The number of static features.
    forecast_horizon: The number of time-steps that will be forecast.
    len_max_lookback: The maximum number of time-step previous to the prediction
    target_index: The index of the target feature in the input_tables.
    shuffle_train_items: Whether or not to reproducibly shuffle the training
      examples. This will apply to the initial dataset that is of length
      num_items prior to the windowing operations.
    shuffle_after_windowing: Whether or not to reproducibly shuffle the training
      examples after windowing. If True the model will be presented data in a
      time-randomized order.
    max_train_examples: Maximum number of training examples to yield. By
      default, all examples are kept.
    cycle_items_first: If true the data will cycle by item and then by time step
      (e.g. X1[0], X2[0], ..., X1[1], x2[1], ...). If false, the opposite will
      occur (e.g. X1[0], X1[1], ..., x2[0], x2[1], ...).
    unused_kwargs: Additional parameters should not be used but are included to
      make calling the function with hyper-parameters easier.

  Returns:
    A tuple of (train tf.data.Dataset, val tf.data.Dataset, test
    tf.data.Dataset). Each dataset yields a (time_series_input, static_input,
    labels) tuple per example containing data for one item at one timestep,
    where time_series_input is a tensor of shape (len_max_lookback,
    num_features), static_input is a tensor of shape (num_static,), and labels
    is a tensor of shape (forecast_horizon,). All shape values are represented
    in the dataset_params dictionary.
  """
  del unused_kwargs  # Unused but makes passing in hyper-parameters easier.

  if shuffle_after_windowing is None:
    shuffle_after_windowing = shuffle_train_items

  if max_train_examples is None:
    max_train_examples = -1

  # Data as numpy objects
  time_sequences = input_tables["time_sequences"]
  static = input_tables["static"]

  num_items = time_sequences.shape[0]
  if num_items != static.shape[0]:
    raise ValueError(
        "The first dimension of time_sequences and static data must match")

  # Training dataset preparation

  def split_tensors(x):
    time_series_features = x[:-forecast_horizon, :-num_static]
    static_features = x[0, -num_static:]
    labels = x[-forecast_horizon:, target_index]
    return (time_series_features, static_features, labels)

  input_sequence_train = time_sequences[:, train_start:train_end + 1]
  input_static_train = np.broadcast_to(
      np.expand_dims(static, axis=1),
      (static.shape[0], input_sequence_train.shape[1], static.shape[1]))
  input_train = np.concatenate([input_sequence_train, input_static_train],
                               axis=-1)
  train_dataset = tf.data.Dataset.from_tensor_slices(input_train)
  if shuffle_train_items:
    train_dataset = train_dataset.shuffle(
        1000, seed=42, reshuffle_each_iteration=True)

  windowed_dataset_num_items = num_items if cycle_items_first else 1
  train_dataset = create_windowed_dataset(
      train_dataset,
      len_max_lookback=len_max_lookback,
      forecast_horizon=forecast_horizon,
      num_items=1,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  train_dataset = train_dataset.map(split_tensors)
  if shuffle_after_windowing:
    train_dataset = train_dataset.shuffle(
        1000, seed=42, reshuffle_each_iteration=True)
  train_dataset = train_dataset.take(max_train_examples)

  del input_sequence_train, input_static_train
  gc.collect()

  # Validation dataset preparation

  # Note that val_start can be smaller than train_end.
  # Indeed, choosing val_start = train_end - len_max_lookback - 1 would yield
  # that the last prediction date of training is followed by the first
  # prediction date of validation.

  input_sequence_valid = time_sequences[:, val_start:val_end + 1]
  input_static_valid = np.broadcast_to(
      np.expand_dims(static, axis=1),
      (static.shape[0], input_sequence_valid.shape[1], static.shape[1]))
  input_valid = np.concatenate([input_sequence_valid, input_static_valid],
                               axis=-1)
  valid_dataset = tf.data.Dataset.from_tensor_slices(input_valid)

  valid_dataset = create_windowed_dataset(
      valid_dataset,
      len_max_lookback=len_max_lookback,
      forecast_horizon=forecast_horizon,
      num_items=windowed_dataset_num_items,
  )
  valid_dataset = valid_dataset.map(split_tensors)

  del input_sequence_valid, input_static_valid
  gc.collect()

  # Testing dataset preparation

  # Note that test_start can be smaller than val_end.
  # Indeed, choosing test_start = val_end - len_max_lookback - 1 would yield
  # that the last prediction date of validation is followed by the first
  # prediction date of test.

  input_sequence_test = time_sequences[:, test_start:test_end + 1]
  input_static_test = np.broadcast_to(
      np.expand_dims(static, axis=1),
      (static.shape[0], input_sequence_test.shape[1], static.shape[1]))
  input_test = np.concatenate([input_sequence_test, input_static_test], axis=-1)
  test_dataset = tf.data.Dataset.from_tensor_slices(input_test)
  test_dataset = create_windowed_dataset(
      test_dataset,
      len_max_lookback=len_max_lookback,
      forecast_horizon=forecast_horizon,
      num_items=windowed_dataset_num_items,
  )
  test_dataset = test_dataset.map(split_tensors)

  del input_sequence_test, input_static_test
  gc.collect()

  return train_dataset, valid_dataset, test_dataset

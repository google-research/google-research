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

"""A base timeseries forecasting model to help reduce copy paste code."""

import abc
import logging
import time

from models import losses

import numpy as np
import tensorflow as tf


class ForecastModel(abc.ABC):
  """Base time series forecasting model."""

  def __init__(self, hparams):

    self.batch_size = hparams["batch_size"]
    self.display_iterations = hparams["display_iterations"]
    self.forecast_horizon = hparams["forecast_horizon"]
    self.num_iterations = hparams["iterations"]
    self.num_encode = hparams["num_encode"]
    self.num_features = hparams["num_features"]
    self.num_test_splits = hparams["num_test_splits"]
    self.num_val_splits = hparams["num_val_splits"]
    self.static_index_cutoff = hparams["static_index_cutoff"]
    self.target_index = hparams["target_index"]
    self.temporal_batch_size_eval = hparams["temporal_batch_size_eval"]

    self.future_features_train = tf.zeros(
        [self.batch_size, self.forecast_horizon, 1])
    self.future_features_eval = tf.zeros(
        [self.temporal_batch_size_eval, self.forecast_horizon, 1])

  @abc.abstractmethod
  def train_step(self, input_sequence, input_static,
                 target):
    pass

  @abc.abstractmethod
  def test_step(self, input_sequence,
                input_static):
    pass

  @tf.function
  def val_step(self, input_sequence,
               input_static):
    # By default call the test step for validation.
    return self.test_step(input_sequence, input_static)

  def run_train_eval_pipeline(self, batched_train_dataset,
                              batched_valid_dataset, batched_test_dataset):
    """Runs training and testing for batched time-series data."""

    val_mae = []
    val_mape = []
    val_wmape = []
    val_mse = []
    val_q_score = []
    test_mae = []
    test_mape = []
    test_wmape = []
    test_mse = []
    test_q_score = []
    train_losses = []
    display_iterations = []

    val_mae_per_split = np.zeros(self.num_val_splits)
    test_mae_per_split = np.zeros(self.num_test_splits)
    val_mape_per_split = np.zeros(self.num_val_splits)
    test_mape_per_split = np.zeros(self.num_test_splits)
    val_wmape_den_per_split = np.zeros(self.num_val_splits)
    test_wmape_den_per_split = np.zeros(self.num_test_splits)
    val_mse_per_split = np.zeros(self.num_val_splits)
    test_mse_per_split = np.zeros(self.num_test_splits)
    val_q_score_den_per_split = np.zeros(self.num_val_splits)
    test_q_score_den_per_split = np.zeros(self.num_test_splits)

    total_train_time = 0
    for iteration in range(self.num_iterations):

      if (iteration % self.display_iterations == 0 or
          iteration == self.num_iterations - 1):

        # Validation stage

        for split_ind in range(self.num_val_splits):
          (input_sequence_valid_batch, input_static_valid_batch,
           target_valid_batch) = batched_valid_dataset.get_next()

          # Do not use the last two static features, as they are for
          # unnormalizing data.
          output_shift = tf.expand_dims(input_static_valid_batch[:, -2], -1)
          output_scale = tf.expand_dims(input_static_valid_batch[:, -1], -1)
          input_static_valid_batch = (
              input_static_valid_batch[:, :-self.static_index_cutoff])

          # Slice the encoding window
          input_sequence_valid_batch = input_sequence_valid_batch[:, -self.
                                                                  num_encode:, :]

          valid_predictions = self.val_step(input_sequence_valid_batch,
                                            input_static_valid_batch)

          # Apply denormalization
          valid_predictions = (valid_predictions * output_scale + output_shift)
          target_valid_batch = (
              target_valid_batch * output_scale + output_shift)

          val_mae_per_split[split_ind] = losses.mae_per_batch(
              valid_predictions, target_valid_batch)
          val_mape_per_split[split_ind] = losses.mape_per_batch(
              valid_predictions, target_valid_batch)
          val_wmape_den_per_split[split_ind] = tf.reduce_mean(
              target_valid_batch)
          val_mse_per_split[split_ind] = losses.mse_per_batch(
              valid_predictions, target_valid_batch)
          val_q_score_den_per_split[
              split_ind] = losses.q_score_denominator_per_batch(
                  target_valid_batch)

        val_mae.append(np.mean(val_mae_per_split))
        val_mape.append(np.mean(val_mape_per_split))
        val_wmape.append(100 * np.mean(val_mae_per_split) /
                         np.mean(val_wmape_den_per_split))
        val_mse.append(np.mean(val_mse_per_split))
        val_q_score.append(
            np.mean(val_mae_per_split) / np.mean(val_q_score_den_per_split))

        # Test stage

        for split_ind in range(self.num_test_splits):
          (input_sequence_test_batch, input_static_test_batch,
           target_test_batch) = batched_test_dataset.get_next()

          # Do not use the last two static features, as they are for
          # unnormalizing data.
          output_shift = tf.expand_dims(input_static_test_batch[:, -2], -1)
          output_scale = tf.expand_dims(input_static_test_batch[:, -1], -1)
          input_static_test_batch = input_static_test_batch[:, :-self.
                                                            static_index_cutoff]

          # Slice the encoding window
          input_sequence_test_batch = input_sequence_test_batch[:, -self
                                                                .num_encode:, :]

          test_predictions = self.test_step(input_sequence_test_batch,
                                            input_static_test_batch)

          # Apply denormalization
          test_predictions = (test_predictions * output_scale + output_shift)
          target_test_batch = (target_test_batch * output_scale + output_shift)

          test_mae_per_split[split_ind] = losses.mae_per_batch(
              test_predictions, target_test_batch)
          test_mape_per_split[split_ind] = losses.mape_per_batch(
              test_predictions, target_test_batch)
          test_wmape_den_per_split[split_ind] = tf.reduce_mean(
              target_test_batch)
          test_mse_per_split[split_ind] = losses.mse_per_batch(
              test_predictions, target_test_batch)
          test_q_score_den_per_split[
              split_ind] = losses.q_score_denominator_per_batch(
                  target_test_batch)

        test_mae.append(np.mean(test_mae_per_split))
        test_mape.append(np.mean(test_mape_per_split))
        test_wmape.append(100 * np.mean(test_mae_per_split) /
                          np.mean(test_wmape_den_per_split))
        test_mse.append(np.mean(test_mse_per_split))
        test_q_score.append(
            np.mean(val_mae_per_split) / np.mean(test_q_score_den_per_split))

        display_iterations.append(iteration)

        # Early stopping condition is defined as no improvement on the
        # validation scores between consecutive self.display_iterations
        # iterations.

        if len(val_mae) > 1 and val_mae[-2] < val_mae[-1]:
          break

      # Training stage
      t = time.perf_counter()
      (input_sequence_train_batch, input_static_train_batch,
       target_train_batch) = batched_train_dataset.get_next()

      # Do not use the last two static features, as they are for unnormalizing
      # data.
      input_static_train_batch = input_static_train_batch[:, :-self
                                                          .static_index_cutoff]

      # Slice the encoding window
      input_sequence_train_batch = input_sequence_train_batch[:, -self
                                                              .num_encode:, :]

      train_loss = self.train_step(input_sequence_train_batch,
                                   input_static_train_batch, target_train_batch)
      train_losses.append(train_loss)

      step_time = time.perf_counter() - t
      if iteration > 0:
        total_train_time += step_time
      if (iteration % self.display_iterations == 0 or
          iteration == self.num_iterations - 1):
        logging.debug("Iteration %d took %0.3g seconds (ave %0.3g)", iteration,
                      step_time, total_train_time / max(iteration, 1))

    evaluation_metrics = {
        "train_losses": train_losses,
        "display_iterations": display_iterations,
        "val_mae": val_mae,
        "val_mape": val_mape,
        "val_wmape": val_wmape,
        "val_mse": val_mse,
        "val_q_score": val_q_score,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "test_wmape": test_wmape,
        "test_mse": test_mse,
        "test_q_score": test_q_score
    }

    return evaluation_metrics

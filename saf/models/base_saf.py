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

"""Forecasting model that adapts during training and testing."""
import logging
import time
from typing import List, Tuple

from models import base
from models import losses
from models import model_utils
import numpy as np
import tensorflow as tf


#   That change should allow us to keep track of the self-adaptation loss
#   without having to change the class methods.
class ForecastModel(base.ForecastModel):
  """Base class for the self-adapting models.

  Note that this class is very similar to the base model except that it resets
  weights and outputs the self-adaptation loss from each step.
  """

  def __init__(self, loss_object, self_supervised_loss_object, hparams):
    super().__init__(hparams)

    self.reset_weights_each_eval_step: bool = hparams[
        "reset_weights_each_eval_step"]

    self.loss_object = loss_object
    self.self_supervised_loss_object = self_supervised_loss_object

    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams["learning_rate"])
    self.optimizer_adaptation = tf.keras.optimizers.SGD(
        learning_rate=hparams["learning_rate_adaptation"])

    # We must reset the layers so we need to record them all here.
    self._keras_layers: List[tf.keras.layers.Layer] = []

  @tf.function
  def train_step(self, input_sequence, input_static,
                 target):
    """Training step that outputs prediction and self-adaptation losses.

    Args:
      input_sequence: The input time-series data. Of size (batch_size,
        num_encode, num_features).
      input_static: The static inputs for each example. Of size (batch_size,
        num_static - static_index_cutoff).
      target: The target output of size (batch_size, forecast_horizon)

    Returns:
      The prediction loss and the self-adaptation loss.
    """
    raise NotImplementedError("Implement in sub-class.")

  @tf.function
  def val_step(self, input_sequence,
               input_static):
    """Validation step that outputs predictions and the self-adaptation loss.

    The default behavior is just to return the output of the test_step.

    Args:
      input_sequence: The input time-series data. Of size
        (temporal_batch_size_eval, num_encode, num_features).
      input_static: The static inputs for each example. Of size
        (temporal_batch_size_eval, num_static - static_index_cutoff).

    Returns:
      The predictions and the self-adaptation loss.
    """
    # The same as the test step for now.
    return self.test_step(input_sequence, input_static)

  @tf.function
  def test_step(self, input_sequence,
                input_static):
    """Test step that outputs predictions and the self-adaptation loss.

    Args:
      input_sequence: The input time-series data. Of size
        (temporal_batch_size_eval, num_encode, num_features).
      input_static: The static inputs for each example. Of size
        (temporal_batch_size_eval, num_static - static_index_cutoff).

    Returns:
      The predictions and the self-adaptation loss.
    """
    raise NotImplementedError("Implement in sub-class.")

  def run_train_eval_pipeline(self, batched_train_dataset,
                              batched_valid_dataset, batched_test_dataset):
    """Runs training and testing for novel self-supervised meta learning approach."""

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
    val_self_adaptation = []
    test_self_adaptation = []
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
    val_self_adaptation_per_split = np.zeros(self.num_val_splits)
    test_self_adaptation_per_split = np.zeros(self.num_test_splits)
    val_q_score_den_per_split = np.zeros(self.num_val_splits)
    test_q_score_den_per_split = np.zeros(self.num_test_splits)

    total_train_time = 0
    for iteration in range(self.num_iterations):

      if (iteration % self.display_iterations == 0 or
          iteration == self.num_iterations - 1):

        # Validation stage
        with model_utils.temporary_weights(
            *self._keras_layers) as reset_weights_fn:
          for split_ind in range(self.num_val_splits):
            if self.reset_weights_each_eval_step:
              reset_weights_fn()
            (input_sequence_valid_batch, input_static_valid_batch,
             target_valid_batch) = batched_valid_dataset.get_next()

            # Do not use the last two static features, as they are for
            # unnormalizing data.
            output_shift = tf.expand_dims(input_static_valid_batch[:, -2], -1)
            output_scale = tf.expand_dims(input_static_valid_batch[:, -1], -1)
            input_static_valid_batch = input_static_valid_batch[:, :-self.
                                                                static_index_cutoff]

            # Slice the encoding window
            input_sequence_valid_batch = input_sequence_valid_batch[:, -self.
                                                                    num_encode:, :]

            valid_predictions, self_adaptation_loss = self.val_step(
                input_sequence_valid_batch, input_static_valid_batch)

            # Apply denormalization
            valid_predictions = (
                valid_predictions * output_scale + output_shift)
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
            val_self_adaptation_per_split[split_ind] = self_adaptation_loss

        val_mae.append(np.mean(val_mae_per_split))
        val_mape.append(np.mean(val_mape_per_split))
        val_wmape.append(
            np.mean(100 * val_mae_per_split) / np.mean(val_wmape_den_per_split))
        val_mse.append(np.mean(val_mse_per_split))
        val_q_score.append(
            np.mean(val_mae_per_split) / np.mean(val_q_score_den_per_split))
        val_self_adaptation.append(np.mean(val_self_adaptation_per_split))

        # Test stage
        with model_utils.temporary_weights(
            *self._keras_layers) as reset_weights_fn:
          for split_ind in range(self.num_test_splits):
            if self.reset_weights_each_eval_step:
              reset_weights_fn()
            (input_sequence_test_batch, input_static_test_batch,
             target_test_batch) = batched_test_dataset.get_next()

            # Do not use the last two static features, as they are for
            # unnormalizing data.
            output_shift = tf.expand_dims(input_static_test_batch[:, -2], -1)
            output_scale = tf.expand_dims(input_static_test_batch[:, -1], -1)
            input_static_test_batch = input_static_test_batch[:, :-self.
                                                              static_index_cutoff]

            # Slice the encoding window
            input_sequence_test_batch = input_sequence_test_batch[:, -self.
                                                                  num_encode:, :]

            test_predictions, self_adaptation_loss = self.test_step(
                input_sequence_test_batch, input_static_test_batch)

            # Apply denormalization
            test_predictions = (test_predictions * output_scale + output_shift)
            target_test_batch = (
                target_test_batch * output_scale + output_shift)

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
            test_self_adaptation_per_split[split_ind] = self_adaptation_loss

          test_mae.append(np.mean(test_mae_per_split))
          test_mape.append(np.mean(test_mape_per_split))
          test_wmape.append(
              np.mean(100 * test_mae_per_split) /
              np.mean(test_wmape_den_per_split))
          test_mse.append(np.mean(test_mse_per_split))
          test_q_score.append(
              np.mean(val_mae_per_split) / np.mean(test_q_score_den_per_split))
          test_self_adaptation.append(np.mean(test_self_adaptation_per_split))

          display_iterations.append(iteration)

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

      train_loss, _ = self.train_step(input_sequence_train_batch,
                                      input_static_train_batch,
                                      target_train_batch)
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
        "val_self_adaptation": val_self_adaptation,
        "test_self_adaptation": test_self_adaptation,
        "test_q_score": test_q_score
    }

    return evaluation_metrics

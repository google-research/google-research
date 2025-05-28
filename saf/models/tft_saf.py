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

"""A TFT forecast model that self-adapts and uses backcast errors as features."""

from models import base_saf
from models import tft_layers

import tensorflow as tf


class ForecastModel(base_saf.ForecastModel):
  """TFT that uses backcast errors as feature."""

  def __init__(self,
               loss_object,
               self_supervised_loss_object,
               hparams,
               quantile_targets=(0.5,)):
    # For now we will include all of the errors as features.

    self.use_backcast_errors = hparams["use_backcast_errors"]
    if self.use_backcast_errors:
      self.num_error_features = hparams["num_features"]
    else:
      self.num_error_features = 0

    if "num_historical_features" not in hparams:
      hparams["num_historical_features"] = (
          hparams["num_features"] + self.num_error_features)
    hparams["num_future_features"] = 1
    hparams["num_static_features"] = hparams["num_static"] - hparams[
        "static_index_cutoff"]

    super().__init__(loss_object, self_supervised_loss_object, hparams)

    # Model layers
    self.quantile_targets = list(quantile_targets)
    tft_architecture = tft_layers.TFTModel(hparams, self.quantile_targets)
    self.tft_model = tft_architecture.return_self_adapting_model()
    self._keras_layers = [self.tft_model, self.optimizer_adaptation]

  @tf.function
  def _self_adaptation_step(self,
                            input_sequence,
                            input_static,
                            is_training=False):
    # We mask half of the input window, replacing the observed values with the
    # repeatedly applied value of the first value after the mask. The
    # objective for self-adaptation is proposed as reconstruction of the
    # entire window. Without masking the reconstruction is trivial by copying
    # the input, however, with masking, the model needs to learn the structure
    # of the data for accurate backcasts.

    repeated_mid_sequence = tf.tile(
        tf.expand_dims(input_sequence[:, self.num_encode // 2, :], 1),
        [1, self.num_encode // 2, 1])
    padded_input_sequence = tf.concat(
        [repeated_mid_sequence, input_sequence[:, self.num_encode // 2:, :]],
        axis=1)

    with tf.GradientTape() as tape:
      future_features = (
          self.future_features_train
          if is_training else self.future_features_eval)

      if self.use_backcast_errors:
        augmented_input_sequence = tf.concat(
            (padded_input_sequence, tf.zeros_like(input_sequence)), axis=2)
      else:
        augmented_input_sequence = padded_input_sequence
      backcasts, _ = self.tft_model.call(
          inputs=[augmented_input_sequence, future_features, input_static],
          training=is_training)
      if self.use_backcast_errors:
        # Remove the forecasts of the error features.
        backcasts = backcasts[:, :, :-self.num_error_features]

      self_adaptation_loss = self.self_supervised_loss_object(
          input_sequence, backcasts)

    adaptation_trainable_variables = self.tft_model.trainable_weights

    gradients = tape.gradient(self_adaptation_loss,
                              adaptation_trainable_variables)

    self.optimizer_adaptation.apply_gradients(
        zip(gradients, adaptation_trainable_variables))

    updated_backcasts, _ = self.tft_model.call(
        inputs=[augmented_input_sequence, future_features, input_static],
        training=is_training)

    if self.use_backcast_errors:
      # Remove the forecasts of the error features.
      updated_backcasts = updated_backcasts[:, :, :-self.num_error_features]

    backcast_errors = (input_sequence - updated_backcasts)

    return self_adaptation_loss, backcast_errors

  @tf.function
  def train_step(self, input_sequence, input_static, target):
    self_adaptation_loss, backcast_errors = self._self_adaptation_step(
        input_sequence, input_static, is_training=True)

    if self.use_backcast_errors:
      input_sequence = tf.concat((input_sequence, backcast_errors), axis=2)

    with tf.GradientTape() as tape:
      _, predictions = self.tft_model.call(
          inputs=[input_sequence, self.future_features_train, input_static],
          training=True)
      prediction_loss = self.loss_object(target, predictions[:, :, 0])

    all_trainable_weights = self.tft_model.trainable_weights
    gradients = tape.gradient(prediction_loss, all_trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, all_trainable_weights))
    return prediction_loss, self_adaptation_loss

  @tf.function
  def test_step(self, input_sequence, input_static):
    self_adaptation_loss, backcast_errors = self._self_adaptation_step(
        input_sequence, input_static, is_training=False)

    if self.use_backcast_errors:
      input_sequence = tf.concat((input_sequence, backcast_errors), axis=2)

    _, predictions = self.tft_model.call(
        inputs=[input_sequence, self.future_features_eval, input_static],
        training=False)
    return predictions[:, :, 0], self_adaptation_loss

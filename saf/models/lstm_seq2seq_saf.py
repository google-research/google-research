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

"""Forecast model class that contains the functions for training and evaluation for non-stationary time-series modeling."""

from models import architectures
from models import base_saf
import tensorflow as tf


class ForecastModel(base_saf.ForecastModel):
  """Self adapting model that uses a shared LSTM encoder."""

  def __init__(self, loss_object, self_supervised_loss_object, hparams):
    super().__init__(loss_object, self_supervised_loss_object, hparams)

    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams["learning_rate"])
    self.optimizer_adaptation = tf.keras.optimizers.SGD(
        learning_rate=hparams["learning_rate_adaptation"])

    # Model layers
    self.use_backcast_errors = hparams["use_backcast_errors"]
    self.encoder_architecture = architectures.LSTMEncoder(
        hparams, return_state=True)
    self.backcast_architecture = architectures.LSTMBackcast(hparams)
    self.forecast_architecture = architectures.LSTMDecoder(hparams)

    self._keras_layers = [
        self.backcast_architecture,
        self.forecast_architecture,
        self.encoder_architecture,
        self.optimizer_adaptation,
    ]

  @tf.function
  def _self_adaptation_step(self,
                            input_sequence,
                            input_static,
                            is_training=False):
    del is_training  # unused
    with tf.GradientTape() as tape:

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

      if self.use_backcast_errors:
        augmented_input_sequence = tf.concat(
            (padded_input_sequence, tf.zeros_like(input_sequence)), axis=2)
      else:
        augmented_input_sequence = padded_input_sequence

      encoded_representation, encoder_states = self.encoder_architecture.forward(
          augmented_input_sequence, input_static)
      reconstructed = self.backcast_architecture.forward(
          encoded_representation, input_static, tf.zeros_like(input_sequence),
          encoder_states)

      loss = self.self_supervised_loss_object(input_sequence, reconstructed)

    adaptation_trainable_variables = (
        self.encoder_architecture.weights + self.backcast_architecture.weights)

    gradients = tape.gradient(loss, adaptation_trainable_variables)

    self.optimizer_adaptation.apply_gradients(
        zip(gradients, adaptation_trainable_variables))

    encoded_representation, encoder_states = self.encoder_architecture.forward(
        augmented_input_sequence, input_static)
    reconstructed = self.backcast_architecture.forward(
        encoded_representation, input_static, tf.zeros_like(input_sequence),
        encoder_states)

    self_adaptation_loss = self.self_supervised_loss_object(
        input_sequence, reconstructed)

    backcast_errors = (input_sequence - reconstructed)

    return self_adaptation_loss, backcast_errors

  @tf.function
  def train_step(self, input_sequence, input_static, target):
    self_adaptation_loss, backcast_errors = self._self_adaptation_step(
        input_sequence, input_static, is_training=False)

    if self.use_backcast_errors:
      input_sequence = tf.concat((input_sequence, backcast_errors), axis=2)

    with tf.GradientTape() as tape:
      encoded_representation, encoder_states = self.encoder_architecture.forward(
          input_sequence, input_static)
      predictions = self.forecast_architecture.forward(
          encoded_representation, input_static, self.future_features_train,
          encoder_states)
      prediction_loss = self.loss_object(target, predictions)

    all_trainable_weights = (
        self.encoder_architecture.trainable_weights +
        self.forecast_architecture.trainable_weights)
    gradients = tape.gradient(prediction_loss, all_trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, all_trainable_weights))
    return prediction_loss, self_adaptation_loss

  @tf.function
  def test_step(self, input_sequence, input_static):
    self_adaptation_loss, backcast_errors = self._self_adaptation_step(
        input_sequence, input_static, is_training=False)

    if self.use_backcast_errors:
      input_sequence = tf.concat((input_sequence, backcast_errors), axis=2)

    encoded_representation, encoder_states = self.encoder_architecture.forward(
        input_sequence, input_static)
    predictions = self.forecast_architecture.forward(encoded_representation,
                                                     input_static,
                                                     self.future_features_eval,
                                                     encoder_states)
    return predictions, self_adaptation_loss

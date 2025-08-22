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

"""Time-series forecasting with encoding-decoding architecture."""

from models import architectures
from models import base

import tensorflow as tf


class ForecastModel(base.ForecastModel):
  """Baseline forecasting model based on LSTM seq2seq."""

  def __init__(self, loss_object, hparams):
    super().__init__(hparams)

    self.loss_object = loss_object
    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams["learning_rate"])

    # Model layers
    self.encoder_architecture = architectures.LSTMEncoder(
        hparams, return_state=True)
    self.forecast_architecture = architectures.LSTMDecoder(hparams)

  @tf.function
  def train_step(self, input_sequence, input_static, target):

    with tf.GradientTape() as tape:
      encoded_representation, encoder_states = self.encoder_architecture.forward(
          input_sequence, input_static)
      predictions = self.forecast_architecture.forward(
          encoded_representation, input_static, self.future_features_train,
          encoder_states)
      loss = self.loss_object(target, predictions)
    all_trainable_weights = (
        self.encoder_architecture.trainable_weights +
        self.forecast_architecture.trainable_weights)
    gradients = tape.gradient(loss, all_trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, all_trainable_weights))
    return loss

  @tf.function
  def test_step(self, input_sequence, input_static):

    encoded_representation, encoder_states = self.encoder_architecture.forward(
        input_sequence, input_static)
    predictions = self.forecast_architecture.forward(encoded_representation,
                                                     input_static,
                                                     self.future_features_eval,
                                                     encoder_states)
    return predictions

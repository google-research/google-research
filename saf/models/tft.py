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

"""Forecast model class that contains the functions for training.

and evaluation for Temporal Fusion Transformers (TFT) model:
https://arxiv.org/pdf/1912.09363.pdf.
"""
from models import base
from models import losses
from models import tft_layers
import tensorflow as tf


class ForecastModel(base.ForecastModel):
  """Forecast model."""

  # pylint: disable=dangerous-default-value
  def __init__(self, loss_object, hparams, quantile_targets=[0.5]):
    super().__init__(hparams)

    self.future_features_train = tf.zeros(
        [self.batch_size, self.forecast_horizon, 1])
    self.future_features_eval = tf.zeros(
        [hparams["temporal_batch_size_eval"], self.forecast_horizon, 1])

    hparams["num_future_features"] = 1
    hparams["num_historical_features"] = hparams["num_features"]
    hparams["num_static_features"] = hparams["num_static"] - hparams[
        "static_index_cutoff"]

    self.loss_object = loss_object
    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams["learning_rate"])

    # Model layers
    self.quantile_targets = quantile_targets
    tft_architecture = tft_layers.TFTModel(hparams, self.quantile_targets)
    self.tft_model = tft_architecture.return_baseline_model()

  @tf.function
  def train_step(self, input_sequence, input_static, target):

    with tf.GradientTape() as tape:
      predictions = self.tft_model.call(
          inputs=[input_sequence, self.future_features_train, input_static],
          training=True)
      if len(self.quantile_targets) == 1:
        loss = self.loss_object(target, predictions[:, :, 0])
      else:
        loss = losses.quantile_loss(target, predictions, self.quantile_targets)

    all_trainable_weights = (self.tft_model.trainable_weights)
    gradients = tape.gradient(loss, all_trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, all_trainable_weights))
    return loss

  @tf.function
  def test_step(self, input_sequence, input_static):

    predictions = self.tft_model.call(
        inputs=[input_sequence, self.future_features_eval, input_static],
        training=False)

    if len(self.quantile_targets) == 1:
      return predictions[:, :, 0]
    else:
      return predictions[:, :, (len(self.quantile_targets) - 1) // 2]

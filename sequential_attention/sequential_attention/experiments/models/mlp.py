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

"""Sparse MLP Model."""

import tensorflow as tf


class MLPModel(tf.keras.Model):
  """MLP model."""

  def __init__(
      self,
      layer_sequence=None,
      is_classification=True,
      num_classes=None,
      learning_rate=0.001,
      decay_steps=500,
      decay_rate=0.8,
      alpha=0,
      batch_norm=True,
  ):
    """Initialize the model."""

    super().__init__()

    if batch_norm:
      self.batch_norm_layer = tf.keras.layers.BatchNormalization()
    self.batch_norm = batch_norm

    mlp_sequence = [
        tf.keras.layers.Dense(
            dim, activation=tf.keras.layers.LeakyReLU(alpha=alpha)
        )
        for dim in layer_sequence
    ]
    self.mlp_model = tf.keras.Sequential(mlp_sequence)
    if is_classification:
      self.mlp_predictor = tf.keras.layers.Dense(
          num_classes, activation="softmax", dtype="float32"
      )
    else:
      self.mlp_predictor = tf.keras.layers.Dense(1, dtype="float32")

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  def call(self, inputs, training=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    inputs = tf.multiply(inputs, self.selected_features)
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction

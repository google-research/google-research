# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""Model definition."""

import tensorflow as tf


def build_model(image_size, bias_last=True, num_classes=1, squeeze=True):
  """Builds model."""

  input_shape = (image_size, image_size, 3)
  image = tf.keras.Input(shape=input_shape, name='input_image')
  training = tf.keras.Input(shape=[], name='training')

  x = tf.keras.layers.Conv2D(
      128, (3, 3), strides=(1, 1), padding='valid', activation=None)(
          image)
  x = tf.keras.layers.BatchNormalization()(x, training)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.Conv2D(
      128, (3, 3), strides=(2, 2), padding='valid', activation=None)(
          x)
  x = tf.keras.layers.BatchNormalization()(x, training)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.Conv2D(
      256, (3, 3), strides=(2, 2), padding='valid', activation=None)(
          x)
  x = tf.keras.layers.BatchNormalization()(x, training)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.Conv2D(
      256, (3, 3), strides=(2, 2), padding='valid', activation=None)(
          x)
  x = tf.keras.layers.BatchNormalization()(x, training)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.Conv2D(
      512, (1, 1), strides=(1, 1), padding='valid', activation=None)(
          x)
  x = tf.keras.layers.BatchNormalization()(x, training)
  x = tf.keras.layers.ReLU()(x)
  # x = tf.keras.layers.Conv2D(64, (2, 2), padding='valid')(x)
  x = tf.keras.layers.Flatten()(x)

  last_layer_fc = tf.keras.layers.Dense(num_classes, use_bias=bias_last)

  if squeeze:
    x = tf.squeeze(last_layer_fc(x))
  else:
    x = last_layer_fc(x)

  model = tf.keras.models.Model(
      inputs=[image, training], outputs=x, name='model')
  model.summary()
  return model

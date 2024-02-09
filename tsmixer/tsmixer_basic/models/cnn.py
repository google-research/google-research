# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Implementation of CNN for forecasting."""

import tensorflow as tf


class Model(tf.keras.Model):
  """CNN model."""

  def __init__(self, n_channel, pred_len, kernel_size):
    super().__init__()
    self.cnn = tf.keras.layers.Conv1D(
        n_channel, kernel_size, padding='same', input_shape=(None, n_channel)
    )
    self.dense = tf.keras.layers.Dense(pred_len)

  def call(self, x):
    # x: [Batch, Input length, Channel]
    x = self.cnn(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = self.dense(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    return x  # [Batch, Output length, Channel]

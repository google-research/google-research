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

"""Temperature scaling.
"""

import tensorflow as tf


class TempCalibratedModel(tf.keras.Model):
  """Apply temperature scaling to trained models."""

  def __init__(self, model):
    super(TempCalibratedModel, self).__init__()
    self.model = model
    self.model.trainable = False
    self.temp = tf.Variable(initial_value=1, trainable=True, dtype=tf.float32)

  def call(self, inputs, training=None, mask=None, **kwargs):
    logits = self.model(inputs, training=training, mask=mask, **kwargs)
    scaled_logits = tf.math.divide(logits, self.temp)
    return scaled_logits


class TempCalibratedCBMModel(tf.keras.Model):
  """Apply temperature scaling to trained models."""

  def __init__(self, model):
    super(TempCalibratedCBMModel, self).__init__()
    self.model = model
    self.model.trainable = False
    self.temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

  def call(self, inputs, training=None, mask=None, **kwargs):
    c_output, y_logits = self.model(
        inputs, training=training, mask=mask, **kwargs
    )
    scaled_y_logits = tf.math.divide(y_logits, self.temp)
    return c_output, scaled_y_logits

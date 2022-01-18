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

"""Tests model architecture functions."""

import tensorflow as tf

from poem.cv_mim.action_recognition import models


class ModelsTest(tf.test.TestCase):

  def test_build_residual_block(self):
    input_features = tf.zeros([4, 128, 16], tf.float32)
    input_layer = tf.keras.Input(input_features.shape[1:])
    output_layer = models.build_residual_block(input_layer, 64, 2)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    outputs = model(input_features)
    self.assertAllEqual(outputs.shape, [4, 64, 64])

  def test_build_residual_temporal_model(self):
    input_features = tf.zeros([4, 128, 16], tf.float32)
    model = models.build_residual_temporal_model(
        input_shape=(128, 16), num_classes=4)

    outputs = model(input_features)
    self.assertAllEqual(outputs.shape, [4, 4])

  def test_build_simple_temporal_model(self):
    input_features = tf.zeros([4, 128, 16], tf.float32)
    model = models.build_simple_temporal_model(
        input_shape=(128, 16), num_classes=4)

    outputs = model(input_features)
    self.assertAllEqual(outputs.shape, [4, 4])

  def test_build_residual_temporal_convolutional_model(self):
    input_features = tf.zeros([4, 128, 16], tf.float32)
    model = models.build_residual_temporal_convolutional_model(
        input_shape=(128, 16), num_classes=4)

    outputs = model(input_features)
    self.assertAllEqual(outputs.shape, [4, 4])


if __name__ == '__main__':
  tf.test.main()

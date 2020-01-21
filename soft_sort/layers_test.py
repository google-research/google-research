# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python3
"""Tests for soft sorting tensorflow layers."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from soft_sort import layers


class LayersTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    self._input_shape = (1, 10, 2)
    self._axis = 1
    self._inputs = tf.random.normal(self._input_shape)

  @parameterized.parameters([6, 3])
  def test_soft_topk_layer(self, topk):
    direction = 'DESCENDING'
    layer = layers.SoftSortLayer(
        axis=self._axis, topk=topk, direction=direction, epsilon=1e-3)
    outputs = layer(self._inputs)
    expected_shape = list(self._input_shape)
    expected_shape[self._axis] = topk
    self.assertAllEqual(outputs.shape, expected_shape)
    sorted_inputs = tf.sort(self._inputs, axis=self._axis, direction=direction)
    self.assertAllClose(sorted_inputs[:, :topk, :], outputs, atol=1e-2)

  def test_softsortlayer(self):
    direction = 'DESCENDING'
    layer = layers.SoftSortLayer(
        axis=self._axis, direction=direction, epsilon=1e-3)
    outputs = layer(self._inputs)
    self.assertAllEqual(outputs.shape, self._inputs.shape)
    sorted_inputs = tf.sort(self._inputs, axis=self._axis, direction=direction)
    self.assertAllClose(sorted_inputs, outputs, atol=1e-2)

  def take_model_output(self, layer, inputs):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=inputs[0].shape),
        layer,
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    model.build(inputs.shape)
    model.compile(tf.keras.optimizers.SGD(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model(inputs)

  @parameterized.parameters([None, 4])
  def test_sortlayer_in_model(self, topk):
    inputs = tf.random.uniform((32, 10))
    outputs = self.take_model_output(layers.SoftSortLayer(topk=topk), inputs)
    self.assertAllEqual([inputs.shape[0], 1], outputs.shape)

  def test_rankslayer_in_model(self):
    inputs = tf.random.uniform((32, 10))
    outputs = self.take_model_output(layers.SoftRanksLayer(), inputs)
    self.assertAllEqual([inputs.shape[0], 1], outputs.shape)

  def test_quantilelayer_in_model(self):
    inputs = tf.random.uniform((32, 10))
    outputs = self.take_model_output(
        layers.SoftQuantilesLayer(
            quantiles=[0.2, 0.5, 0.8], output_shape=(32, 3)),
        inputs)
    self.assertAllEqual([inputs.shape[0], 1], outputs.shape)

  def test_softranks(self):
    layer = layers.SoftRanksLayer(axis=self._axis, epsilon=1e-4)
    outputs = layer(self._inputs)
    self.assertAllEqual(outputs.shape, self._inputs.shape)
    ranks = tf.argsort(
        tf.argsort(self._inputs, axis=self._axis), axis=self._axis)
    self.assertAllClose(ranks, outputs, atol=0.5)

  def test_softquantiles(self):
    inputs = tf.reshape(tf.range(101, dtype=tf.float32), (1, -1))
    axis = 1
    quantiles = [0.25, 0.50, 0.75]
    layer = layers.SoftQuantilesLayer(
        quantiles=quantiles, output_shape=None, axis=axis, epsilon=1e-3)

    outputs = layer(inputs)
    self.assertAllEqual(outputs.shape, (1, 3))

    self.assertAllClose(tf.constant([[25., 50., 75.]]), outputs, atol=0.5)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for helper layers."""

import tensorflow as tf

from readtwice import layers as readtwice_layers


class LayersTest(tf.test.TestCase):

  def test_dense_layers(self):
    inputs = tf.constant([[0.5, -1.5], [1.0, 0.5]])

    layers1 = readtwice_layers.DenseLayers(
        hidden_sizes=[2], activation='relu', kernel_initializer='ones')
    layers1_output = layers1(inputs)

    layers2 = readtwice_layers.DenseLayers(
        hidden_sizes=[3, 1], activation='relu', kernel_initializer='ones')
    layers2_output = layers2(inputs)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose([[-1.0, -1.0], [1.5, 1.5]], layers1_output)
    self.assertAllClose([[0.0], [4.5]], layers2_output)

  def test_dense_layers_3d_input(self):
    inputs = tf.constant([
        [[0.5, -1.5], [-0.5, -2.0]],  #
        [[1.0, 0.5], [0.0, 2.0]]
    ])

    layers = readtwice_layers.DenseLayers(
        hidden_sizes=[2], activation='relu', kernel_initializer='ones')
    result = layers(inputs)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose(
        [
            [[-1.0, -1.0], [-2.5, -2.5]],  #
            [[1.5, 1.5], [2.0, 2.0]]
        ],
        result)

  def test_dense_layers_custom_activation_function(self):
    inputs = tf.constant([[0.5, -1.5], [1.0, 0.5]])

    layers = readtwice_layers.DenseLayers(
        hidden_sizes=[3, 1],
        activation=lambda x: -tf.nn.relu(x),
        kernel_initializer='ones')
    result = layers(inputs)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose([[0.0], [-4.5]], result)

  def test_dense_layers_invalid_hidden_sizes(self):
    with self.assertRaises(ValueError):
      readtwice_layers.DenseLayers(hidden_sizes=[3, 0, 1])

    with self.assertRaises(ValueError):
      readtwice_layers.DenseLayers(hidden_sizes=[-1])

  def test_tracked_lambda(self):
    inputs = tf.constant([[0.5, -1.5], [1.0, 0.5]])

    scaling_variable = tf.Variable(initial_value=10.0)
    linear_layer = readtwice_layers.DenseLayers(
        hidden_sizes=[2], activation='relu', kernel_initializer='ones')

    def function(inputs, return_zeros=False):
      if return_zeros:
        return tf.zeros_like(inputs)
      else:
        return scaling_variable * linear_layer(inputs)

    lambda_layer = readtwice_layers.TrackedLambda(
        function, dependencies=[scaling_variable, linear_layer])

    result = lambda_layer(inputs)
    zero_result = lambda_layer(inputs, return_zeros=True)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose([[-10.0, -10.0], [15.0, 15.0]], result)
    self.assertAllClose([[0.0, 0.0], [0.0, 0.0]], zero_result)

    # Check variable tracking
    self.assertLen(lambda_layer.variables, 3)
    self.assertIs(scaling_variable, lambda_layer.variables[0])
    self.assertEqual(linear_layer.variables, lambda_layer.variables[1:])


if __name__ == '__main__':
  tf.test.main()

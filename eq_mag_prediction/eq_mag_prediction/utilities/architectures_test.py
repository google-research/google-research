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

"""Tests for architectures."""

import numpy as np
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import architectures


class ArchitecturesTest(parameterized.TestCase):

  def test_same_output_size_cnn_model(self):
    name = 'vi'
    input_shape = (13, 37, 123)
    filter_sizes = (7, 5, 5)
    filter_numbers = (8, 6, 4)
    activation = 'tanh'

    model = architectures.same_output_size_cnn_model(
        name, input_shape, filter_sizes, filter_numbers, activation
    )

    self.assertTupleEqual(model.layers[0].input_shape[0], (None, *input_shape))
    self.assertTupleEqual(
        tuple(layer.filters for layer in model.layers[1:]), filter_numbers
    )
    self.assertTupleEqual(
        tuple(layer.kernel_size for layer in model.layers[1:]),
        tuple((f, f) for f in filter_sizes),
    )
    for i, layer in enumerate(model.layers[1:]):
      self.assertEqual(layer.activation.__name__, activation)
      self.assertEqual(layer.name, f'{name}_{i}_cnn')
      self.assertEqual(layer.output.shape[1:-1], input_shape[:2])

  def test_same_output_size_3d_cnn_model(self):
    name = 'vi'
    input_shape = (13, 37, 8, 123)
    filter_shapes = ((7, 7, 3), (5, 5, 2), (3, 3, 1))
    filter_numbers = (8, 6, 4)
    activation = 'tanh'

    model = architectures.same_output_size_3d_cnn_model(
        name, input_shape, filter_shapes, filter_numbers, activation
    )

    self.assertTupleEqual(model.layers[0].input_shape[0], (None, *input_shape))
    self.assertTupleEqual(
        tuple(layer.filters for layer in model.layers[1:-1]), filter_numbers
    )
    self.assertTupleEqual(
        tuple(layer.kernel_size for layer in model.layers[1:-1]), filter_shapes
    )
    for i, layer in enumerate(model.layers[1:-1]):
      self.assertEqual(layer.activation.__name__, activation)
      self.assertEqual(layer.name, f'{name}_{i}_cnn')
      self.assertEqual(layer.output.shape[1:-1], input_shape[:3])
    self.assertEqual(model.output.shape[1:], (13, 37, 8 * filter_numbers[-1]))

  def test_spatial_order_invariant_cnn_model(self):
    name = 'powder'
    input_shape = (13, 37, 88, 222)
    filter_numbers = (8, 6, 4)
    activation = 'tanh'

    model = architectures.spatial_order_invariant_cnn_model(
        name, input_shape, filter_numbers, activation
    )

    self.assertTupleEqual(model.layers[0].input_shape[0], (None, *input_shape))
    for i, layer in enumerate(model.layers[1:-1]):
      self.assertEqual(layer.filters, filter_numbers[i])
      self.assertTupleEqual(layer.kernel_size, (1, 1, 1))
      self.assertEqual(layer.activation.__name__, activation)
      self.assertEqual(layer.name, f'{name}_{i}_cnn')
      self.assertEqual(layer.output.shape[1:-1], input_shape[:3])

  def test_cnn_1d_model(self):
    cnn_filters = (4, 8, 15)
    dense_units = (16, 23, 42)
    input_shape = (13, 37)
    name = 'moo'

    model = architectures.cnn_1d_model(
        name, input_shape, cnn_filters, dense_units
    )

    self.assertTupleEqual(model.layers[0].input_shape[0], (None, *input_shape))
    self.assertTupleEqual(
        tuple(
            layer.filters for layer in model.layers[2::2][: len(cnn_filters)]
        ),
        cnn_filters,
    )
    self.assertTupleEqual(
        tuple(
            layer.kernel_size
            for layer in model.layers[2::2][: len(cnn_filters)]
        ),
        tuple((1, f) for f in [input_shape[-1]] + list(cnn_filters)[:-1]),
    )
    self.assertTupleEqual(
        tuple(layer.units for layer in model.layers[-len(dense_units) :]),
        dense_units,
    )

  def test_fully_connected_model(self):
    layer_sizes = (4, 8, 15, 16, 23, 42)
    input_size = 100
    name = 'moo'

    model = architectures.fully_connected_model(name, input_size, layer_sizes)

    self.assertTupleEqual(
        tuple(layer.units for layer in model.layers[1:]), layer_sizes
    )
    self.assertTupleEqual(model.layers[0].input_shape[0], (None, input_size))
    self.assertEqual(model.layers[0].name, name)

  @parameterized.parameters(None, 'sigmoid')
  def test_multi_layer_perceptron(self, output_activation):
    n_filters = [2, 12, 34, 5]
    input_shape = (50,)
    activation = 'tanh'
    kernel_regularization = tf.keras.regularizers.l1_l2()
    model_name = 'testing'

    model = architectures.multi_layer_perceptron(
        n_filters,
        input_shape=input_shape,
        activation=activation,
        model_name=model_name,
        kernel_regularization=kernel_regularization,
        output_activation=output_activation,
    )

    for n, layer in zip(n_filters, model.layers[1:]):
      self.assertEqual(layer.output_shape, (None, n))
      self.assertEqual(layer.kernel_regularizer, kernel_regularization)
      self.assertTrue(layer.name.startswith(model_name + '_'))
      if layer == model.layers[-1]:
        if output_activation is None:
          self.assertEqual(layer.activation.__name__, 'linear')
        else:
          self.assertEqual(layer.activation.__name__, output_activation)
      else:
        self.assertEqual(layer.activation.__name__, activation)
    self.assertLen(model.input_names, 1)
    self.assertEqual(model.input_names[0], model_name)

  def test_identity_model(self):
    batch_size, input_size = 64, 100
    name = 'moo'
    data = np.random.random((batch_size, input_size))

    model = architectures.identity_model(name, input_size)

    self.assertEqual(model.input.name, name)
    np.testing.assert_array_almost_equal(model.predict(data), data)

  def test_cnn_model(self):
    filters = [16, 8]
    kernels = [(5, 5), (3, 3)]
    input_shape = (10, 9, 8)
    name = 'moo'

    model = architectures.cnn_model(name, input_shape, filters, kernels)

    self.assertEqual(
        model.layers[0].input_shape[0], (None, *input_shape)
    )
    self.assertEqual(model.layers[0].name, name)
    for i, layer in enumerate(model.layers[1:-1]):
      self.assertEqual(layer.filters, filters[i])
      self.assertEqual(layer.kernel_size, kernels[i])
      self.assertEqual(layer.name, f'{name}_{i}_cnn')
    self.assertLen(model.output.shape, 2)

  @parameterized.parameters('LSTM', 'GRU')
  def test_rnn_model(self, rnn_layer_type):
    units = [13, 37]
    input_features = 42
    name = 'moo'

    model = architectures.rnn_model(
        name, input_features, units, rnn_layer_type=rnn_layer_type
    )

    self.assertEqual(
        model.layers[0].input_shape[0], (None, None, input_features)
    )
    self.assertEqual(model.layers[0].name, name)
    for i, layer in enumerate(model.layers[1:]):
      if i == len(units) - 1:
        self.assertEqual(layer.units, units[i])
        self.assertEqual(layer.name, f'{name}_{i}_{rnn_layer_type}')
      else:
        self.assertEqual(layer.forward_layer.units, units[i])
        self.assertEqual(
            layer.forward_layer.name, f'forward_{name}_{i}_{rnn_layer_type}'
        )


if __name__ == '__main__':
  absltest.main()

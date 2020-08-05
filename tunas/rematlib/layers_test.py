# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python2, python3
"""Unit tests for layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tunas.rematlib import layers


class ScalarMultiplicationLayer(layers.Layer):

  def __init__(self, initial_value, regularizer=None, name=None):
    super(ScalarMultiplicationLayer, self).__init__()
    self._initial_value = initial_value
    self._regularizer = regularizer
    self._name = name
    self._built = False

  def build(self, input_shape):
    with tf.variable_scope(self._name, 'ScalarMultiplicationLayer') as scope:
      self._scope = scope
      if not self._built:
        self._create_trainable_variable(
            name='scalar',
            initializer=self._initial_value,
            regularizer=self._regularizer)

      self._built = True
      return input_shape

  def apply(self, inputs, training):
    del training
    assert self._built
    with tf.variable_scope(self._scope, reuse=True):
      return self._get_trainable_tensor('scalar') * inputs


class Constant(layers.Layer):

  def __init__(self, value):
    self._value = tf.constant(value, tf.float32)

  def build(self, input_shape):
    return self._value.shape

  def apply(self, inputs, training):
    del inputs, training
    return self._value


class LayersTest(tf.test.TestCase):

  def test_with_data_dependencies(self):
    var1 = tf.get_variable(
        name='var1',
        initializer=0,
        dtype=tf.int32,
        use_resource=True)
    with tf.control_dependencies([var1.assign_add(1)]):
      increment_var1 = var1.read_value()

    var2 = tf.get_variable(
        name='var2',
        initializer=[0, 0],
        dtype=tf.int32,
        use_resource=True)
    with tf.control_dependencies([var2.assign_add([1, 1])]):
      increment_var2 = var2.read_value()

    var3 = tf.get_variable(
        name='var3',
        initializer=[[0, 0], [0, 0], [0, 0]],
        dtype=tf.int32,
        use_resource=True)
    with tf.control_dependencies([var3.assign_add([[1, 1], [1, 1], [1, 1]])]):
      increment_var3 = var3.read_value()

    output1 = tf.constant(2.0)
    output2 = tf.constant([3.0, 4.0])
    output3 = tf.constant([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

    tensors = layers.with_data_dependencies(
        [increment_var1, increment_var2, increment_var3],
        [output1, output2, output3])

    self.evaluate(tf.global_variables_initializer())

    # Verify that the output values are correct.
    arrays = self.evaluate(tensors)
    self.assertAllClose(arrays, [
        2.0,
        [3.0, 4.0],
        [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
    ])

    # Verify that the dependencies are evaluated.
    self.assertAllClose(self.evaluate(var1), 1)
    self.assertAllClose(self.evaluate(var2), [1, 1])
    self.assertAllClose(self.evaluate(var3), [[1, 1], [1, 1], [1, 1]])

  def test_with_data_dependencies_grads(self):
    tensor1 = tf.constant(1.0)
    tensor2 = tf.constant(2.0)
    outputs = layers.with_data_dependencies([tensor1], [5.0 * tensor2])

    self.assertLen(outputs, 1)
    grads = tf.gradients(outputs[0], [tensor1, tensor2])

    self.assertLen(grads, 2)
    self.assertIsNone(grads[0])
    self.assertAllClose(self.evaluate(grads[1]), 5.0)

  def test_layer_regularization_loss(self):
    initial_value = 3.0
    l2_weight = 5.0
    layer = ScalarMultiplicationLayer(
        initial_value=initial_value,
        regularizer=tf.keras.regularizers.l2(l2_weight))

    inputs = tf.constant(10.0)
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        l2_weight * initial_value**2,
        self.evaluate(layer.regularization_loss()))

  def test_regularization_loss_for_layer_without_variables(self):
    layer = layers.Identity()

    inputs = tf.constant([1.0, -2.0, 3.0])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    self.assertAllClose(0, self.evaluate(layer.regularization_loss()))

  def test_merge_shapes_with_broadcast(self):
    self.assertEqual(
        layers.merge_shapes_with_broadcast(None, None),
        tf.TensorShape(None))
    self.assertEqual(
        layers.merge_shapes_with_broadcast(None, [1, 3]),
        tf.TensorShape([1, 3]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([8, 1], None),
        tf.TensorShape([8, 1]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([8, 1], []),
        tf.TensorShape([8, 1]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([], [1, 3]),
        tf.TensorShape([1, 3]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([None], [1]),
        tf.TensorShape([1]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([1], [None]),
        tf.TensorShape([1]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([None], [2]),
        tf.TensorShape([2]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([2], [None]),
        tf.TensorShape([2]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([1], [1]),
        tf.TensorShape([1]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([1], [2]),
        tf.TensorShape([2]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([2], [1]),
        tf.TensorShape([2]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast([2], [2]),
        tf.TensorShape([2]))
    self.assertEqual(
        layers.merge_shapes_with_broadcast(
            [2, None, 8, 1, 32],
            [None, 4, 8, 16, 32]),
        tf.TensorShape([2, 4, 8, 16, 32]))

    with self.assertRaisesRegex(ValueError,
                                'Tensor shapes must have the same rank'):
      layers.merge_shapes_with_broadcast([1, 1], [1])

    with self.assertRaisesRegex(ValueError, 'Tensor shapes are not compatible'):
      layers.merge_shapes_with_broadcast([2], [3])

  def test_identity(self):
    layer = layers.Identity()
    inputs = tf.constant([1.0, -2.0, 3.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), [1.0, -2.0, 3.0])
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_zeros(self):
    layer = layers.Zeros()
    inputs = tf.constant([1.0, -2.0, 3.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), [0.0, 0.0, 0.0])
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_zeros_with_output_shape(self):
    layer = layers.Zeros(output_shape=tf.TensorShape([1, 2]))
    inputs = tf.constant([[1.0, -2.0, 3.0]])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), [[0.0, 0.0]])
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_zeros_with_output_shape_and_unknown_batch_dim(self):
    layer = layers.Zeros(output_shape=tf.TensorShape([None, 2]))
    inputs = tf.constant([[1.0, -2.0, 3.0]])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), [[0.0, 0.0]])
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_relu(self):
    layer = layers.ReLU()
    inputs = tf.constant([1.0, -2.0, 3.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), [1.0, 0.0, 3.0])
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_relu6(self):
    layer = layers.ReLU6()
    inputs = tf.constant([1.0, -2.0, 3.0, 7.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), [1.0, 0.0, 3.0, 6.0])
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_sigmoid(self):
    layer = layers.Sigmoid()
    inputs = tf.constant([1.0, -2.0, 3.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    expected_output = tf.nn.sigmoid(inputs)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), self.evaluate(expected_output))
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_swish(self):
    layer = layers.Swish()
    inputs = tf.constant([1.0, -2.0, 3.0, 7.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    expected_output = tf.nn.swish(inputs)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(
        self.evaluate(output),
        self.evaluate(expected_output))
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_swish6(self):
    layer = layers.Swish6()
    inputs = tf.constant([1.0, -2.0, 3.0, 7.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    # Swish6(x) = x * relu6(x + 3) / 6
    relu6 = lambda x: max(0, min(x, 6))
    expected_output = [
        1 * relu6(1 + 3) / 6,
        -2 * relu6(-2 + 3) / 6,
        3 * relu6(3 + 3) / 6,
        7 * relu6(7 + 3) / 6,
    ]

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_elu(self):
    layer = layers.ELU()
    inputs = tf.constant([1.0, -2.0, 3.0])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    expected_output = tf.nn.elu(inputs)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), self.evaluate(expected_output))
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_space2depth(self):
    layer = layers.SpaceToDepth(block_size=2)
    inputs = tf.fill([1, 8, 8, 2], 1.0)
    expected_output = tf.fill([1, 4, 4, 8], 1.0)

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output),
                        self.evaluate(expected_output))
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_space2depth_error(self):
    layer = layers.SpaceToDepth(block_size=2)
    inputs = tf.fill([1, 5, 5, 2], 1.0)

    with self.assertRaisesRegex(ValueError,
                                'Image height 5 must be a multiple of 2'):
      layer.build(inputs.shape)

  def test_depth_padding(self):
    layer = layers.DepthPadding(filters=4)
    inputs = tf.fill([2, 8, 8, 2], 1.0)
    expected_output = np.concatenate(
        (np.ones((2, 8, 8, 2)),
         np.zeros((2, 8, 8, 2))),
        axis=3)

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output),
                        expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_depth_padding_wrong_filter(self):
    layer = layers.DepthPadding(filters=1)
    inputs = tf.fill([2, 8, 8, 2], 1.0)

    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Output filters is smaller than input filters.'):
      layer.build(inputs.shape)

  def test_max_pool(self):
    layer = layers.MaxPool(kernel_size=(2, 2), strides=2)
    inputs = tf.concat(
        [
            tf.fill([2, 2, 2, 2], 1.0),
            tf.fill([2, 2, 2, 2], 0.5),
        ],
        axis=3)
    first_row = np.ones((2, 1, 1, 2))
    second_row = np.empty((2, 1, 1, 2))
    second_row.fill(0.5)
    expected_output = np.concatenate(
        (first_row,
         second_row),
        axis=3)

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_max_pool_3x3_strides2(self):
    layer = layers.MaxPool(kernel_size=(3, 3), strides=2)
    inputs = tf.reshape(tf.range(36), [1, 6, 6, 1])
    expected_output = [[[[14], [16], [17]], [[26], [28], [29]],
                        [[32], [34], [35]]]]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_max_pool_3x3_strides2_explicit_padding(self):
    layer = layers.MaxPool(
        kernel_size=(3, 3), strides=2, use_explicit_padding=True)
    inputs = tf.reshape(tf.range(36), [1, 6, 6, 1])
    expected_output = [[[[7], [9], [11]], [[19], [21], [23]], [[31], [33],
                                                               [35]]]]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output),
                        expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_avg_pool(self):
    layer = layers.AveragePool(kernel_size=(2, 2), strides=2)
    inputs = tf.concat(
        [
            tf.fill([2, 2, 2, 2], 1.0),
            tf.fill([2, 2, 2, 2], 0.5),
        ],
        axis=3)
    first_row = np.ones((2, 1, 1, 2))
    second_row = np.empty((2, 1, 1, 2))
    second_row.fill(0.5)
    expected_output = np.concatenate(
        (first_row,
         second_row),
        axis=3)

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output),
                        expected_output.tolist())
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_global_average_pool_no_keepdims(self):
    layer = layers.GlobalAveragePool(keepdims=False)
    inputs = tf.concat(
        [
            tf.fill([2, 8, 8, 2], 1.0),
            tf.fill([2, 8, 8, 2], 2.0),
            tf.fill([2, 8, 8, 2], 3.0),
        ],
        axis=3)
    expected_output = [
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
    ]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_global_average_pool_keepdims_size_1(self):
    layer = layers.GlobalAveragePool(keepdims=True)
    inputs = tf.concat(
        [
            tf.fill([2, 1, 1, 2], 1.0),
            tf.fill([2, 1, 1, 2], 2.0),
            tf.fill([2, 1, 1, 2], 3.0),
        ],
        axis=3)
    expected_output = [
        [[[1, 1, 2, 2, 3, 3]]],
        [[[1, 1, 2, 2, 3, 3]]],
    ]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_global_average_pool_no_keepdims_size_1(self):
    layer = layers.GlobalAveragePool(keepdims=False)
    inputs = tf.concat(
        [
            tf.fill([2, 1, 1, 2], 1.0),
            tf.fill([2, 1, 1, 2], 2.0),
            tf.fill([2, 1, 1, 2], 3.0),
        ],
        axis=3)
    expected_output = [
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
    ]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertAllClose(self.evaluate(output), expected_output)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_global_average_pool_no_keepdims_dynamic_shape(self):
    layer = layers.GlobalAveragePool(keepdims=False)
    inputs = tf.placeholder(dtype=tf.float32, shape=[2, None, None, 6])

    inputs_value = np.concatenate(
        [
            np.full([2, 8, 8, 2], 1.0),
            np.full([2, 8, 8, 2], 2.0),
            np.full([2, 8, 8, 2], 3.0),
        ],
        axis=3)
    expected_output = [
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
    ]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output.shape, output_shape)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

    with self.session() as sess:
      self.assertAllClose(
          sess.run(output, {inputs: inputs_value}),
          expected_output)

  def test_global_average_pool_keepdims(self):
    layer = layers.GlobalAveragePool(keepdims=True)
    inputs = tf.concat(
        [
            tf.fill([2, 8, 8, 2], 1.0),
            tf.fill([2, 8, 8, 2], 2.0),
            tf.fill([2, 8, 8, 2], 3.0),
        ],
        axis=3)
    expected_output = [
        [[[1, 1, 2, 2, 3, 3]]],
        [[[1, 1, 2, 2, 3, 3]]],
    ]

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output_shape, output.shape)
    self.assertAllClose(expected_output, self.evaluate(output))
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_dropout_train(self):
    layer = layers.Dropout(0.5)

    inputs = tf.ones([32, 40])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output_shape.as_list(), [32, 40])
    self.assertEqual(output_shape, output.shape)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

    value = self.evaluate(output)
    self.assertGreater(np.sum(value), 0)
    self.assertTrue(np.any(np.equal(value, 0)))

    # We will drop out some of the values by multiplying by zero. We will scale
    # up the remaining by multiplying by 1/(1-rate) = 1/(1-0.5) = 2.
    is_zero_or_two = np.logical_or(np.equal(value, 0), np.equal(value, 2))
    self.assertTrue(np.all(is_zero_or_two))

  def test_dropout_eval(self):
    layer = layers.Dropout(0.5)

    inputs = tf.ones([32, 40])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=False)

    self.assertEqual(output_shape.as_list(), [32, 40])
    self.assertEqual(output_shape, output.shape)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

    value = self.evaluate(output)
    self.assertAllEqual(value, np.ones([32, 40]))

  def test_multiply_by_constant(self):
    layer = layers.MultiplyByConstant(6.0)

    inputs = tf.ones([2, 3])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)

    self.assertEqual(output_shape, output.shape)
    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

    value = self.evaluate(output)
    self.assertAllEqual(value, np.full([2, 3], 6.0))

  def test_switch_without_variables(self):
    inputs = tf.constant([1.0, 2.0, -3.0])
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    layer = layers.Switch(
        mask,
        [
            layers.Identity(),
            layers.ReLU()
        ])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    grad = tf.gradients(output, [inputs])[0]
    self.assertEqual(output_shape, output.shape)

    with self.cached_session() as sess:
      self.assertAllClose(sess.run(output, {mask: [0, 0]}), [0, 0, 0])
      self.assertAllClose(sess.run(output, {mask: [1, 0]}), [1, 2, -3])
      self.assertAllClose(sess.run(output, {mask: [0, 1]}), [1, 2, 0])
      self.assertAllClose(sess.run(output, {mask: [1, 1]}), [2, 4, -3])
      self.assertAllClose(sess.run(output, {mask: [.6, .4]}), [1, 2, -1.8])

      self.assertAllClose(sess.run(grad, {mask: [0, 0]}), [0, 0, 0])
      self.assertAllClose(sess.run(grad, {mask: [1, 0]}), [1, 1, 1])
      self.assertAllClose(sess.run(grad, {mask: [0, 1]}), [1, 1, 0])
      self.assertAllClose(sess.run(grad, {mask: [1, 1]}), [2, 2, 1])
      self.assertAllClose(sess.run(grad, {mask: [.6, .4]}), [1, 1, .6])

    self.assertEmpty(layer.trainable_tensors())
    self.assertEmpty(layer.trainable_variables())

  def test_switch_with_bfloat16_input_and_float32_mask(self):
    inputs = tf.constant([1.0, 2.0, -3.0], dtype=tf.bfloat16)
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    layer = layers.Switch(
        mask,
        [
            layers.Identity(),
            layers.ReLU()
        ])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)
    self.assertEqual(output.dtype, tf.bfloat16)

  def test_two_switches_with_different_variables(self):
    inputs = tf.constant([1.0, 2.0, 3.0])
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    lhs = ScalarMultiplicationLayer(3.0)
    rhs = ScalarMultiplicationLayer(4.0)
    layer = layers.Switch(mask, [lhs, rhs])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.assertLen(layer.trainable_tensors(), 2)
    grads = tf.gradients(output, [inputs] + layer.trainable_tensors())

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())

      self.assertAllClose(
          sess.run(output, {mask: [1, 0]}),
          [3, 6, 9])
      self.assertAllClose(
          sess.run(output, {mask: [0, 1]}),
          [4, 8, 12])
      self.assertAllClose(
          sess.run(output, {mask: [0.6, 0.4]}),
          [3.4, 6.8, 10.2])

      self.assertAllClose(
          sess.run(grads, {mask: [1, 0]}),
          [
              [3, 3, 3],        # grad w.r.t. input
              1 + 2 + 3,        # grad w.r.t. lhs.scalar
              0,                # grad w.r.t. rhs.scalar
          ])
      self.assertAllClose(
          sess.run(grads, {mask: [0, 1]}),
          [
              [4, 4, 4],        # grad w.r.t. input
              0,                # grad w.r.t. lhs.scalar
              1 + 2 + 3,        # grad w.r.t. rhs.scalar
          ])
      self.assertAllClose(
          sess.run(grads, {mask: [0.6, 0.4]}),
          [
              [3.4, 3.4, 3.4],  # grad w.r.t. input
              (1 + 2 + 3)*0.6,  # grad w.r.t. lhs.scalar
              (1 + 2 + 3)*0.4,  # grad w.r.t. rhs.scalar
          ])
      self.assertAllClose(
          sess.run(layer.trainable_tensors()),
          sess.run(layer.trainable_variables()))

  def test_same_switch_used_twice(self):
    inputs = tf.constant([1.0, 2.0, 3.0])
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    inner = ScalarMultiplicationLayer(3.0)
    layer = layers.Switch(mask, [inner, inner])

    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.assertLen(layer.trainable_tensors(), 1)
    grads = tf.gradients(output, [inputs] + layer.trainable_tensors())

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())

      self.assertAllClose(sess.run(output, {mask: [1, 0]}), [3, 6, 9])
      self.assertAllClose(sess.run(output, {mask: [0, 1]}), [3, 6, 9])

      self.assertAllClose(
          sess.run(grads, {mask: [0, 1]}),
          [
              [3, 3, 3],  # grad w.r.t. input
              1 + 2 + 3,  # grad w.r.t. inner.scalar
          ])
      self.assertAllClose(
          sess.run(grads, {mask: [1, 0]}),
          [
              [3, 3, 3],  # grad w.r.t. input
              1 + 2 + 3,  # grad w.r.t. inner.scalar
          ])
      self.assertAllClose(
          sess.run(grads, {mask: [0.6, 0.4]}),
          [
              [3, 3, 3],  # grad w.r.t. input
              1 + 2 + 3,  # grad w.r.t. inner.scalar
          ])
      self.assertAllClose(
          sess.run(grads, {mask: [1.2, 0.8]}),
          [
              [6, 6, 6],  # grad w.r.t. input
              2 + 4 + 6,  # grad w.r.t. inner.scalar
          ])

      self.assertAllClose(
          sess.run(layer.trainable_tensors()),
          sess.run(layer.trainable_variables()))

  def test_switch_regularization_loss(self):
    inputs = tf.constant([1.0, 2.0, 3.0])
    mask = tf.placeholder(shape=(3,), dtype=tf.float32)

    layer = layers.Switch(
        mask,
        [
            layers.Identity(),
            ScalarMultiplicationLayer(
                initial_value=8.0,
                regularizer=tf.keras.regularizers.l2(4.0)),
            ScalarMultiplicationLayer(
                initial_value=5.0,
                regularizer=tf.keras.regularizers.l2(3.0)),
        ])

    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    reg_loss = layer.regularization_loss()
    with self.session() as sess:
      reg_loss0 = 0.0
      reg_loss1 = 4.0 * 8.0**2
      reg_loss2 = 3.0 * 5.0**2
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(reg_loss0, sess.run(reg_loss, {mask: [1, 0, 0]}))
      self.assertAllClose(reg_loss1, sess.run(reg_loss, {mask: [0, 1, 0]}))
      self.assertAllClose(reg_loss2, sess.run(reg_loss, {mask: [0, 0, 1]}))
      self.assertAllClose(
          0.2*reg_loss0 + 0.3*reg_loss1 + 0.5*reg_loss2,
          sess.run(reg_loss, {mask: [0.2, 0.3, 0.5]}))

  def test_switch_regularization_loss_with_variable_reuse(self):
    inputs = tf.constant([1.0, 2.0, 3.0])
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    inner_layer = ScalarMultiplicationLayer(
        initial_value=5.0,
        regularizer=tf.keras.regularizers.l2(3.0))
    outer_layer = layers.Switch(mask, [inner_layer, inner_layer])

    outer_layer.build(inputs.shape)
    outer_layer.apply(inputs, training=True)

    reg_loss = outer_layer.regularization_loss()
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(3.0 * 5.0**2, sess.run(reg_loss, {mask: [1, 0]}))
      self.assertAllClose(3.0 * 5.0**2, sess.run(reg_loss, {mask: [0, 1]}))
      # NOTE: It's not clear what our behavior should be in this case.
      # Another option would be to multiply the regularizer by 2 (since it's
      # used in both branches. For now, we just document the current behavior.
      self.assertAllClose(3.0 * 5.0**2, sess.run(reg_loss, {mask: [1, 1]}))

  def test_sequential(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer2 = ScalarMultiplicationLayer(6.0)
    layer = layers.Sequential([layer0, layer1, layer2])

    inputs = tf.constant([1.0, -2.0, 3.0])
    output_shape = layer.build(inputs.shape)
    self.assertLen(layer.trainable_tensors(), 3)

    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    product = 4.0 * 5.0 * 6.0
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(output),
        [1.0*product, -2.0*product, 3.0*product])

    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_sequential_with_shared_layers(self):
    inner_layer = ScalarMultiplicationLayer(2.0)
    layer = layers.Sequential([inner_layer]*5)

    inputs = tf.constant([1.0, -2.0, 3.0])
    output_shape = layer.build(inputs.shape)
    self.assertLen(layer.trainable_tensors(), 1)

    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    product = 2.0 ** 5
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(output),
        [1.0*product, -2.0*product, 3.0*product])

    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_sequential_regularization_loss(self):
    layer = layers.Sequential([
        ScalarMultiplicationLayer(
            initial_value=2.0,
            regularizer=tf.keras.regularizers.l2(3.0)),
        ScalarMultiplicationLayer(
            initial_value=4.0,
            regularizer=tf.keras.regularizers.l2(5.0))
    ])

    inputs = tf.constant(10.0)
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        (3.0 * 2.0**2) + (5.0 * 4.0**2),
        self.evaluate(layer.regularization_loss()))

  def test_sequential_regularization_loss_with_reused_layer(self):
    # Verify that we only regularize a layer's variables once, even if we use
    # that layer more than once in a model.
    inner_layer = ScalarMultiplicationLayer(
        initial_value=2.0,
        regularizer=tf.keras.regularizers.l2(3.0))
    outer_layer = layers.Sequential([inner_layer, inner_layer])

    inputs = tf.constant(10.0)
    outer_layer.build(inputs.shape)
    outer_layer.apply(inputs, training=True)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        3.0 * 2.0**2,
        self.evaluate(outer_layer.regularization_loss()))

  def test_sequential_with_one_aux_output(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer2 = ScalarMultiplicationLayer(6.0)
    layer = layers.Sequential(
        [layer0, layer1, layer2],
        aux_outputs=[layer1])

    inputs = tf.constant([1.0])
    output_shape = layer.build(inputs.shape)
    self.assertLen(layer.trainable_tensors(), 3)

    output, aux_outputs = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    output_value, aux_output_values = self.evaluate([output, aux_outputs])

    self.assertAllEqual(output_value, [4.0*5.0*6.0])
    self.assertLen(aux_outputs, 1)
    self.assertAllEqual(aux_output_values[0], [4.0*5.0])

  def test_sequential_with_three_aux_outputs(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer2 = ScalarMultiplicationLayer(6.0)
    layer = layers.Sequential(
        [layer0, layer1, layer2],
        aux_outputs=[layer0, layer1, layer2])

    inputs = tf.constant([1.0])
    output_shape = layer.build(inputs.shape)
    self.assertLen(layer.trainable_tensors(), 3)

    output, aux_outputs = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    output_value, aux_output_values = self.evaluate([output, aux_outputs])

    self.assertAllEqual(output_value, [4.0*5.0*6.0])
    self.assertLen(aux_outputs, 3)
    self.assertAllEqual(aux_output_values[0], [4.0])
    self.assertAllEqual(aux_output_values[1], [4.0*5.0])
    self.assertAllEqual(aux_output_values[2], [4.0*5.0*6.0])

  def test_sequential_with_invalid_aux_output(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer2 = ScalarMultiplicationLayer(6.0)

    with self.assertRaisesRegex(
        ValueError, 'element of aux_outputs does not appear in layers'):
      layers.Sequential([layer0, layer1], aux_outputs=[layer2])

  def test_sequential_with_repeated_aux_outputs(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer2 = ScalarMultiplicationLayer(6.0)
    layer = layers.Sequential(
        [layer0, layer1, layer2],
        aux_outputs=[layer1, layer1])

    inputs = tf.constant([1.0])
    output_shape = layer.build(inputs.shape)
    self.assertLen(layer.trainable_tensors(), 3)

    output, aux_outputs = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    output_value, aux_output_values = self.evaluate([output, aux_outputs])

    self.assertAllEqual(output_value, [4.0*5.0*6.0])
    self.assertLen(aux_outputs, 2)
    self.assertAllEqual(aux_output_values[0], [4.0*5.0])
    self.assertAllEqual(aux_output_values[1], [4.0*5.0])

  def test_switch_applied_twice_with_different_batch_sizes(self):
    inputs1 = tf.constant([1.0, 2.0, -3.0])  # batch size = 3
    inputs2 = tf.constant([-10.0, 11.0])     # batch size = 2
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    layer = layers.Switch(mask, [layers.Identity(), layers.ReLU()])

    layer.build(inputs1.shape)
    output1 = layer.apply(inputs1, training=True)
    output2 = layer.apply(inputs2, training=True)

    with self.test_session() as sess:
      self.assertAllClose(
          [1.0, 2.0, -3.0],
          sess.run(output1, {mask: [1, 0]}))
      self.assertAllClose(
          [-10.0, 11.0],
          sess.run(output2, {mask: [1, 0]}))
      self.assertAllClose(
          [1.0, 2.0, 0],
          sess.run(output1, {mask: [0, 1]}))
      self.assertAllClose(
          [0, 11.0],
          sess.run(output2, {mask: [0, 1]}))

  def test_parallel_sum(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer2 = ScalarMultiplicationLayer(6.0)
    layer = layers.ParallelSum([layer0, layer1, layer2])

    inputs = tf.constant([1.0, -2.0, 3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    layer_sum = 4.0 + 5.0 + 6.0
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(output),
        [1.0*layer_sum, -2.0*layer_sum, 3.0*layer_sum])
    self.assertLen(layer.trainable_tensors(), 3)

    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_parallel_sum_with_shared_layers(self):
    inner_layer = ScalarMultiplicationLayer(2.0)
    layer = layers.ParallelSum([inner_layer]*5)

    inputs = tf.constant([1.0, -2.0, 3.0])

    output_shape = layer.build(inputs.shape)
    self.assertLen(layer.trainable_tensors(), 1)

    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    layer_sum = 2.0 * 5
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(output),
        [1.0*layer_sum, -2.0*layer_sum, 3.0*layer_sum])

    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_parallel_sum_regularization_loss(self):
    layer = layers.ParallelSum([
        ScalarMultiplicationLayer(
            initial_value=2.0,
            regularizer=tf.keras.regularizers.l2(3.0)),
        ScalarMultiplicationLayer(
            initial_value=4.0,
            regularizer=tf.keras.regularizers.l2(5.0))
    ])

    inputs = tf.constant(10.0)
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        (3.0 * 2.0**2) + (5.0 * 4.0**2),
        self.evaluate(layer.regularization_loss()))

  def test_parallel_sum_regularization_loss_with_reused_layer(self):
    # Verify that we only regularize a layer's variables once, even if we use
    # that layer more than once in a model.
    inner_layer = ScalarMultiplicationLayer(
        initial_value=2.0,
        regularizer=tf.keras.regularizers.l2(3.0))
    outer_layer = layers.ParallelSum([inner_layer, inner_layer])

    inputs = tf.constant(10.0)
    outer_layer.build(inputs.shape)
    outer_layer.apply(inputs, training=True)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        3.0 * 2.0**2,
        self.evaluate(outer_layer.regularization_loss()))

  def test_parallel_sum_with_broadcast_scalar(self):
    layer = layers.ParallelSum([layers.Identity(), Constant(4.0)])

    inputs = tf.constant([1.0, -2.0, 3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(output), [1.0+4.0, -2.0+4.0, 3.0+4.0])

  def test_parallel_product_with_variables(self):
    layer0 = ScalarMultiplicationLayer(4.0)
    layer1 = ScalarMultiplicationLayer(5.0)
    layer = layers.ParallelProduct([layer0, layer1])

    inputs = tf.constant([1.0, -2.0, 3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(output),
        [1.0*4.0 * 1.0*5.0, -2.0*4.0 * -2.0*5.0, 3.0*4.0 * 3.0*5.0])
    self.assertLen(layer.trainable_tensors(), 2)

    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_parallel_product_with_broadcast_scalar(self):
    layer = layers.ParallelProduct([layers.Identity(), Constant(4.0)])

    inputs = tf.constant([1.0, -2.0, 3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(output), [1.0*4.0, -2.0*4.0, 3.0*4.0])

  def test_sequential_switch_regularization_loss_with_weird_layer_reuse(self):
    # Document how regularization losses should be computed when layers are
    # reuse in weird ways.
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)

    # Weird behavior: `inner_layer` is used both inside and outside the Switch.
    inner_layer = ScalarMultiplicationLayer(
        initial_value=4.0,
        regularizer=tf.keras.regularizers.l2(5.0))
    outer_layer = layers.Sequential(
        [inner_layer,
         layers.Switch(mask, [layers.Zeros(), inner_layer])])

    inputs = tf.constant([1.0, -2.0, 3.0])
    outer_layer.build(inputs.shape)
    outer_layer.apply(inputs, training=True)

    reg_loss = outer_layer.regularization_loss()
    with self.session() as sess:
      # The ScalarMultiplicationLayer should be regularized regardless of which
      # option is selected, since it is used both inside and outside the Switch.
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(5.0 * 4.0**2, sess.run(reg_loss, {mask: [1, 0]}))
      self.assertAllClose(5.0 * 4.0**2, sess.run(reg_loss, {mask: [0, 1]}))

  def test_conv2d_3x3(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [6], [6], [6], [4]], [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]], [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]], [[4], [6], [6], [6], [6], [4]]]])

  def test_conv2d_3x3_explicit_padding(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        kernel_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [6], [6], [6], [4]],
          [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]],
          [[4], [6], [6], [6], [6], [4]]]])

  def test_conv2d_3x3_bias(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        kernel_initializer=tf.initializers.ones(),
        bias_initializer=tf.initializers.constant(0.5),
        use_bias=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4.5], [6.5], [6.5], [6.5], [6.5], [4.5]],
          [[6.5], [9.5], [9.5], [9.5], [9.5], [6.5]],
          [[6.5], [9.5], [9.5], [9.5], [9.5], [6.5]],
          [[6.5], [9.5], [9.5], [9.5], [9.5], [6.5]],
          [[6.5], [9.5], [9.5], [9.5], [9.5], [6.5]],
          [[4.5], [6.5], [6.5], [6.5], [6.5], [4.5]]]])

  def test_conv2d_3x3_stride2(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(2, 2),
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9], [9], [6]],
          [[9], [9], [6]],
          [[6], [6], [4]]]])

  def test_conv2d_3x3_stride2_explicit_padding(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(2, 2),
        kernel_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [6]], [[6], [9], [9]], [[6], [9], [9]]]])

  def test_conv2d_3x3_dilation2(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        dilation_rates=(2, 2),
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [4], [6], [6], [4], [4]], [[4], [4], [6], [6], [4], [4]],
          [[6], [6], [9], [9], [6], [6]], [[6], [6], [9], [9], [6], [6]],
          [[4], [4], [6], [6], [4], [4]], [[4], [4], [6], [6], [4], [4]]]])

  def test_conv2d_3x3_dilation2_explicit_padding(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        dilation_rates=(2, 2),
        kernel_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [4], [6], [6], [4], [4]],
          [[4], [4], [6], [6], [4], [4]],
          [[6], [6], [9], [9], [6], [6]],
          [[6], [6], [9], [9], [6], [6]],
          [[4], [4], [6], [6], [4], [4]],
          [[4], [4], [6], [6], [4], [4]]]])

  def test_conv2d_3x3_dilation2_stride2(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(2, 2),
        dilation_rates=(2, 2),
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [4]],
          [[6], [9], [6]],
          [[4], [6], [4]]]])

  def test_conv2d_3x3_dilation2_stride2_explicit_padding(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(2, 2),
        dilation_rates=(2, 2),
        kernel_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [4]],
          [[6], [9], [6]],
          [[4], [6], [4]]]])

  def test_conv2d_3x3_int_kernel_size_and_strides(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=2,
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9], [9], [6]],
          [[9], [9], [6]],
          [[6], [6], [4]]]])

  def test_conv2d_output_shape(self):
    for image_size in [1, 2, 3, 32, 201, 224]:
      inputs = tf.ones([32, image_size, image_size, 1])
      for kernel_size in [1, 2, 3, 4, 5]:
        for strides in [1, 2, 3]:
          layer = layers.Conv2D(
              filters=1,
              kernel_size=(kernel_size, kernel_size),
              strides=(strides, strides),
              kernel_initializer=tf.initializers.ones())
          output_shape = layer.build(inputs.shape)
          output = layer.apply(inputs, training=True)
          self.assertEqual(output.shape, output_shape)

  def test_conv2d_trainable_tensors(self):
    layer = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        kernel_initializer=tf.initializers.ones())

    input_shape = tf.TensorShape([1, 6, 6, 1])
    layer.build(input_shape)

    trainable_tensors = layer.trainable_tensors()
    self.assertNotEmpty(trainable_tensors)
    for tensor in trainable_tensors:
      self.assertIsInstance(tensor, tf.Tensor)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_conv2d_kernel_regularization_loss(self):
    layer = layers.Conv2D(
        filters=12,
        kernel_size=(3, 3),
        kernel_initializer=tf.initializers.constant(0.5),
        kernel_regularizer=tf.keras.regularizers.l2(3.0))

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    kernel_params = 3 * 3 * 8 * 12
    self.assertAllClose(
        kernel_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_conv2d_bias_regularization_loss(self):
    layer = layers.Conv2D(
        filters=12,
        kernel_size=(3, 3),
        bias_initializer=tf.initializers.constant(0.5),
        bias_regularizer=tf.keras.regularizers.l2(3.0),
        use_bias=True)

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    bias_params = 12
    self.assertAllClose(
        bias_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_depthwise_conv2d_3x3(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [6], [6], [6], [4]], [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]], [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]], [[4], [6], [6], [6], [6], [4]]]])

  def test_depthwise_conv2d_3x3_explicit_padding(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depthwise_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [6], [6], [6], [4]],
          [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]],
          [[6], [9], [9], [9], [9], [6]],
          [[4], [6], [6], [6], [6], [4]]]])

  def test_depthwise_conv2d_3x3_stride2(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9], [9], [6]],
          [[9], [9], [6]],
          [[6], [6], [4]]]])

  def test_depthwise_conv2d_3x3_stride2_explicit_padding(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        depthwise_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [6], [6]],
          [[6], [9], [9]],
          [[6], [9], [9]]]])

  def test_depthwise_conv2d_3x3_dilation2(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=1,
        dilation_rates=2,
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [4], [6], [6], [4], [4]], [[4], [4], [6], [6], [4], [4]],
          [[6], [6], [9], [9], [6], [6]], [[6], [6], [9], [9], [6], [6]],
          [[4], [4], [6], [6], [4], [4]], [[4], [4], [6], [6], [4], [4]]]])

  def test_depthwise_conv2d_3x3_dilation2_explicit_padding(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=1,
        dilation_rates=2,
        depthwise_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4], [4], [6], [6], [4], [4]],
          [[4], [4], [6], [6], [4], [4]],
          [[6], [6], [9], [9], [6], [6]],
          [[6], [6], [9], [9], [6], [6]],
          [[4], [4], [6], [6], [4], [4]],
          [[4], [4], [6], [6], [4], [4]]]])

  def test_depthwise_conv2d_3x3_dilation2_stride2_error(self):
    with self.assertRaises(ValueError):
      _ = layers.DepthwiseConv2D(
          kernel_size=(3, 3),
          strides=2,
          dilation_rates=2,
          depthwise_initializer=tf.initializers.ones())

  def test_depthwise_conv2d_3x3_stride2_int_kernel_size_and_strides(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 1])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9], [9], [6]],
          [[9], [9], [6]],
          [[6], [6], [4]]]])

  def test_depthwise_conv2d_output_shape(self):
    for image_size in [1, 2, 3, 32, 201, 224]:
      inputs = tf.ones([32, image_size, image_size, 1])
      for kernel_size in [1, 2, 3, 4, 5]:
        for strides in [1, 2, 3]:
          layer = layers.DepthwiseConv2D(
              kernel_size=(kernel_size, kernel_size),
              strides=(strides, strides),
              depthwise_initializer=tf.initializers.ones())
          output_shape = layer.build(inputs.shape)
          output = layer.apply(inputs, training=True)
          self.assertEqual(output.shape, output_shape)

  def test_depthwise_conv2d_trainable_tensors(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depthwise_initializer=tf.initializers.ones())

    input_shape = tf.TensorShape([1, 6, 6, 1])
    layer.build(input_shape)

    trainable_tensors = layer.trainable_tensors()
    self.assertNotEmpty(trainable_tensors)
    for tensor in trainable_tensors:
      self.assertIsInstance(tensor, tf.Tensor)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_depthwise_conv2d_kernel_regularization_loss(self):
    layer = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        depthwise_initializer=tf.initializers.constant(0.5),
        depthwise_regularizer=tf.keras.regularizers.l2(3.0))

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    kernel_params = 3 * 3 * 8 * 1
    self.assertAllClose(
        kernel_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_stateless_batch_norm(self):
    layer = layers.BatchNorm(
        center=True,
        scale=True,
        beta_initializer=tf.initializers.zeros(),
        gamma_initializer=tf.initializers.ones(),
        epsilon=1e-12,
        stateful=False)

    inputs = tf.random_uniform([32, 28, 28, 16])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output.shape, output_shape)

    mean, std = tf.nn.moments(output, axes=[0, 1, 2])
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(mean),
        np.zeros([16]),
        rtol=1e-3,
        atol=1e-3)
    self.assertAllClose(
        self.evaluate(std),
        np.ones([16]),
        rtol=1e-3,
        atol=1e-3)

    self.assertLen(layer.trainable_tensors(), 2)
    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_stateful_batch_norm(self):
    layer = layers.BatchNorm(
        momentum=0.0,
        center=True,
        scale=True,
        beta_initializer=tf.initializers.zeros(),
        gamma_initializer=tf.initializers.ones(),
        epsilon=1e-12,
        stateful=True)

    inputs_bias = tf.placeholder(dtype=tf.float32, shape=())
    inputs = tf.random_normal([32, 28, 28, 16]) + inputs_bias
    output_shape = layer.build(inputs.shape)
    train_output = layer.apply(inputs, training=True)
    eval_output = layer.apply(inputs, training=False)
    self.assertEqual(train_output.shape, output_shape)
    self.assertEqual(eval_output.shape, output_shape)

    update_ops = layer.updates()
    self.assertLen(update_ops, 2)
    self.assertCountEqual(
        update_ops,
        tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())

      # After initialization, moving average will be 0s and moving variance will
      # be 1s. Evaluating with training=False on any input should return similar
      # input values (also assuming gamma=1 and beta=0).
      inputs_array, eval_array = sess.run([inputs, eval_output],
                                          {inputs_bias: 5.0})
      self.assertAllClose(eval_array, inputs_array, atol=0.0001)

      # Since the batch norm momentum is 0, we'll set the moving average
      # statistics for the batch norm equal to the statistics for the current
      # batch.
      sess.run(update_ops, {inputs_bias: 2.0})

      # Evaluate a batch of input examples with the same input distribution
      # that was seen during training.
      eval_array = sess.run(eval_output, {inputs_bias: 2.0})
      self.assertAllClose(
          np.mean(eval_array, axis=(0, 1, 2)),
          np.zeros([16]),
          rtol=0.1,
          atol=0.1)
      self.assertAllClose(
          np.std(eval_array, axis=(0, 1, 2)),
          np.ones([16]),
          rtol=0.1,
          atol=0.1)

      # Verify that the batch norm op is actually stateful and running in eval
      # mode by changing the mean of the input distribution and verifying that
      # the mean of the output distribution also changes.
      eval_array = sess.run(eval_output, {inputs_bias: 4.0})
      self.assertAllClose(
          np.mean(eval_array, axis=(0, 1, 2)),
          np.full([16], 2.0),
          rtol=0.1,
          atol=0.1)

  def test_masked_conv2d_3x3(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=layers.create_mask([1, 2], 0),
        output_mask=layers.create_mask([1, 2], 0),
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]]]])

  def test_masked_conv2d_3x3_explicit_padding(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=layers.create_mask([1, 2], 0),
        output_mask=layers.create_mask([1, 2], 0),
        kernel_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]]]])

  def test_masked_conv2d_3x3_stride2_explicit_padding(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        input_mask=layers.create_mask([1, 2], 0),
        output_mask=layers.create_mask([1, 2], 0),
        kernel_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4, 0], [6, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0]],
          [[6, 0], [9, 0], [9, 0]]]])

  def test_masked_conv2d_3x3_with_none_input_mask(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=None,
        output_mask=layers.create_mask([1, 2], 0),
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[8, 0], [12, 0], [12, 0], [12, 0], [12, 0], [8, 0]],
          [[12, 0], [18, 0], [18, 0], [18, 0], [18, 0], [12, 0]],
          [[12, 0], [18, 0], [18, 0], [18, 0], [18, 0], [12, 0]],
          [[12, 0], [18, 0], [18, 0], [18, 0], [18, 0], [12, 0]],
          [[12, 0], [18, 0], [18, 0], [18, 0], [18, 0], [12, 0]],
          [[8, 0], [12, 0], [12, 0], [12, 0], [12, 0], [8, 0]]]])

  def test_masked_conv2d_3x3_bias(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=layers.create_mask([1, 2], 0),
        output_mask=layers.create_mask([1, 2], 0),
        kernel_initializer=tf.initializers.ones(),
        bias_initializer=tf.initializers.constant(0.5),
        use_bias=True)

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4.5, 0], [6.5, 0], [6.5, 0], [6.5, 0], [6.5, 0], [4.5, 0]],
          [[6.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [6.5, 0]],
          [[6.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [6.5, 0]],
          [[6.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [6.5, 0]],
          [[6.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [9.5, 0], [6.5, 0]],
          [[4.5, 0], [6.5, 0], [6.5, 0], [6.5, 0], [6.5, 0], [4.5, 0]]]])

  def test_masked_conv2d_3x3_int_kernel_size_and_strides(self):
    layer = layers.MaskedConv2D(
        kernel_size=3,
        input_mask=layers.create_mask([1, 2], 0),
        output_mask=layers.create_mask([1, 2], 0),
        strides=2,
        kernel_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9, 0], [9, 0], [6, 0]],
          [[9, 0], [9, 0], [6, 0]],
          [[6, 0], [6, 0], [4, 0]]]])

  def test_masked_conv2d_output_shape(self):
    for image_size in [1, 2, 3, 32, 201, 224]:
      inputs = tf.ones([32, image_size, image_size, 2])
      for kernel_size in [1, 2, 3, 4, 5]:
        for strides in [1, 2, 3]:
          layer = layers.MaskedConv2D(
              kernel_size=(kernel_size, kernel_size),
              input_mask=layers.create_mask([1, 2], 0),
              output_mask=layers.create_mask([1, 2], 0),
              strides=(strides, strides),
              kernel_initializer=tf.initializers.ones())
          output_shape = layer.build(inputs.shape)
          output = layer.apply(inputs, training=True)
          self.assertEqual(output.shape, output_shape)

  def test_masked_conv2d_trainable_tensors(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=layers.create_mask([1, 2], 0),
        output_mask=layers.create_mask([1, 2], 0),
        kernel_initializer=tf.initializers.ones())

    input_shape = tf.TensorShape([1, 6, 6, 2])
    layer.build(input_shape)

    trainable_tensors = layer.trainable_tensors()
    self.assertNotEmpty(trainable_tensors)
    for tensor in trainable_tensors:
      self.assertIsInstance(tensor, tf.Tensor)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_masked_conv2d_kernel_regularization_loss(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=layers.create_mask([2, 4, 8], 1),
        output_mask=layers.create_mask([3, 6, 12], 1),
        kernel_initializer=tf.initializers.constant(0.5),
        kernel_regularizer=tf.keras.regularizers.l2(3.0))

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    kernel_params = 3 * 3 * 4 * 6
    self.assertAllClose(
        kernel_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_masked_conv2d_kernel_regularization_loss_with_none_input_mask(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=None,
        output_mask=layers.create_mask([3, 6, 12], 1),
        kernel_initializer=tf.initializers.constant(0.5),
        kernel_regularizer=tf.keras.regularizers.l2(3.0))

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    kernel_params = 3 * 3 * 8 * 6
    self.assertAllClose(
        kernel_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_masked_conv2d_bias_regularization_loss(self):
    layer = layers.MaskedConv2D(
        kernel_size=(3, 3),
        input_mask=layers.create_mask([2, 4, 8], 1),
        output_mask=layers.create_mask([3, 6, 12], 1),
        bias_initializer=tf.initializers.constant(0.5),
        bias_regularizer=tf.keras.regularizers.l2(3.0),
        use_bias=True)

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    bias_params = 6
    self.assertAllClose(
        bias_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_masked_depthwise_conv2d_3x3(self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=(3, 3),
        mask=layers.create_mask([1, 2], 0),
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]]]])

  def test_masked_depthwise_conv2d_3x3_explicit_padding(self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=(3, 3),
        mask=layers.create_mask([1, 2], 0),
        depthwise_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0], [9, 0], [9, 0], [6, 0]],
          [[4, 0], [6, 0], [6, 0], [6, 0], [6, 0], [4, 0]]]])

  def test_masked_depthwise_conv2d_kernel_regularization_loss(self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=(3, 3),
        mask=layers.create_mask([4, 8], 0),
        depthwise_initializer=tf.initializers.constant(0.5),
        depthwise_regularizer=tf.keras.regularizers.l2(3.0))

    inputs = tf.random_uniform([32, 28, 28, 8])
    layer.build(inputs.shape)
    layer.apply(inputs, training=True)

    # Number of parameters in the convolutional kernel.
    self.evaluate(tf.global_variables_initializer())

    kernel_params = 3 * 3 * 4 * 1
    self.assertAllClose(
        kernel_params * 3.0 * 0.5**2,
        self.evaluate(layer.regularization_loss()))

  def test_masked_depthwise_conv2d_3x3_stride2(self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=(3, 3),
        mask=layers.create_mask([1, 2], 0),
        strides=(2, 2),
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9, 0], [9, 0], [6, 0]],
          [[9, 0], [9, 0], [6, 0]],
          [[6, 0], [6, 0], [4, 0]]]])

  def test_masked_depthwise_conv2d_3x3_stride2_explicit_padding(self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        mask=layers.create_mask([1, 2], 0),
        depthwise_initializer=tf.initializers.ones(),
        use_explicit_padding=True)

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[4, 0], [6, 0], [6, 0]],
          [[6, 0], [9, 0], [9, 0]],
          [[6, 0], [9, 0], [9, 0]]]])

  def test_masked_depthwise_conv2d_3x3_stride2_int_kernel_size_and_strides(
      self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=3,
        mask=layers.create_mask([1, 2], 0),
        strides=2,
        depthwise_initializer=tf.initializers.ones())

    inputs = tf.ones([1, 6, 6, 2])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(output),
        [[[[9, 0], [9, 0], [6, 0]],
          [[9, 0], [9, 0], [6, 0]],
          [[6, 0], [6, 0], [4, 0]]]])

  def test_masked_depthwise_conv2d_output_shape(self):
    for image_size in [1, 2, 3, 32, 201, 224]:
      inputs = tf.ones([32, image_size, image_size, 2])
      for kernel_size in [1, 2, 3, 4, 5]:
        for strides in [1, 2, 3]:
          layer = layers.MaskedDepthwiseConv2D(
              kernel_size=(kernel_size, kernel_size),
              mask=layers.create_mask([1, 2], 0),
              strides=(strides, strides),
              depthwise_initializer=tf.initializers.ones())
          output_shape = layer.build(inputs.shape)
          output = layer.apply(inputs, training=True)
          self.assertEqual(output.shape, output_shape)

  def test_masked_depthwise_conv2d_trainable_tensors(self):
    layer = layers.MaskedDepthwiseConv2D(
        kernel_size=(3, 3),
        mask=layers.create_mask([1, 2], 0),
        depthwise_initializer=tf.initializers.ones())

    input_shape = tf.TensorShape([1, 6, 6, 2])
    layer.build(input_shape)

    trainable_tensors = layer.trainable_tensors()
    self.assertNotEmpty(trainable_tensors)
    for tensor in trainable_tensors:
      self.assertIsInstance(tensor, tf.Tensor)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_masked_stateless_batch_norm(self):
    layer = layers.MaskedStatelessBatchNorm(
        mask=layers.create_mask([8, 16], 0),
        center=True,
        scale=True,
        beta_initializer=tf.initializers.zeros(),
        gamma_initializer=tf.initializers.ones(),
        epsilon=1e-12)

    inputs = tf.random_uniform([32, 28, 28, 16])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output.shape, output_shape)

    mean, std = tf.nn.moments(output, axes=[0, 1, 2])
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(mean),
        np.zeros([16]),
        rtol=1e-3,
        atol=1e-3)
    self.assertAllClose(
        self.evaluate(std),
        np.concatenate([np.ones([8]), np.zeros([8])]),
        rtol=1e-3,
        atol=1e-3)

    self.assertLen(layer.trainable_tensors(), 2)
    self.assertAllClose(
        self.evaluate(layer.trainable_tensors()),
        self.evaluate(layer.trainable_variables()))

  def test_create_mask(self):
    mask = layers.create_mask([1, 2, 5], 1)
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(mask),
        np.array([1, 1, 0, 0, 0]))

  def test_create_mask_all_zeros(self):
    mask = layers.create_mask([0, 2, 5], 0)
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(mask),
        np.array([0, 0, 0, 0, 0]))

  def test_get_conv_output_shape(self):
    inputs = tf.ones([1, 128, 128, 16])
    kernel = tf.ones([3, 3, 16, 32])
    output = tf.nn.conv2d(inputs, kernel, strides=2, padding='SAME')
    self.assertEqual(layers.get_conv_output_shape(inputs.shape, 2, 32),
                     output.shape)
    self.assertEqual(layers.get_conv_output_shape(inputs.shape, (2, 2), 32),
                     output.shape)

  def test_get_conv_output_shape_resolution_indivisible_by_two(self):
    inputs = tf.ones([1, 127, 127, 16])
    kernel = tf.ones([3, 3, 16, 32])
    output = tf.nn.conv2d(inputs, kernel, strides=2, padding='SAME')
    self.assertEqual(layers.get_conv_output_shape(inputs.shape, 2, 32),
                     output.shape)

  def test_maybe_switch_v2_basic(self):
    mask = tf.placeholder(shape=(2,), dtype=tf.float32)
    layer = layers.maybe_switch_v2(
        mask,
        [
            layers.Identity(),
            layers.ReLU()
        ])

    inputs = tf.constant([1.0, 2.0, -3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    with self.cached_session() as sess:
      self.assertAllClose(sess.run(output, {mask: [0, 0]}), [0, 0, 0])
      self.assertAllClose(sess.run(output, {mask: [1, 0]}), [1, 2, -3])
      self.assertAllClose(sess.run(output, {mask: [0, 1]}), [1, 2, 0])
      self.assertAllClose(sess.run(output, {mask: [1, 1]}), [2, 4, -3])

  def test_maybe_switch_v2_with_tensor_mask_and_one_choice(self):
    mask = tf.placeholder(shape=(1,), dtype=tf.float32)
    layer = layers.maybe_switch_v2(mask, [layers.Identity()])

    inputs = tf.constant([1.0, 2.0, -3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)

    with self.cached_session() as sess:
      self.assertAllClose(sess.run(output, {mask: [0]}), [0.0, 0.0, 0.0])
      self.assertAllClose(sess.run(output, {mask: [1]}), [1.0, 2.0, -3.0])
      self.assertAllClose(sess.run(output, {mask: [2]}), [2.0, 4.0, -6.0])

  def test_maybe_switch_v2_with_none_mask_and_one_choice(self):
    layer = layers.maybe_switch_v2(None, [layers.ReLU()])

    inputs = tf.constant([1.0, 2.0, -3.0])
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)
    self.assertAllClose(self.evaluate(output), [1.0, 2.0, 0.0])

  def test_maybe_switch_v2_with_none_mask_and_multiple_choices(self):
    with self.assertRaisesRegexp(ValueError, 'Mask cannot be None'):
      layers.maybe_switch_v2(None, [layers.Identity(), layers.ReLU()])

  def test_spatial_masking(self):
    # dims: 1 x 3 x 3 x 1
    inputs = tf.constant([[[[1], [2], [3]],
                           [[4], [5], [6]],
                           [[7], [8], [9]]]])
    mask = tf.constant([[1, 1, 0],
                        [1, 1, 0],
                        [0, 0, 0]], dtype=tf.float32)
    layer = layers.SpatialMasking(mask)
    output_shape = layer.build(inputs.shape)
    output = layer.apply(inputs, training=True)
    self.assertEqual(output_shape, output.shape)
    self.assertAllClose(
        self.evaluate(output),
        np.array([[[[1], [2], [0]],
                   [[4], [5], [0]],
                   [[0], [0], [0]]]]))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()

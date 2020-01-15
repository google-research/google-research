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

"""Base class for variational method testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import absl.testing.parameterized as parameterized
import numpy as np
import tensorflow.compat.v1 as tf


class TestCase(parameterized.TestCase, tf.test.TestCase):
  """Base class for all variational layer tests."""

  def fix_random_seeds(self, np_value=10, tf_value=15):
    np.random.seed(np_value)
    tf.set_random_seed(tf_value)

  def get_data_and_weights(
      self,
      data_shape,
      weights_shape,
      data_dtype,
      weights_dtype,
      variance_value=0.0):
    raise NotImplementedError(
        "get_data_and_weight must be implemented by derived class.")

  def fix_padding_and_strides(  # pylint: disable=dangerous-default-value
      self,
      conv_op,
      strides=[1, 1, 1, 1],
      padding="VALID"):
    return functools.partial(conv_op, strides=strides, padding=padding)

  def flip_input_wrapper(self, test_op):
    return lambda x, y: test_op(y, x)

  def assertSameResult(
      self,
      test_op,
      ref_op,
      data_shape,
      weights_shape,
      data_dtype,
      weights_dtype,
      variance_value):
    data, weights = self.get_data_and_weights(
        data_shape,
        weights_shape,
        data_dtype,
        weights_dtype,
        variance_value)

    # Run the reference operator
    ref_output = ref_op(data, weights[0])

    ref_result = self.evaluate(ref_output)
    ref_shape = ref_result.shape

    # Run the operator under test
    test_output = test_op(data, weights)
    test_result = self.evaluate(test_output)
    test_shape = test_result.shape

    # Verify the output shapes and values match
    self.assertEqual(test_shape, ref_shape)
    self.assertAllClose(test_result, ref_result)

  def _determinism_helper(
      self,
      test_op,
      data_shape,
      weights_shape,
      data_dtype,
      weights_dtype,
      variance_value,
      check_same=True):
    data, weights = self.get_data_and_weights(
        data_shape,
        weights_shape,
        data_dtype,
        weights_dtype,
        variance_value)

    test_output = test_op(data, weights)
    test_result_1 = self.evaluate(test_output)
    test_result_2 = self.evaluate(test_output)

    self.assertEqual(test_result_1.shape, test_result_2.shape)
    if check_same:
      self.assertAllClose(test_result_1, test_result_2)
    else:
      self.assertNotAllClose(test_result_1, test_result_2)


class RNNTestCase(parameterized.TestCase, tf.test.TestCase):
  """Base class for all variational recurrent cell tests."""

  def input_weights_wrapper(self, rnn_cell):
    def wrapper(weights, biases, num_units):
      cell = rnn_cell(num_units)
      cell._bias = biases  # pylint: disable=protected-access
      cell._kernel = weights  # pylint: disable=protected-access
      cell.built = True
      return cell
    return wrapper

  def fix_random_seeds(self, np_value=10, tf_value=15):
    np.random.seed(np_value)
    tf.set_random_seed(tf_value)

  def set_training(self, rnn_cell):
    return functools.partial(rnn_cell, training=True)

  def set_evaluation(self, rnn_cell):
    return functools.partial(rnn_cell, training=False)

  def get_data_and_weights_and_biases(
      self,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value=0.0):
    raise NotImplementedError(
        "get_data_and_weight_and_biases must be implemented by derived class.")

  def _unroll_rnn_no_state(
      self,
      rnn_cell,
      initial_state,
      input_data):
    seq_length = input_data.get_shape().as_list()[1]

    # Apply the rnn to each timestep
    input_timesteps = tf.split(input_data, seq_length, 1)
    outputs = []

    # Run the first timestep outside the loop
    x = input_timesteps[0]
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    output, _ = rnn_cell(x, initial_state)
    outputs.append(output)

    for i in range(1, len(input_timesteps)):
      x = input_timesteps[i]
      # Get rid of the pesky time dimension of length 1
      x = tf.reshape(x, [tf.shape(x)[0], -1])

      with tf.control_dependencies([outputs[-1]]):
        output, _ = rnn_cell(x, initial_state)
      outputs.append(output)

    # Stack the outputs into a single tensor
    return tf.stack(outputs, 1)

  def assertSameNoiseForAllTimesteps(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value):
    data, weights, biases = self.get_data_and_weights_and_biases(
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value)

    rnn_cell = test_cell(weights, biases, num_units)
    initial_state = rnn_cell.zero_state(data_shape[0], dtype=data_dtype)

    output = self._unroll_rnn_no_state(
        rnn_cell,
        initial_state,
        data)
    result = self.evaluate(output)

    # Verify the output shape
    expected_result_shape = tuple(data_shape[:2] + [num_units])
    self.assertEqual(result.shape, expected_result_shape)

    # Verify that each timestep has equal results
    seq_length = data_shape[1]
    output_0 = result[:, 0, :]
    for i in range(1, seq_length):
      output_i = result[:, i, :]
      self.assertAllClose(output_0, output_i)

  def assertDifferentNoiseAcrossBatches(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value):
    data, weights, biases = self.get_data_and_weights_and_biases(
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value)

    rnn_cell = test_cell(weights, biases, num_units)
    initial_state = rnn_cell.zero_state(data_shape[0], dtype=data_dtype)

    output, _ = tf.nn.dynamic_rnn(
        rnn_cell,
        data,
        initial_state=initial_state)

    result_1 = self.evaluate(output)
    result_2 = self.evaluate(output)

    # Verify the output shape
    expected_result_shape = tuple(data_shape[:2] + [num_units])
    self.assertEqual(result_1.shape, expected_result_shape)
    self.assertEqual(result_2.shape, expected_result_shape)

    self.assertNotAllClose(result_1, result_2)

  def _determinism_helper(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value,
      check_same=True):
    data, weights, biases = self.get_data_and_weights_and_biases(
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value)

    rnn_cell = test_cell(weights, biases, num_units)
    initial_state = rnn_cell.zero_state(data_shape[0], dtype=data_dtype)

    output, _ = tf.nn.dynamic_rnn(
        rnn_cell,
        data,
        initial_state=initial_state)

    result_1 = self.evaluate(output)
    result_2 = self.evaluate(output)

    self.assertEqual(result_1.shape, result_2.shape)
    if check_same:
      self.assertAllClose(result_1, result_2)
    else:
      self.assertNotAllClose(result_1, result_2)

  def assertDeterministic(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value):
    self._determinism_helper(
        test_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value,
        check_same=True)

  def assertNonDeterministic(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value):
    self._determinism_helper(
        test_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value,
        check_same=False)

  def assertSameResult(
      self,
      test_cell,
      ref_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value):
    data, weights, biases = self.get_data_and_weights_and_biases(
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value)

    # Run the reference rnn cell
    ref_cell = self.input_weights_wrapper(ref_cell)
    ref_rnn_cell = ref_cell(weights[0], biases, num_units)
    ref_initial_state = ref_rnn_cell.zero_state(data_shape[0], dtype=data_dtype)

    ref_output, _ = tf.nn.dynamic_rnn(
        ref_rnn_cell,
        data,
        initial_state=ref_initial_state)

    ref_result = self.evaluate(ref_output)
    ref_shape = ref_result.shape

    # Run the operator under test
    test_rnn_cell = test_cell(weights, biases, num_units)
    test_initial_state = test_rnn_cell.zero_state(
        data_shape[0], dtype=data_dtype)

    test_output, _ = tf.nn.dynamic_rnn(
        test_rnn_cell,
        data,
        initial_state=test_initial_state)

    test_result = self.evaluate(test_output)
    test_shape = test_result.shape

    # Verify the output shapes and values match
    self.assertEqual(test_shape, ref_shape)
    self.assertAllClose(test_result, ref_result)

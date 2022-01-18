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

"""Base class for l0-regularization tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from state_of_sparsity.layers.utils import test_utils


class TestCase(test_utils.TestCase):
  """Base class for all l0-regularization tests."""

  def get_data_and_weights(
      self,
      data_shape,
      weights_shape,
      data_dtype,
      weights_dtype,
      variance_value=0.0):
    x = tf.ones(data_shape, dtype=data_dtype)
    theta = tf.ones(weights_shape, dtype=weights_dtype)
    log_alpha = tf.constant(
        variance_value,
        shape=weights_shape,
        dtype=weights_dtype)
    return x, (theta, log_alpha)

  def assertSameResult(
      self,
      test_op,
      ref_op,
      data_shape,
      weights_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=1e5):
    super(TestCase, self).assertSameResult(
        test_op,
        ref_op,
        data_shape,
        weights_shape,
        data_dtype,
        weights_dtype,
        variance_value=log_alpha_value)

  def assertDeterministic(
      self,
      test_op,
      data_shape,
      weights_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=0.0):
    self._determinism_helper(
        test_op,
        data_shape,
        weights_shape,
        data_dtype,
        weights_dtype,
        log_alpha_value,
        check_same=True)

  def assertNonDeterministic(
      self,
      test_op,
      data_shape,
      weights_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=0.0):
    self._determinism_helper(
        test_op,
        data_shape,
        weights_shape,
        data_dtype,
        weights_dtype,
        log_alpha_value,
        check_same=False)


class RNNTestCase(test_utils.RNNTestCase):
  """Base class for all l0-regularization recurrent cell tests."""

  def get_data_and_weights_and_biases(
      self,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype,
      weights_dtype,
      variance_value=0.0):
    x = tf.constant(0.1, data_dtype, data_shape)
    theta = tf.constant(1.0, weights_dtype, weights_shape)
    log_alpha = tf.constant(variance_value, weights_dtype, weights_shape)
    biases = tf.constant(1.0, weights_dtype, biases_shape)
    return x, (theta, log_alpha), biases

  def assertSameNoiseForAllTimesteps(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=0):
    super(RNNTestCase, self).assertSameNoiseForAllTimesteps(
        test_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value=log_alpha_value)

  def assertDifferentNoiseAcrossBatches(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=0):
    super(RNNTestCase, self).assertDifferentNoiseAcrossBatches(
        test_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value=log_alpha_value)

  def assertDeterministic(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=0):
    super(RNNTestCase, self).assertDeterministic(
        test_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value=log_alpha_value)

  def assertNonDeterministic(
      self,
      test_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=0):
    super(RNNTestCase, self).assertNonDeterministic(
        test_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value=log_alpha_value)

  def assertSameResult(
      self,
      test_cell,
      ref_cell,
      num_units,
      data_shape,
      weights_shape,
      biases_shape,
      data_dtype=tf.float32,
      weights_dtype=tf.float32,
      log_alpha_value=1e5):
    super(RNNTestCase, self).assertSameResult(
        test_cell,
        ref_cell,
        num_units,
        data_shape,
        weights_shape,
        biases_shape,
        data_dtype,
        weights_dtype,
        variance_value=log_alpha_value)

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

"""Tests for depthwise_initializers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow.compat.v1 as tf

from tunas import depthwise_initializers


class ModelOpsTest(tf.test.TestCase):

  def test_variance_scaling_untruncated_normal_fan_in(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='untruncated_normal')
    tensor = initializer([3, 5, 1024, 1])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 1))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), 1.0 / math.sqrt(3 * 5), 0.01)

  def test_variance_scaling_truncated_normal_fan_in(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='truncated_normal')
    tensor = initializer([3, 5, 1024, 1])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 1))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), 1.0 / math.sqrt(3 * 5), 0.01)

  def test_variance_scaling_uniform_fan_in(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='uniform')
    tensor = initializer([3, 5, 1024, 1])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 1))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), 1.0 / math.sqrt(3 * 5), 0.01)

  def test_variance_scaling_scale_is_2(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=2.0,
        mode='fan_in',
        distribution='untruncated_normal')
    tensor = initializer([3, 5, 1024, 1])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 1))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), math.sqrt(2.0 / (3 * 5)), 0.01)

  def test_fan_in_depth_multiplier_is_2(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='untruncated_normal')
    tensor = initializer([3, 5, 1024, 2])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 2))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), 1.0 / math.sqrt(3 * 5), 0.01)

  def test_fan_out_depth_multiplier_is_2(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_out',
        distribution='untruncated_normal')
    tensor = initializer([3, 5, 1024, 2])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 2))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), 1.0 / math.sqrt(2 * 3 * 5), 0.01)

  def test_fan_avg_depth_multiplier_is_2(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_avg',
        distribution='untruncated_normal')
    tensor = initializer([3, 5, 1024, 2])

    value = self.evaluate(tensor)
    self.assertEqual(value.shape, (3, 5, 1024, 2))
    self.assertNear(np.mean(value), 0.0, 0.01)
    self.assertNear(np.std(value), 1.0 / math.sqrt(1.5 * 3 * 5), 0.01)

  def test_depthwise_variance_scaling_end_to_end(self):
    # This is an end-to-end test for the VarianceScaling() class.
    # We apply he initializer to a tensor, and verify that the
    # distribution of outputs matches what we expect.
    input_tensor = tf.random.normal(
        shape=(32, 20, 20, 1024),
        mean=0.0,
        stddev=1)

    kernel_initializer = depthwise_initializers.DepthwiseVarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='truncated_normal')
    kernel = tf.get_variable(
        name='kernel',
        initializer=kernel_initializer,
        shape=[5, 5, 1024, 1])
    output_tensor = tf.nn.depthwise_conv2d(
        input_tensor,
        kernel,
        strides=(1, 1, 1, 1),
        padding='VALID')

    self.evaluate(tf.global_variables_initializer())
    result = self.evaluate(output_tensor)
    self.assertNear(np.mean(result), 0.0, 0.05)
    self.assertNear(np.std(result), 1.0, 0.05)

  def test_depthwise_he_normal_initializer_end_to_end(self):
    # This is an end-to-end test for the depthwise_he_normal() function.
    # We apply a depthwise_he_normal() to a tensor, and verify that the
    # distribution of outputs matches what we expect.
    input_tensor = tf.random.normal(
        shape=(32, 20, 20, 1024),
        mean=0.0,
        stddev=1)

    kernel_initializer = depthwise_initializers.depthwise_he_normal()
    kernel = tf.get_variable(
        name='kernel',
        initializer=kernel_initializer,
        shape=[5, 5, 1024, 1])
    output_tensor = tf.nn.depthwise_conv2d(
        tf.nn.relu(input_tensor),
        kernel,
        strides=(1, 1, 1, 1),
        padding='VALID')

    self.evaluate(tf.global_variables_initializer())
    result = self.evaluate(output_tensor)
    self.assertNear(np.mean(result), 0.0, 0.05)
    self.assertNear(np.std(result), 1.0, 0.05)

  def test_variance_scaling_initializer_dtypes(self):
    initializer0 = depthwise_initializers.DepthwiseVarianceScaling()
    tensor0 = initializer0([3, 3, 128, 1])
    self.assertEqual(tensor0.dtype, tf.float32)

    initializer1 = depthwise_initializers.DepthwiseVarianceScaling()
    tensor1 = initializer1([3, 3, 128, 1], dtype=tf.float64)
    self.assertEqual(tensor1.dtype, tf.float64)

    initializer2 = depthwise_initializers.DepthwiseVarianceScaling(
        dtype=tf.float64)
    tensor2 = initializer2([3, 3, 128, 1])
    self.assertEqual(tensor2.dtype, tf.float64)

  def test_variance_scaling_seed(self):
    initializer = depthwise_initializers.DepthwiseVarianceScaling(seed=42)
    tensor1 = initializer([3, 3, 128, 1])
    tensor2 = initializer([3, 3, 128, 1])
    self.assertAllClose(self.evaluate(tensor1), self.evaluate(tensor2))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()

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

"""Tests the soft_sort.squash module."""

import functools
import tensorflow.compat.v2 as tf

from soft_sort import squash


class SquashTest(tf.test.TestCase):

  def setUp(self):
    super(SquashTest, self).setUp()
    tf.random.set_seed(0)
    self.input_tensor = tf.random.normal((3, 2, 10000, 2))
    self.axis = 2

  def test_reduce_softmax(self):
    softmax_fn = functools.partial(
        squash.reduce_softmax, x=self.input_tensor, axis=self.axis)

    # Tests the dimension of the output.
    output_tensor = softmax_fn(tau=1e-6)
    self.assertAllEqual((3, 2, 2), tf.shape(output_tensor))

    # Tests the value of the output when the inverse temperature is low.
    mu = tf.math.reduce_mean(self.input_tensor, axis=self.axis)
    self.assertAllClose(mu, output_tensor, atol=1e-3)

    # Tests the ordering of two softmaxes for two different taus.
    output_tensor2 = softmax_fn(tau=1.0)
    self.assertTrue(tf.math.reduce_all(output_tensor < output_tensor2))

    # Tests the value of the output when the inverse temperature is high.
    self.assertAllClose(
        tf.math.reduce_max(self.input_tensor, axis=self.axis),
        softmax_fn(tau=1e12))

    # Tests the value of the output when the temperature is very negative.
    self.assertAllClose(
        tf.math.reduce_min(self.input_tensor, axis=self.axis),
        softmax_fn(tau=-1e12))

  def test_whiten(self):
    mu = 20.0
    sigma = 4.0
    input_tensor = sigma * self.input_tensor + mu

    # Tests that the input tensor is not standardized and centered.
    means = tf.math.reduce_mean(input_tensor, axis=self.axis)
    stds = tf.math.reduce_std(input_tensor, axis=self.axis)
    self.assertAllClose(means, mu * tf.ones(means.shape), rtol=1e-2)
    self.assertAllClose(stds, sigma * tf.ones(stds.shape), rtol=1e-2)

    # Whiten the data and tests that the output tensor is standardized and
    # centered.
    output_tensor = squash.whiten(input_tensor, axis=self.axis)
    output_means = tf.math.reduce_mean(output_tensor, axis=self.axis)
    output_stds = tf.math.reduce_std(output_tensor, axis=self.axis)
    self.assertAllClose(
        output_means, 0.0 * tf.ones(output_means.shape), atol=1e-3)
    self.assertAllClose(
        output_stds, 1.0 * tf.ones(output_stds.shape), atol=1e-3)

  def test_soft_stretch(self):
    in_segment_tensor = tf.random.uniform(self.input_tensor.shape)
    input_tensor = 0.2 + 0.4 * in_segment_tensor
    output_tensor = squash.soft_stretch(input_tensor, axis=self.axis)
    self.assertAllClose(output_tensor, in_segment_tensor, atol=1e-2)

  def test_group_rescale(self):

    def count_low_values(tau):
      output = squash.group_rescale(input_tensor, tau=tau, stretch=True)
      return tf.reduce_sum(tf.cast(output < 0.5, dtype=tf.int32), axis=1)

    num_samples = 10000
    input_tensor = tf.random.normal((1, num_samples), mean=2.0, stddev=4.0)

    # When tau is zero, half of the output should be lower than half.
    num_low_zero = count_low_values(tau=0.0)
    self.assertAlmostEqual(num_low_zero / num_samples, 0.5, delta=0.01)

    # When tau is negative there will be less values under half.
    num_low_negative = count_low_values(tau=-10.0)
    num_low_positive = count_low_values(tau=10.0)

    self.assertGreater(num_low_positive, num_low_negative)
    self.assertGreater(num_low_positive, num_low_zero)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

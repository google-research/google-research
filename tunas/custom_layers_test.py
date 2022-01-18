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

"""Tests for custom_layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from tunas import custom_layers


class CustomLayersTest(tf.test.TestCase, parameterized.TestCase):

  def test_update_exponential_moving_average(self):
    tensor = tf.placeholder(dtype=tf.float32, shape=())
    average = custom_layers.update_exponential_moving_average(
        tensor, momentum=0.9)

    with self.cached_session() as sess:
      # Initial values:
      #   numerator = 0
      #   denominator = 0
      sess.run(tf.global_variables_initializer())

      # After first update:
      #   numerator = 0.9*numerator + 0.1*1 = 0.1
      #   denominator = 0.9*denominator + 0.1 = 0.1
      #   average1 = numerator / denominator = 1.0
      average1 = sess.run(average, {tensor: 1.0})
      self.assertNear(average1, 1.0, err=1e-5)

      # After second update:
      #   numerator = 0.9*numerator + 0.1*2 = 0.9*0.1 + 0.2 = 0.29
      #   denominator = 0.9*denominator + 0.1 = 0.9*0.1 + 0.1 = 0.19
      #   average1 = numerator / denominator = 0.29 / 0.19, or about 1.53
      average2 = sess.run(average, {tensor: 2.0})
      self.assertNear(average2, 0.29/0.19, err=1e-5)

  def test_cosine_decay_with_linear_warmup(self):
    global_steps = [0, 2, 5, 10, 100]
    expected_lrs = [0, 0.8, 2, 1+np.cos(5/95*np.pi), 0]

    for global_step, expected_lr in zip(global_steps, expected_lrs):
      lr = custom_layers.cosine_decay_with_linear_warmup(
          peak_learning_rate=2.,
          global_step=tf.constant(global_step),
          max_global_step=100,
          warmup_steps=5)
      self.assertAllClose(self.evaluate(lr), expected_lr)

  def test_cosine_decay_with_linear_warmup_and_zero_warmup_steps(self):
    global_steps = [0, 10, 50, 70, 90]
    expected_lrs = [2, 1+np.cos(10/50*np.pi), 0, 0, 0]

    for global_step, expected_lr in zip(global_steps, expected_lrs):
      lr = custom_layers.cosine_decay_with_linear_warmup(
          peak_learning_rate=2.,
          global_step=tf.constant(global_step),
          max_global_step=50,
          warmup_steps=0)
      self.assertAllClose(self.evaluate(lr), expected_lr)

  def test_linear_warmup(self):
    global_step = tf.placeholder(dtype=tf.int64, shape=())
    output = custom_layers.linear_warmup(global_step, warmup_steps=10)

    with self.session() as sess:
      self.assertAllClose(0.0, sess.run(output, {global_step: 0}))
      self.assertAllClose(0.1, sess.run(output, {global_step: 1}))
      self.assertAllClose(0.2, sess.run(output, {global_step: 2}))
      self.assertAllClose(0.9, sess.run(output, {global_step: 9}))
      self.assertAllClose(1.0, sess.run(output, {global_step: 10}))
      self.assertAllClose(1.0, sess.run(output, {global_step: 11}))
      self.assertAllClose(1.0, sess.run(output, {global_step: 500}))
      self.assertAllClose(1.0, sess.run(output, {global_step: 1 << 32}))

  def test_linear_warmup_with_zero_warmup_steps(self):
    global_step = tf.placeholder(dtype=tf.int64, shape=())
    output = custom_layers.linear_warmup(global_step, warmup_steps=0)

    with self.session() as sess:
      self.assertAllClose(1.0, sess.run(output, {global_step: 0}))
      self.assertAllClose(1.0, sess.run(output, {global_step: 1}))
      self.assertAllClose(1.0, sess.run(output, {global_step: 50}))

  def test_linear_warmup_with_negative_warmup_steps(self):
    global_step = tf.placeholder(dtype=tf.int64, shape=())
    with self.assertRaisesRegex(ValueError, 'Invalid warmup_steps'):
      custom_layers.linear_warmup(global_step, warmup_steps=-1)

  def test_linear_decay(self):
    global_step = tf.placeholder(dtype=tf.int32, shape=())
    decay_value = custom_layers.linear_decay(global_step, 10)

    with self.session() as sess:
      self.assertAllClose(sess.run(decay_value, {global_step: -500}), 1.0)
      self.assertAllClose(sess.run(decay_value, {global_step: 0}), 1.0)
      self.assertAllClose(sess.run(decay_value, {global_step: 5}), 0.5)
      self.assertAllClose(sess.run(decay_value, {global_step: 9}), 0.1)
      self.assertAllClose(sess.run(decay_value, {global_step: 10}), 0)
      self.assertAllClose(sess.run(decay_value, {global_step: 50}), 0)
      self.assertAllClose(sess.run(decay_value, {global_step: 5000000}), 0)

  def test_transposed_initializer(self):
    initializer = tf.initializers.random_normal(seed=23)
    transposed_initializer = custom_layers.TransposedInitializer(initializer)

    tensor = initializer([3, 5, 64, 2])
    value = self.evaluate(tensor)

    transposed_tensor = transposed_initializer([3, 5, 2, 64])
    transposed_value = self.evaluate(transposed_tensor)
    self.assertAllClose(
        transposed_value, value.transpose([0, 1, 3, 2]))

if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()

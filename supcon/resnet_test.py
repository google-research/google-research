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

# Lint as: python3
"""Tests for supcon.resnet."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from supcon import resnet as resnet_lib


class ResNetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('ResNetV1-32', resnet_lib.ResNetV1, 32),
      ('ResNetV1-51', resnet_lib.ResNetV1, 51),
      ('ResNext-34', resnet_lib.ResNext, 34),
      ('ResNext-51', resnet_lib.ResNext, 51),
  )
  def testInvalidDepth(self, ctor, depth):
    with self.assertRaisesRegex(ValueError, r'is not a valid \w+ depth'):
      ctor(depth)

  @parameterized.named_parameters(
      ('ResNetV1-18', resnet_lib.ResNetV1, 18),
      ('ResNetV1-34', resnet_lib.ResNetV1, 34),
      ('ResNetV1-50', resnet_lib.ResNetV1, 50),
      ('ResNetV1-101', resnet_lib.ResNetV1, 101),
      ('ResNetV1-152', resnet_lib.ResNetV1, 152),
      ('ResNetV1-200', resnet_lib.ResNetV1, 200),
      ('ResNext-50', resnet_lib.ResNext, 50),
      ('ResNext-101', resnet_lib.ResNext, 101),
      ('ResNext-152', resnet_lib.ResNext, 152),
      ('ResNext-200', resnet_lib.ResNext, 200),
  )
  def testValidDepth(self, ctor, depth):
    batch_size = 2
    out_channels = 2048 if depth >= 50 else 512
    expected_output_shape = [batch_size, out_channels]
    inputs = tf.random.uniform(
        dtype=tf.float32, shape=[batch_size, 224, 224, 3], seed=1)
    resnet = ctor(depth)
    outputs = resnet(inputs, training=True)
    gradient = tf.gradients(outputs, inputs)
    self.assertListEqual(expected_output_shape, outputs.shape.as_list())
    self.assertEqual(inputs.dtype, outputs.dtype)
    self.assertIsNotNone(gradient)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      outputs = sess.run(outputs)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(outputs).any())

  @parameterized.named_parameters(
      ('ResNetV1-18', resnet_lib.ResNetV1, 18),
      ('ResNext-50', resnet_lib.ResNext, 50),
  )
  def testTrainingFalse(self, ctor, depth):
    batch_size = 2
    out_channels = 2048 if depth >= 50 else 512
    expected_output_shape = [batch_size, out_channels]
    inputs = tf.random.uniform(
        dtype=tf.float32, shape=[batch_size, 224, 224, 3], seed=1)
    resnet = ctor(depth)
    outputs = resnet(inputs, training=False)
    gradient = tf.gradients(outputs, inputs)
    self.assertListEqual(expected_output_shape, outputs.shape.as_list())
    self.assertEqual(inputs.dtype, outputs.dtype)
    self.assertIsNotNone(gradient)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      outputs = sess.run(outputs)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(outputs).any())

  @parameterized.named_parameters(
      ('ResNetV1-18', resnet_lib.ResNetV1, 18),
      ('ResNext-50', resnet_lib.ResNext, 50),
  )
  def testWRN(self, ctor, depth):
    batch_size = 2
    out_channels = 4096 if depth >= 50 else 1024
    expected_output_shape = [batch_size, out_channels]
    inputs = tf.random.uniform(
        dtype=tf.float32, shape=[batch_size, 32, 32, 3], seed=1)
    resnet = ctor(
        depth,
        width=2,
        first_conv_kernel_size=3,
        first_conv_stride=1,
        use_initial_max_pool=False)
    outputs = resnet(inputs, training=False)
    gradient = tf.gradients(outputs, inputs)
    self.assertListEqual(expected_output_shape, outputs.shape.as_list())
    self.assertEqual(inputs.dtype, outputs.dtype)
    self.assertIsNotNone(gradient)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      outputs = sess.run(outputs)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(outputs).any())


if __name__ == '__main__':
  tf.test.main()

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

# Lint as: python3
"""Tests for supcon.classification_head."""
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf

from supcon import classification_head


class ClassificationHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('rank_1', 1),
      ('rank_4', 4),
      ('rank_8', 8),
  )
  def testIncorrectRank(self, rank):
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[10] * rank)
    with self.assertRaisesRegex(ValueError, 'is expected to have rank 2'):
      classifier = classification_head.ClassificationHead(num_classes=10)
      classifier(inputs)

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
      ('float16', tf.float16),
  )
  def testConstructClassificationHead(self, dtype):
    batch_size = 3
    num_classes = 10
    input_shape = [batch_size, 4]
    expected_output_shape = [batch_size, num_classes]
    inputs = tf.random.uniform(input_shape, seed=1, dtype=dtype)
    classifier = classification_head.ClassificationHead(num_classes=num_classes)
    output = classifier(inputs)
    self.assertListEqual(expected_output_shape, output.shape.as_list())
    self.assertEqual(inputs.dtype, output.dtype)

  def testGradient(self):
    inputs = tf.random.uniform((3, 4), dtype=tf.float64, seed=1)
    classifier = classification_head.ClassificationHead(num_classes=10)
    output = classifier(inputs)
    gradient = tf.gradients(output, inputs)
    self.assertIsNotNone(gradient)

  def testCreateVariables(self):
    inputs = tf.random.uniform((3, 4), dtype=tf.float64, seed=1)
    classifier = classification_head.ClassificationHead(num_classes=10)
    classifier(inputs)
    self.assertLen(
        [var for var in tf.trainable_variables() if 'kernel' in var.name], 1)
    self.assertLen(
        [var for var in tf.trainable_variables() if 'bias' in var.name], 1)

  def testInputOutput(self):
    batch_size = 3
    num_classes = 10
    expected_output_shape = (batch_size, num_classes)
    inputs = tf.random.uniform((batch_size, 4), dtype=tf.float64, seed=1)
    classifier = classification_head.ClassificationHead(num_classes=num_classes)
    output_tensor = classifier(inputs)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      outputs = sess.run(output_tensor)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(outputs).any())
      self.assertEqual(outputs.shape, expected_output_shape)


if __name__ == '__main__':
  tf.test.main()

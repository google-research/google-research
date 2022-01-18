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
"""Tests for supcon.projection_head."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from supcon import projection_head as projection_head_lib


class ProjectionHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('rank_1', 1),
      ('rank_4', 4),
      ('rank_8', 8),
  )
  def testIncorrectRank(self, rank):
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[10] * rank)
    with self.assertRaisesRegex(ValueError, 'is expected to have rank 2'):
      projection_head = projection_head_lib.ProjectionHead()
      projection_head(inputs)

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
      ('float16', tf.float16),
  )
  def testConstructProjectionHead(self, dtype):
    shape = [3, 4]
    feature_dims = [2048, 128]
    expected_output_shape = [3, 128]
    inputs = tf.random.uniform(shape, seed=1, dtype=dtype)
    projection_head = projection_head_lib.ProjectionHead(
        feature_dims=feature_dims)
    output = projection_head(inputs)
    self.assertListEqual(expected_output_shape, output.shape.as_list())
    self.assertEqual(inputs.dtype, output.dtype)

  def testGradient(self):
    inputs = tf.random.uniform((3, 4), dtype=tf.float64, seed=1)
    projection_head = projection_head_lib.ProjectionHead()
    output = projection_head(inputs)
    gradient = tf.gradients(output, inputs)
    self.assertIsNotNone(gradient)

  @parameterized.named_parameters(
      ('1_layer', 1, False, False),
      ('1_layer_bn_with_beta', 1, True, True),
      ('1_layer_bn_no_beta', 1, True, False),
      ('2_layer', 2, False, False),
      ('2_layer_bn_with_beta', 2, True, True),
      ('2_layer_bn_no_beta', 2, True, False),
      ('4_layer', 4, False, False),
      ('4_layer_bn_with_beta', 4, True, True),
      ('4_layer_bn_no_beta', 4, True, False),
  )
  def testCreateVariables(self, num_projection_layers, use_batch_norm,
                          use_batch_norm_beta):
    feature_dims = (128,) * num_projection_layers
    inputs = tf.random.uniform((3, 4), dtype=tf.float64, seed=1)
    projection_head = projection_head_lib.ProjectionHead(
        feature_dims=feature_dims,
        use_batch_norm=use_batch_norm,
        use_batch_norm_beta=use_batch_norm_beta)
    projection_head(inputs)
    self.assertLen(
        [var for var in tf.trainable_variables() if 'kernel' in var.name],
        num_projection_layers)
    self.assertLen(
        [var for var in tf.trainable_variables() if 'bias' in var.name],
        0 if use_batch_norm else num_projection_layers - 1)
    self.assertLen(
        [var for var in tf.trainable_variables() if 'gamma' in var.name],
        num_projection_layers - 1 if use_batch_norm else 0)
    self.assertLen(
        [var for var in tf.trainable_variables() if 'beta' in var.name],
        (num_projection_layers - 1 if
         (use_batch_norm and use_batch_norm_beta) else 0))

  def testInputOutput(self):
    feature_dims = (128, 128)
    expected_output_shape = (3, 128)
    inputs = tf.random.uniform((3, 4), dtype=tf.float64, seed=1)
    projection_head = projection_head_lib.ProjectionHead(
        feature_dims=feature_dims)
    output_tensor = projection_head(inputs)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      outputs = sess.run(output_tensor)
      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(outputs).any())
      self.assertEqual(outputs.shape, expected_output_shape)

  @parameterized.named_parameters(
      ('training', True),
      ('not_training', False),
  )
  def testBatchNormIsTraining(self, is_training):
    feature_dims = (128, 128)
    inputs = tf.random.uniform((3, 4), dtype=tf.float64, seed=1)
    projection_head = projection_head_lib.ProjectionHead(
        feature_dims=feature_dims, use_batch_norm=True)
    outputs = projection_head(inputs, training=is_training)
    statistics_vars = [
        var for var in tf.all_variables() if 'moving_' in var.name
    ]
    self.assertLen(statistics_vars, 2)
    grads = tf.gradients(outputs, statistics_vars)
    self.assertLen(grads, 2)
    if is_training:
      self.assertAllEqual([None, None], grads)
      self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    else:
      self.assertNotIn(None, grads)


if __name__ == '__main__':
  tf.test.main()

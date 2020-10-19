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

# Lint as: python3
"""Tests for supcon.blocks."""

import tensorflow.compat.v1 as tf

from supcon import blocks


# This is the equivalent of tf.tpu.bfloat16_scope but it can run on CPU where
# bfloat16 isn't supported.
def custom_float16_getter(getter, *args, **kwargs):
  cast_to_float16 = False
  requested_dtype = kwargs['dtype']
  if requested_dtype == tf.float16:
    # Only change the variable dtype if doing so does not decrease variable
    # precision.
    kwargs['dtype'] = tf.float32
    cast_to_float16 = True
  var = getter(*args, **kwargs)
  # This if statement is needed to guard the cast, because batch norm
  # assigns directly to the return value of this custom getter. The cast
  # makes the return value not a variable so it cannot be assigned. Batch
  # norm variables are always in fp32 so this if statement is never
  # triggered for them.
  if cast_to_float16:
    var = tf.cast(var, tf.float16)
  return var


class BlocksTest(tf.test.TestCase):

  def test_padded_conv_can_be_called(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float32)
    block = blocks.Conv2DFixedPadding(
        filters=4, kernel_size=3, strides=2, data_format='channels_last')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertListEqual([2, 8, 8, 4], outputs.shape.as_list())

  def test_padded_conv_can_be_called_float16(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float16)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.Conv2DFixedPadding(
          filters=4, kernel_size=3, strides=2, data_format='channels_last')
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertListEqual([2, 8, 8, 4], outputs.shape.as_list())

  def test_padded_conv_can_be_called_channels_first(self):
    inputs = tf.random.normal([2, 32, 16, 16], dtype=tf.float32)
    block = blocks.Conv2DFixedPadding(
        filters=4, kernel_size=3, strides=2, data_format='channels_first')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertListEqual([2, 4, 8, 8], outputs.shape.as_list())

  def test_group_conv2d_can_be_called(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float32)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.GroupConv2D(
          filters=4,
          kernel_size=3,
          strides=2,
          data_format='channels_last',
          groups=2)
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertListEqual([2, 8, 8, 4], outputs.shape.as_list())

  def test_group_conv2d_can_be_called_float16(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float16)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.GroupConv2D(
          filters=4,
          kernel_size=3,
          strides=2,
          data_format='channels_last',
          groups=2)
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertListEqual([2, 8, 8, 4], outputs.shape.as_list())

  def test_group_conv2d_can_be_called_channels_first(self):
    inputs = tf.random.normal([2, 32, 16, 16], dtype=tf.float32)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.GroupConv2D(
          filters=4,
          kernel_size=3,
          strides=2,
          data_format='channels_first',
          groups=2)
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertListEqual([2, 4, 8, 8], outputs.shape.as_list())

  def test_bottleneck_block_can_be_called(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float32)
    block = blocks.BottleneckResidualBlock(
        filters=3,
        strides=2,
        use_projection=True,
        data_format='channels_last')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 8, 8, 12], outputs.shape.as_list())

  def test_bottleneck_block_can_be_called_float16(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float16)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.BottleneckResidualBlock(
          filters=3,
          strides=2,
          use_projection=True,
          data_format='channels_last')
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 8, 8, 12], outputs.shape.as_list())

  def test_bottleneck_block_can_be_called_channels_first(self):
    inputs = tf.random.normal([2, 32, 16, 16], dtype=tf.float32)
    block = blocks.BottleneckResidualBlock(
        filters=3,
        strides=2,
        use_projection=True,
        data_format='channels_first')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 12, 8, 8], outputs.shape.as_list())

  def test_residual_block_can_be_called(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float32)
    block = blocks.ResidualBlock(
        filters=3,
        strides=2,
        use_projection=True,
        data_format='channels_last')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 8, 8, 3], outputs.shape.as_list())

  def test_residual_block_can_be_called_float16(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float16)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.ResidualBlock(
          filters=3,
          strides=2,
          use_projection=True,
          data_format='channels_last')
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 8, 8, 3], outputs.shape.as_list())

  def test_residual_block_can_be_called_channels_first(self):
    inputs = tf.random.normal([2, 32, 16, 16], dtype=tf.float32)
    block = blocks.ResidualBlock(
        filters=3,
        strides=2,
        use_projection=True,
        data_format='channels_first')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 3, 8, 8], outputs.shape.as_list())

  def test_resnext_block_can_be_called(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float32)
    block = blocks.ResNextBlock(
        filters=64,
        strides=2,
        use_projection=True,
        data_format='channels_last')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 8, 8, 256], outputs.shape.as_list())

  def test_resnext_block_can_be_called_float16(self):
    inputs = tf.random.normal([2, 16, 16, 32], dtype=tf.float16)
    with tf.variable_scope('float16', custom_getter=custom_float16_getter):
      block = blocks.ResNextBlock(
          filters=64,
          strides=2,
          use_projection=True,
          data_format='channels_last')
      outputs = block(inputs, training=True)
      grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 8, 8, 256], outputs.shape.as_list())

  def test_resnext_block_can_be_called_channels_first(self):
    inputs = tf.random.normal([2, 32, 16, 16], dtype=tf.float32)
    block = blocks.ResNextBlock(
        filters=64,
        strides=2,
        use_projection=True,
        data_format='channels_first')
    outputs = block(inputs, training=True)
    grads = tf.gradients(outputs, inputs)
    self.assertTrue(tf.compat.v1.trainable_variables())
    self.assertTrue(grads)
    self.assertTrue(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertListEqual([2, 256, 8, 8], outputs.shape.as_list())


if __name__ == '__main__':
  tf.test.main()

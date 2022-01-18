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

"""Tests the `vgg` module."""

import tensorflow as tf

from flare_removal.python import vgg


class Vgg19Test(tf.test.TestCase):

  def test_tap_out_layers_populated(self):
    vgg_19 = vgg.Vgg19(
        tap_out_layers=['block1_conv2', 'block2_pool'], weights=None)
    self.assertAllEqual(vgg_19.tap_out_layers, ['block1_conv2', 'block2_pool'])

  def test_duplicate_layers(self):
    with self.assertRaises(ValueError):
      vgg.Vgg19(tap_out_layers=['block1_conv1', 'block1_conv1'], weights=None)

  def test_invalid_layers(self):
    with self.assertRaisesRegex(ValueError, 'block1_conv3'):
      vgg.Vgg19(tap_out_layers=['block1_conv3'], weights=None)

  def test_output_shape(self):
    vgg_19 = vgg.Vgg19(
        tap_out_layers=['block1_conv2', 'block4_conv2', 'block5_pool'],
        weights=None)
    images = tf.ones((4, 384, 512, 3)) * 0.5
    features = vgg_19(images)
    self.assertAllEqual(features[0].shape, [4, 384, 512, 64])
    self.assertAllEqual(features[1].shape, [4, 48, 64, 512])
    self.assertAllEqual(features[2].shape, [4, 12, 16, 512])


class IdentityInitializerTest(tf.test.TestCase):

  def setUp(self):
    super(IdentityInitializerTest, self).setUp()
    self._initializer = vgg.IdentityInitializer()

  def test_one_channel(self):
    kernel = self._initializer([5, 5, 1, 1])
    x = tf.random.uniform([2, 512, 512, 1], seed=0)
    y = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
    self.assertAllClose(y, x)

  def test_equal_input_output_channels(self):
    kernel = self._initializer([3, 5, 64, 64])
    x = tf.random.uniform([2, 256, 256, 64], seed=0)
    y = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
    self.assertAllClose(y, x)

  def test_more_output_channels_than_input(self):
    kernel = self._initializer([5, 3, 3, 64])
    x = tf.random.uniform([2, 512, 512, 3], seed=0)
    y = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
    self.assertAllClose(y[Ellipsis, :3], x)
    self.assertAllEqual(tf.math.count_nonzero(y[Ellipsis, 3:]), 0)

  def test_more_input_channels_than_output(self):
    kernel = self._initializer([3, 3, 64, 3])
    x = tf.random.uniform([2, 256, 256, 64], seed=0)
    y = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
    self.assertAllClose(y, x[Ellipsis, :3])


class ContextAggregationNetworkTest(tf.test.TestCase):

  def test_output_shape(self):
    x = tf.random.uniform([2, 256, 256, 3], seed=0)
    can = vgg.build_can(input_shape=x.shape[1:])
    y = can(x)
    self.assertAllEqual(y.shape, x.shape)

  def test_contains_named_conv_blocks(self):
    can = vgg.build_can(name='can')
    for i in range(9):
      self.assertIsNotNone(can.get_layer(name=f'can_g_conv{i}'))
    self.assertIsNotNone(can.get_layer(name='can_g_conv_last'))

  def test_first_conv_block_shapes(self):
    can = vgg.build_can(input_shape=[512, 512, 3], name='can')
    conv0 = can.get_layer(name='can_g_conv0')
    # The following shapes are explicitly described by Zhang et al. in Section 3
    # of the paper.
    self.assertAllEqual(conv0.input.shape, [None, 512, 512, 1475])
    self.assertAllEqual(conv0.output.shape, [None, 512, 512, 64])


if __name__ == '__main__':
  tf.test.main()

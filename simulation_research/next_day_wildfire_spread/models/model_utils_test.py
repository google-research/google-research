# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for model_utils."""

from absl.testing import parameterized
import tensorflow as tf

from simulation_research.next_day_wildfire_spread.models import model_utils

# TODO(b/237700589): Remove this once the global flag is on.
tf.compat.v2.keras.backend.experimental.enable_tf_random_generator()


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  _FILTERS = (model_utils.CONV2D_FILTERS_DEFAULT, 32)

  @parameterized.parameters(*zip(_FILTERS))
  def testConv2DLayerShape(self, filters):
    """Checks that Conv2D layer has the right shape."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.conv2d_layer(filters=filters)(input_tensor)
    self.assertEqual(result.shape, [1, 4, 4, filters])

  def testConv2DLayerValues(self):
    """Tests Conv2D layer values."""
    input_tensor = tf.ones([1, 2, 2, 1], dtype=tf.float32)
    result = model_utils.conv2d_layer(filters=2)(input_tensor)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-0.79610705, 0.38973483], [-0.9016242, -0.08420736]],
          [[-0.45097342, 0.06224325], [-0.46152127, 0.1847316]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testConv2DLayerKernelSize(self):
    """Tests Conv2D layer values with kernel_size."""
    input_tensor = tf.ones([1, 2, 2, 1], dtype=tf.float32)
    result = model_utils.conv2d_layer(filters=2, kernel_size=1)(input_tensor)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-0.05686855, 1.0695227], [-0.05686855, 1.0695227]],
          [[-0.05686855, 1.0695227], [-0.05686855, 1.0695227]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testConv2DLayerStrides(self):
    """Tests Conv2D layer values with strides."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.conv2d_layer(filters=2, strides=2)(input_tensor)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.2016071, 0.94787395], [-1.3545234, 0.16193926]],
          [[-0.13540268, 0.5588134], [-0.46152127, 0.1847316]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testConv2DLayerPadding(self):
    """Tests Conv2D layer values with padding."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.conv2d_layer(filters=2, padding='valid')(input_tensor)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.2016071, 0.94787395], [-1.2016071, 0.94787395]],
          [[-1.2016071, 0.94787395], [-1.2016071, 0.94787395]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testConv2DLayerBias(self):
    """Tests Conv2D layer values with bias."""
    input_tensor = tf.ones([1, 2, 2, 1], dtype=tf.float32)
    result = model_utils.conv2d_layer(
        filters=2, use_bias=True, bias_initializer='ones')(
            input_tensor)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[0.20389295, 1.3897349], [0.0983758, 0.91579264]],
          [[0.5490266, 1.0622432], [0.53847873, 1.1847316]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testConv2DLayerL1Regularization(self):
    """Tests Conv2D layer values with L1 regularization."""
    input_tensor = tf.ones([1, 2, 2, 1], dtype=tf.float32) * 2.0
    layer = model_utils.conv2d_layer(filters=2, l1_regularization=0.01)
    layer(input_tensor)
    self.assertAllClose(tf.reduce_sum(layer.losses), 0.0456087)

  def testConv2DLayerL2Regularization(self):
    """Tests Conv2D layer values with L2 regularization."""
    input_tensor = tf.ones([1, 2, 2, 1], dtype=tf.float32)
    layer = model_utils.conv2d_layer(filters=2, l2_regularization=0.01)
    layer(input_tensor)
    self.assertAllClose(tf.reduce_sum(layer.losses), 0.0150722)

  _BATCH_SIZES = (2, 4, 8)
  _POOL_SIZE = (2, 1)

  @parameterized.parameters(*zip(_BATCH_SIZES, _POOL_SIZE))
  def testResBlockShape(self, batch_size, pool_size):
    """Checks that the residual block output has the correct shape."""
    with self.subTest(name='withDownsampling'):
      input_tensor = tf.ones([batch_size, 8, 8, 1], dtype=tf.float32)
      result = model_utils.res_block(
          input_tensor, filters=(2, 2), strides=(2, 1), pool_size=pool_size)
      self.assertEqual(result.shape, [batch_size, 4, 4, 2])
    with self.subTest(name='noDownsampling'):
      input_tensor = tf.ones([batch_size, 8, 8, 1], dtype=tf.float32)
      result = model_utils.res_block(
          input_tensor, filters=(2, 2), strides=(1, 1), pool_size=2)
      self.assertEqual(result.shape, [batch_size, 8, 8, 2])

  def testResBlockValues(self):
    """Checks that the residual block is computed correctly."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor, filters=(2, 2), strides=(2, 1), pool_size=2)
    expected = tf.constant(
        [[[[-1.6231261, -0.41469464], [-1.0504293, 0.2297965]],
          [[-1.294716, -1.0198706], [-0.90404123, 0.06797227]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testResBlockValuesDefault(self):
    """Checks that the residual block is computed correctly."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor, filters=(2, 2), strides=(2, 1), pool_size=2)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.6231261, -0.41469464], [-1.0504293, 0.2297965]],
          [[-1.294716, -1.0198706], [-0.90404123, 0.06797227]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testResBlockValuesDropout(self):
    """Checks that the residual block is computed correctly with dropout."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor, filters=(2, 2), strides=(2, 1), pool_size=2, dropout=0.5)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.6231261, -0.41469464], [-1.0504293, 0.2297965]],
          [[-1.294716, -1.0198706], [-0.90404123, 0.06797227]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testResBlockValuesBatchNormAll(self):
    """Checks residual block is computed correctly with `batch_norm` all."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor,
        filters=(2, 2),
        strides=(2, 1),
        pool_size=2,
        batch_norm='all')
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.622118, -0.41433296], [-1.0499933, 0.22951439]],
          [[-1.2940359, -1.0189044], [-0.9037515, 0.06785183]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testResBlockValuesBatchNormSome(self):
    """Checks residual block is computed correctly with `batch_norm` some."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor,
        filters=(2, 2),
        strides=(2, 1),
        pool_size=2,
        batch_norm='some')
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.6229289, -0.41454008], [-1.0505182, 0.2296291]],
          [[-1.2946827, -1.0194137], [-0.90420324, 0.06788576]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testResBlockValuesL1Regularization(self):
    """Checks residual block is computed correctly with L1 regularization."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor,
        filters=(2, 2),
        strides=(2, 1),
        pool_size=2,
        l1_regularization=1e-6)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.6231261, -0.41469464], [-1.0504293, 0.2297965]],
          [[-1.294716, -1.0198706], [-0.90404123, 0.06797227]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)

  def testResBlockValuesL2Regularization(self):
    """Checks residual block is computed correctly with L2 regularization."""
    input_tensor = tf.ones([1, 4, 4, 1], dtype=tf.float32)
    result = model_utils.res_block(
        input_tensor,
        filters=(2, 2),
        strides=(2, 1),
        pool_size=2,
        l2_regularization=1e-5)
    self.assertEqual(result.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-1.6231261, -0.41469464], [-1.0504293, 0.2297965]],
          [[-1.294716, -1.0198706], [-0.90404123, 0.06797227]]]],
        dtype=tf.float32)
    self.assertAllClose(result, expected)


if __name__ == '__main__':
  tf.test.main()

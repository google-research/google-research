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
"""Tests for ResNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim
import tf_slim.nets as nets

from data_free_distillation.main import resnet

resnet_utils = nets.resnet_utils


def create_test_input(batch_size, size=32):
  """Creates test input tensor."""
  hw = size  # Height and width of input images
  if batch_size is None:
    return tf.placeholder(tf.float32, (None, hw, hw, 3))
  else:
    return tf.constant(1.0, shape=[batch_size, hw, hw, 3], dtype=tf.float32)


def resnet_small(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 include_root_block=True,
                 reuse=None,
                 standard=True,
                 scope='resnet_small'):
  """A shallow and thin ResNet for tests."""
  block = resnet.resnet_block
  blocks = [
      block('block1', depth=2, num_units=3, stride=1, standard=standard),
      block('block2', depth=4, num_units=3, stride=2, standard=standard),
      block('block3', depth=8, num_units=3, stride=2, standard=standard),
      block('block4', depth=16, num_units=2, stride=2, standard=standard),
  ]
  return resnet.resnet(
      inputs,
      blocks,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      include_root_block=include_root_block,
      conv1_depth=64,
      reuse=reuse,
      scope=scope)


class ResNetTest(tf.test.TestCase):

  def testClassificationEndpoints(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, endpoints = resnet_small(
          inputs, num_classes, global_pool=global_pool, scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertIn('predictions', endpoints)
    self.assertListEqual(endpoints['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])
    self.assertIn('global_pool', endpoints)
    self.assertListEqual(endpoints['global_pool'].get_shape().as_list(),
                         [2, 1, 1, 16])

  def testEndpointNames(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet_small(
          inputs, num_classes, global_pool=global_pool, scope='resnet')
    expected = ['resnet/conv1']
    for block in range(1, 5):
      for unit in range(1, 4 if block < 4 else 3):
        for conv in range(1, 3):
          expected.append('resnet/block%d/unit_%d/basic_block/conv%d' %
                          (block, unit, conv))
        expected.append('resnet/block%d/unit_%d/basic_block' % (block, unit))
      expected.append('resnet/block%d/unit_1/basic_block/shortcut' % block)
      expected.append('resnet/block%d' % block)
    expected.extend(['global_pool', 'resnet/logits', 'predictions'])
    self.assertCountEqual(endpoints.keys(), expected)

  def testEndpointNamesWithBottleneckBlock(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          standard=False,
          scope='resnet')
    expected = ['resnet/conv1']
    for block in range(1, 5):
      for unit in range(1, 4 if block < 4 else 3):
        for conv in range(1, 4):
          expected.append('resnet/block%d/unit_%d/bottleneck/conv%d' %
                          (block, unit, conv))
        expected.append('resnet/block%d/unit_%d/bottleneck' % (block, unit))
      expected.append('resnet/block%d/unit_1/bottleneck/shortcut' % block)
      expected.append('resnet/block%d' % block)
    expected.extend(['global_pool', 'resnet/logits', 'predictions'])
    self.assertCountEqual(endpoints.keys(), expected)

  def testClassificationShapes(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet_small(
          inputs, num_classes, global_pool=global_pool, scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 32, 32, 2],
          'resnet/block2': [2, 16, 16, 4],
          'resnet/block3': [2, 8, 8, 8],
          'resnet/block4': [2, 4, 4, 16],
      }
      for endpoint, shape in endpoint_to_shape.items():
        self.assertListEqual(endpoints[endpoint].get_shape().as_list(), shape)

  def testClassificationShapesWithBottleneckBlock(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          standard=False,
          scope='resnet')
      endpoint_to_shape = {
          'resnet/block1/unit_1/bottleneck/conv1': [2, 32, 32, 2],
          'resnet/block1/unit_1/bottleneck/conv2': [2, 32, 32, 2],
          'resnet/block1/unit_1/bottleneck/conv3': [2, 32, 32, 8],
          'resnet/block1': [2, 32, 32, 8],
          'resnet/block2': [2, 16, 16, 16],
          'resnet/block3': [2, 8, 8, 32],
          'resnet/block4': [2, 4, 4, 64],
      }
      for endpoint, shape in endpoint_to_shape.items():
        self.assertListEqual(endpoints[endpoint].get_shape().as_list(), shape)

  def testShapesWithInputSize128x128(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, size=128)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          standard=False,
          scope='resnet')
      endpoint_to_shape = {
          'resnet/conv1': [2, 64, 64, 64],
          'resnet/block1': [2, 64, 64, 8],
          'resnet/block2': [2, 32, 32, 16],
          'resnet/block3': [2, 16, 16, 32],
          'resnet/block4': [2, 8, 8, 64],
      }
      for endpoint, shape in endpoint_to_shape.items():
        self.assertListEqual(endpoints[endpoint].get_shape().as_list(), shape)

  def testShapesWithInputSize256x256(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, size=256)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet_small(
          inputs,
          num_classes,
          global_pool=global_pool,
          standard=False,
          scope='resnet')
      endpoint_to_shape = {
          'resnet/conv1': [2, 128, 128, 64],
          'resnet/block1': [2, 64, 64, 8],
          'resnet/block2': [2, 32, 32, 16],
          'resnet/block3': [2, 16, 16, 32],
          'resnet/block4': [2, 8, 8, 64],
      }
      for endpoint, shape in endpoint_to_shape.items():
        self.assertListEqual(endpoints[endpoint].get_shape().as_list(), shape)

  def testSkipStrideShapes(self):
    num_classes = 10
    inputs = create_test_input(2)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, endpoints = resnet.resnet_18(
          inputs, num_classes, skip_first_n_strides=1, scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 32, 32, 64],
          'resnet/block2': [2, 32, 32, 128],
          'resnet/block3': [2, 16, 16, 256],
          'resnet/block4': [2, 8, 8, 512],
      }
      for endpoint, shape in endpoint_to_shape.items():
        print(endpoints[endpoint].get_shape().as_list())
        self.assertListEqual(endpoints[endpoint].get_shape().as_list(), shape)

  def testUnknownBatchSize(self):
    batch_size = 2
    global_pool = True
    num_classes = 10
    inputs = create_test_input(None)
    # noinspection PyCallingNonCallable
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, _ = resnet_small(
          inputs, num_classes, global_pool=global_pool, scope='resnet')
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, 1, 1, num_classes])
    images = create_test_input(batch_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch_size, 1, 1, num_classes))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()

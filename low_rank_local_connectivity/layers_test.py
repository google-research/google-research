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

"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from low_rank_local_connectivity import layers


def kernel_low_rank_lc_to_lc(low_rank_kernel, data_format):
  """Transforms low-rank locally connected kernel to locally connected format.

  The transformation matches implementation 1 of keras LocallyConnected2D layer.

  Args:
    low_rank_kernel: (Tensor) Low Rank Locallay Connected Network Kernel.
    data_format: (String) Can be 'channels_first' or 'channels_last'.

  Returns: A tensor that matches the shape of LocallyConnected2D
      implementation 1 kernel.

  Raises: ValueError if data_format is incorrectly specified.
  """
  (
      output_row,
      output_col,
      kernel_size_row,
      kernel_size_col,
      input_filter,
      filters) = low_rank_kernel.shape.as_list()
  if data_format == 'channels_last':
    kernel = low_rank_kernel
  elif data_format == 'channels_first':
    kernel = tf.transpose(low_rank_kernel, [0, 1, 4, 2, 3, 5])
  else:
    raise ValueError('Unsupported data format: %s' %data_format)
  return tf.reshape(
      kernel,
      (output_row * output_col,
       kernel_size_row * kernel_size_col * input_filter,
       filters))


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(layers)

  @parameterized.parameters(
      itertools.product(['', 'softmax', 'norm'],
                        list(itertools.product([True, False], [True, False])),
                        ['channels_last', 'channels_first',])
  )
  def test_variable_shapes(
      self, normalize_weights, share_combining_weights, data_format):
    spatial_rank = 2
    kernel_size = 3
    filters = 16
    if data_format == 'channels_last':
      input_shape = [2, 32, 32, 3]
    elif data_format == 'channels_first':
      input_shape = [2, 3, 32, 32]

    images = tf.constant(
        np.random.randn(*tuple(input_shape)), dtype=tf.float32)

    # Test handling variable input size if weights shared across one dimension.
    if share_combining_weights == (True, False):
      input_shape_ = input_shape[:]
      if data_format == 'channels_last':
        input_shape_[2] = None
      elif data_format == 'channels_first':
        input_shape_[3] = None
      images = tf.placeholder_with_default(images, shape=tuple(input_shape_))

    elif share_combining_weights == (False, True):
      input_shape_ = input_shape[:]
      if data_format == 'channels_last':
        input_shape_[1] = None
      elif data_format == 'channels_first':
        input_shape_[2] = None
      images = tf.placeholder_with_default(images, shape=tuple(input_shape_))

    layer = layers.LowRankLocallyConnected2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        spatial_rank=spatial_rank,
        normalize_weights=normalize_weights,
        share_row_combining_weights=share_combining_weights[0],
        share_col_combining_weights=share_combining_weights[1],
        data_format=data_format,
        input_dependent=False)
    output = layer(images)

    var_dict = {v.op.name: v for v in tf.global_variables()}

    # Make sure all generated weights are tracked in layer.weights.
    self.assertLen(var_dict, len(layer.weights))

    # Make sure the number of weights generated is correct.
    if share_combining_weights[0] and share_combining_weights[1]:
      # weights rows, weights cols, bias (rows, cols, channels), kernel bases
      self.assertLen(var_dict, 6)
    else:
      self.assertLen(var_dict, 4)

    self.evaluate(tf.global_variables_initializer())
    combining_weights = self.evaluate(layer.combining_weights)
    if data_format == 'channels_last':
      self.assertEqual(
          self.evaluate(output).shape,
          (
              input_shape[0],
              input_shape[1]-kernel_size+1,
              input_shape[2]-kernel_size+1,
              filters)
          )
    elif data_format == 'channels_first':
      self.assertEqual(
          self.evaluate(output).shape,
          (
              input_shape[0],
              filters,
              input_shape[2]-kernel_size+1,
              input_shape[3]-kernel_size+1,
              )
          )
    if normalize_weights == 'softmax':
      self.assertNDArrayNear(
          np.sum(combining_weights, axis=-1),
          np.ones(combining_weights.shape[:-1], dtype=np.float32),
          err=1e-5)
    elif normalize_weights == 'norm':
      self.assertNDArrayNear(
          np.sqrt(np.sum(combining_weights**2, axis=-1)),
          np.ones(combining_weights.shape[:-1], dtype=np.float32),
          err=1e-5)

  @parameterized.parameters(
      itertools.product(['', 'softmax', 'norm'],
                        ['channels_last', 'channels_first',])
  )
  def test_implementations(self, normalize_weights, data_format):
    spatial_rank = 2
    kernel_size = 3
    filters = 16
    if data_format == 'channels_last':
      input_shape = (2, 32, 32, 3)
    if data_format == 'channels_first':
      input_shape = (2, 3, 32, 32)

    images = tf.constant(
        np.random.randn(*input_shape), dtype=tf.float32)
    layer1 = layers.LowRankLocallyConnected2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='valid',
        spatial_rank=spatial_rank,
        normalize_weights=normalize_weights,
        share_row_combining_weights=False,
        share_col_combining_weights=False,
        data_format=data_format,
        input_dependent=False)

    layer2 = layers.LowRankLocallyConnected2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='valid',
        spatial_rank=spatial_rank,
        normalize_weights=normalize_weights,
        share_row_combining_weights=True,
        share_col_combining_weights=True,
        data_format=data_format,
        input_dependent=False)

    output1 = layer1(images)
    weights1 = tf.global_variables()

    output2 = layer2(images)
    weights2 = list(set(tf.global_variables()) - set(weights1))

    # Weights from separable_weights implementation.
    kernel_weights1 = [
        v for v in weights1 if 'combining_weights' in v.op.name][0]
    spatial_bias1 = [v for v in weights1 if 'spatial_bias' in v.op.name][0]
    bias_channels1 = [v for v in weights1 if 'bias_channels' in v.op.name][0]
    kernel_bases1 = [v for v in weights1 if 'kernel_bases' in v.op.name][0]

    # Weights from No separable_weights implementation.
    kernel_weights_col2 = [
        v for v in weights2 if 'combining_weights_col' in v.op.name][0]
    kernel_weights_row2 = [
        v for v in weights2 if 'combining_weights_row' in v.op.name][0]
    bias_row2 = [v for v in weights2 if 'bias_row' in v.op.name][0]
    bias_col2 = [v for v in weights2 if 'bias_col' in v.op.name][0]
    bias_channels2 = [v for v in weights2 if 'bias_channels' in v.op.name][0]
    kernel_bases2 = [v for v in weights2 if 'kernel_bases' in v.op.name][0]

    # Assign No separable_weights to separable_weights.
    assign_ops = []
    assign_ops.append(tf.assign(kernel_bases1, kernel_bases2))
    assign_ops.append(tf.assign(spatial_bias1, bias_col2 + bias_row2))
    assign_ops.append(tf.assign(bias_channels1, bias_channels2))
    assign_ops.append(
        tf.assign(
            kernel_weights1,
            kernel_weights_col2[tf.newaxis] +
            kernel_weights_row2[:, tf.newaxis]))
    assign_ops = tf.group(assign_ops)

    # Test different implementations give same result.
    self.evaluate(tf.global_variables_initializer())
    self.evaluate(assign_ops)
    max_error = np.max(np.abs(self.evaluate(output1 - output2)))
    self.assertLess(
        max_error,
        1e-5)

  @parameterized.parameters(
      itertools.product(
          ['', 'softmax', 'norm'],
          [True, False],
          ['channels_last', 'channels_first'],
          ['he_uniform', 'conv_init'])
      )
  def test_correct_output(
      self, normalize_weights, input_dependent,
      data_format, combining_weights_initializer):
    spatial_rank = 2
    kernel_size = 3
    filters = 16
    input_chs = 3
    if data_format == 'channels_last':
      input_shape = (1, 32, 32, input_chs)
    if data_format == 'channels_first':
      input_shape = (1, input_chs, 32, 32)

    images = tf.constant(
        np.random.randn(*input_shape), dtype=tf.float32)
    layer1 = tf.keras.layers.LocallyConnected2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='valid',
        data_format=data_format)

    layer2 = layers.LowRankLocallyConnected2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='valid',
        spatial_rank=spatial_rank,
        normalize_weights=normalize_weights,
        combining_weights_initializer=combining_weights_initializer,
        share_row_combining_weights=False,
        share_col_combining_weights=False,
        data_format=data_format,
        input_dependent=input_dependent)

    output1 = layer1(images)
    output2 = layer2(images)

    assign_ops = []

    # Kernel from locally connected network.
    kernel1 = layer1.kernel

    combining_weights = layer2.combining_weights
    if input_dependent:
      combining_weights = tf.reduce_mean(combining_weights, axis=0)
    # Kernel from low rank locally connected network.
    kernel2 = tf.tensordot(
        combining_weights,
        tf.reshape(layer2.kernel_bases,
                   (layer2.kernel_size[0],
                    layer2.kernel_size[1],
                    input_chs,
                    layer2.spatial_rank,
                    layer2.filters)),
        [[-1], [-2]],
        name='kernel')
    kernel2 = kernel_low_rank_lc_to_lc(kernel2, data_format)

    assign_ops.append(tf.assign(kernel1, kernel2))

    # Test results consistent with keras locallyconnected2d layer.
    self.evaluate(tf.global_variables_initializer())
    for op in assign_ops:
      self.evaluate(op)

    max_error = np.max(np.abs(self.evaluate(output1 - output2)))
    self.assertLess(
        max_error,
        1e-5)

  @parameterized.parameters(
      itertools.product(
          ['', 'softmax', 'norm'],
          [True, False],
          ['channels_last', 'channels_first'],
          [[(3, 3), (2, 2)], (3, 3)],
          [1, [1, 2]])
  )
  def test_different_filters(self,
                             normalize_weights,
                             input_dependent,
                             data_format,
                             kernel_size,
                             dilations):
    spatial_rank = 2
    filters = 16
    if data_format == 'channels_last':
      input_shape = (1, 32, 32, 3)
    if data_format == 'channels_first':
      input_shape = (1, 3, 32, 32)

    images = tf.constant(
        np.random.randn(*input_shape), dtype=tf.float32)

    layer = layers.LowRankLocallyConnected2D(
        filters=filters,
        kernel_size=kernel_size,
        dilations=dilations,
        strides=(1, 1),
        padding='same',
        spatial_rank=spatial_rank,
        normalize_weights=normalize_weights,
        share_row_combining_weights=False,
        share_col_combining_weights=False,
        data_format=data_format,
        input_dependent=input_dependent)

    output = layer(images)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(output)


if __name__ == '__main__':
  tf.test.main()

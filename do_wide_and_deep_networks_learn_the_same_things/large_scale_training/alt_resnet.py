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

"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 1e-5


class BatchNormRelu(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               relu=True,
               init_zero=False,
               center=True,
               scale=True,
               data_format='channels_last',
               **kwargs):
    super(BatchNormRelu, self).__init__(**kwargs)
    self.relu = relu
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()
    if data_format == 'channels_first':
      axis = 1
    else:
      axis = -1
    self.bn = tf.keras.layers.BatchNormalization(
        axis=axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=scale,
        fused=False,
        gamma_initializer=gamma_initializer)

  def call(self, inputs, training):
    inputs = self.bn(inputs, training=training)
    if self.relu:
      inputs = tf.nn.relu(inputs)
    return inputs


class FixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, kernel_size, data_format='channels_last', **kwargs):
    super(FixedPadding, self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.data_format = data_format

  def call(self, inputs, training):
    kernel_size = self.kernel_size
    data_format = self.data_format
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
      padded_inputs = tf.pad(
          inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs


class Conv2dFixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               kernel_size,
               strides,
               kernel_regularizer,
               data_format='channels_last',
               **kwargs):
    super(Conv2dFixedPadding, self).__init__(**kwargs)
    if strides > 1:
      self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
    else:
      self.fixed_padding = None
    self.conv2d = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        kernel_regularizer=kernel_regularizer,
        data_format=data_format)

  def call(self, inputs, training):
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv2d(inputs, training=training)


class BottleneckBlock(tf.keras.layers.Layer):
  """BottleneckBlock."""

  def __init__(self,
               filters,
               strides,
               kernel_regularizer,
               use_projection=False,
               data_format='channels_last',
               **kwargs):
    super(BottleneckBlock, self).__init__(**kwargs)
    self.projection_layers = []
    if use_projection:
      filters_out = 4 * filters
      self.projection_layers.append(
          Conv2dFixedPadding(
              filters=filters_out,
              kernel_size=1,
              strides=strides,
              kernel_regularizer=kernel_regularizer,
              data_format=data_format))
      self.projection_layers.append(
          BatchNormRelu(relu=False, data_format=data_format))

    self.conv_relu_layers = []

    self.conv_relu_layers.append(
        Conv2dFixedPadding(
            filters=filters, kernel_size=1, strides=1,
            kernel_regularizer=kernel_regularizer,
            data_format=data_format))
    self.conv_relu_layers.append(
        BatchNormRelu(data_format=data_format))

    self.conv_relu_layers.append(
        Conv2dFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=strides,
            kernel_regularizer=kernel_regularizer,
            data_format=data_format))
    self.conv_relu_layers.append(
        BatchNormRelu(data_format=data_format))

    self.conv_relu_layers.append(
        Conv2dFixedPadding(
            filters=4 * filters,
            kernel_size=1,
            strides=1,
            kernel_regularizer=kernel_regularizer,
            data_format=data_format))
    self.conv_relu_layers.append(
        BatchNormRelu(relu=False, init_zero=True, data_format=data_format))

  def call(self, inputs, training):
    layers = []

    shortcut = inputs
    for layer in self.projection_layers:
      shortcut = layer(shortcut, training=training)
      layers.append(shortcut)

    for layer in self.conv_relu_layers:
      inputs = layer(inputs, training=training)
      layers.append(inputs)

    relu = tf.nn.relu(inputs + shortcut)
    layers.append(relu)
    return relu, layers


class BlockGroup(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               block_fn,
               blocks,
               strides,
               kernel_regularizer,
               data_format='channels_last',
               **kwargs):
    self._name = kwargs.get('name')
    super(BlockGroup, self).__init__(**kwargs)

    self.layers = []
    self.layers.append(
        block_fn(
            filters,
            strides,
            use_projection=True,
            kernel_regularizer=kernel_regularizer,
            data_format=data_format))

    for _ in range(1, blocks):
      self.layers.append(
          block_fn(
              filters,
              1,
              kernel_regularizer=kernel_regularizer,
              data_format=data_format))

  def call(self, inputs, training):
    layers = []
    for layer in self.layers:
      inputs, block_layers = layer(inputs, training=training)
      layers.append(block_layers)
    return tf.identity(inputs, self._name), layers


class Resnet(tf.keras.models.Model):  # pylint: disable=missing-docstring

  def __init__(self,
               block_fn,
               layers,
               width_multipliers,
               num_classes,
               kernel_regularizer=tf.keras.regularizers.l2(5e-5),
               data_format='channels_last',
               skip_dense_layer=False,
               **kwargs):
    super(Resnet, self).__init__(**kwargs)
    self.data_format = data_format
    trainable = True
    self.initial_conv_relu_max_pool = []
    self.initial_conv_relu_max_pool.append(
        Conv2dFixedPadding(
            filters=int(round(64 * width_multipliers[0])),
            kernel_size=7,
            strides=2,
            kernel_regularizer=kernel_regularizer,
            data_format=data_format,
            trainable=trainable))
    self.initial_conv_relu_max_pool.append(
        BatchNormRelu(data_format=data_format, trainable=trainable))
    self.initial_conv_relu_max_pool.append(
        tf.keras.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding='SAME',
            data_format=data_format,
            trainable=trainable))

    self.block_groups = []
    self.block_groups.append(
        BlockGroup(
            filters=int(round(64 * width_multipliers[1])),
            block_fn=block_fn,
            blocks=layers[0],
            strides=1,
            name='block_group1',
            kernel_regularizer=kernel_regularizer,
            data_format=data_format,
            trainable=trainable))
    self.block_groups.append(
        BlockGroup(
            filters=int(round(128 * width_multipliers[2])),
            block_fn=block_fn,
            blocks=layers[1],
            strides=2,
            name='block_group2',
            kernel_regularizer=kernel_regularizer,
            data_format=data_format,
            trainable=trainable))
    self.block_groups.append(
        BlockGroup(
            filters=int(round(256 * width_multipliers[3])),
            block_fn=block_fn,
            blocks=layers[2],
            strides=2,
            name='block_group3',
            kernel_regularizer=kernel_regularizer,
            data_format=data_format,
            trainable=trainable))
    self.block_groups.append(
        BlockGroup(
            filters=int(round(512 * width_multipliers[4])),
            block_fn=block_fn,
            blocks=layers[3],
            strides=2,
            name='block_group4',
            kernel_regularizer=kernel_regularizer,
            data_format=data_format,
            trainable=trainable))

    self.kernel_regularizer = kernel_regularizer
    self.num_classes = num_classes
    if not skip_dense_layer:
      self.final_layer = tf.keras.layers.Dense(
          num_classes,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          kernel_regularizer=kernel_regularizer)

  def add_dense_layer(self):
    self.final_layer = tf.keras.layers.Dense(
          self.num_classes,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          kernel_regularizer=self.kernel_regularizer)

  def call(self, inputs, training, return_layers=False):
    inputs = inputs['input_1']
    layers = []
    for layer in self.initial_conv_relu_max_pool:
      inputs = layer(inputs, training=training)
      layers.append(inputs)

    for layer in self.block_groups:
      inputs, block_layers = layer(inputs, training=training)
      layers.append(block_layers)

    if self.data_format == 'channels_last':
      inputs = tf.reduce_mean(inputs, [1, 2])
    else:
      inputs = tf.reduce_mean(inputs, [2, 3])

    inputs = tf.identity(inputs, 'final_avg_pool')
    layers.append(inputs)
    inputs = self.final_layer(inputs)
    layers.append(inputs)

    if return_layers:
      return layers
    return inputs

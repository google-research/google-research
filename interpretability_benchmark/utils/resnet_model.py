# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python2, python3
"""This is a ResNet-50 model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from six.moves import range
import tensorflow as tf

FLAGS = flags.FLAGS

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU."""
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_first'):
  """Pads the input along the spatial dimensions."""
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


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_first'):
  """Strided 2-D convolution with explicit padding."""
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs,
                   filters,
                   is_training,
                   strides,
                   use_projection=False,
                   data_format='channels_first'):
  """Standard building block for residual networks."""
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs, is_training, relu=False, init_zero=True, data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training,
                     strides,
                     use_projection=False,
                     data_format='channels_first'):
  """Bottleneck block variant for residual networks."""
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs, is_training, relu=False, init_zero=True, data_format=data_format)
  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training,
                name,
                data_format='channels_first'):
  """Creates one group of blocks for the ResNet model."""
  inputs = block_fn(
      inputs,
      filters,
      is_training,
      strides,
      use_projection=True,
      data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, 1, data_format=data_format)

  return tf.identity(inputs, name)


def resnet_50_generator(block_fn,
                        layers,
                        num_classes,
                        data_format='channels_first'):
  """Generator for ResNet v1 models."""

  def model(inputs, is_training):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_group(
        inputs=inputs,
        filters=64,
        block_fn=block_fn,
        blocks=layers[0],
        strides=1,
        is_training=is_training,
        name='block_group1',
        data_format=data_format)
    inputs = block_group(
        inputs=inputs,
        filters=128,
        block_fn=block_fn,
        blocks=layers[1],
        strides=2,
        is_training=is_training,
        name='block_group2',
        data_format=data_format)
    inputs = block_group(
        inputs=inputs,
        filters=256,
        block_fn=block_fn,
        blocks=layers[2],
        strides=2,
        is_training=is_training,
        name='block_group3',
        data_format=data_format)
    inputs = block_group(
        inputs=inputs,
        filters=512,
        block_fn=block_fn,
        blocks=layers[3],
        strides=2,
        is_training=is_training,
        name='block_group4',
        data_format=data_format)

    pool_size = (inputs.shape[1], inputs.shape[2])
    inputs = tf.layers.average_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=1,
        padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs,
                        [-1, 2048 if block_fn is bottleneck_block else 512])
    inputs = tf.layers.dense(
        inputs=inputs,
        units=num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=.01))
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def resnet_50(num_classes, data_format='channels_first'):
  """Returns the ResNet model for a given size and number of output classes."""
  return resnet_50_generator(
      block_fn=bottleneck_block,
      layers=[3, 4, 6, 3],
      num_classes=num_classes,
      data_format=data_format)

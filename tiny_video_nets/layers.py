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

"""Layers used for Video Nets."""
import numpy as np

import tensorflow.compat.v1 as tf

from tiny_video_nets import batch_norm as bn
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers


def hard_swish(x):
  with tf.name_scope('hard_swish'):
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def reshape_temporal_conv1d_bn(inputs,
                               is_training,
                               filters,
                               kernel_size,
                               num_frames=32,
                               stride=1,
                               use_relu=True,
                               data_format='channels_last'):
  """Performs 1D temporal conv.

  followed by batch normalization with reshaping.

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    is_training: 'bool' specifying whether in training mode or not.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    num_frames: 'int' number of frames in the input tensor.
    stride: 'int' temporal stride
    use_relu: 'boolean' determines if the activation function is used or not.
    data_format: `str`. Only supports 'channels_last' as the data format.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  assert data_format == 'channels_last'

  feature_shape = inputs.shape
  batch_size = feature_shape[0] // num_frames

  inputs = tf.reshape(inputs, [
      int(feature_shape[0] // num_frames), num_frames,
      feature_shape[1] * feature_shape[2], -1
  ])

  inputs = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=(kernel_size, 1),
      strides=[stride, 1],
      padding='SAME',
      use_bias=False,
      kernel_initializer=contrib_layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_IN', uniform=False),
      data_format=data_format)

  new_frames = inputs.shape[1]
  inputs = tf.reshape(
      inputs, [batch_size * new_frames, feature_shape[1], feature_shape[2], -1])

  inputs = bn.batch_norm_relu(
      inputs,
      is_training,
      relu=use_relu,
      data_format=data_format)

  return inputs, new_frames


def conv2d(inputs,
           kernel,
           filters,
           stride,
           is_training,
           use_relu=True,
           init_zero=False,
           use_bn=True,
           data_format='channels_last'):
  """Performs 2d conv.

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    kernel: `int` kernel size to be used for `conv2d` or max_pool2d` operations.
      Should be a positive integer.
    filters: `int` number of filters in the convolution.
    stride: 'int' temporal stride
    is_training: 'bool' specifying whether in training mode or not.
    use_relu: bool, apply relu after batch norm
    init_zero: bool, initialize batch_norm weights to 0 (i.e., initially ignore
      layer)
    use_bn: bool, use batch norm or not
    data_format: `str`. Only supports 'channels_last' as the data format.

  Returns:
    A `Tensor` of the same data_format
  """
  inputs = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel,
      strides=stride,
      padding='SAME',
      use_bias=False,
      kernel_initializer=contrib_layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_IN', uniform=False),
      data_format=data_format)
  if use_bn:
    inputs = bn.batch_norm_relu(
        inputs,
        is_training,
        relu=use_relu,
        init_zero=init_zero,
        data_format=data_format)
  if not use_bn and use_relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def spatial_conv(inputs,
                 conv_type,
                 kernel,
                 filters,
                 stride,
                 is_training,
                 activation_fn='relu',
                 data_format='channels_last'):
  """Performs 1x1 conv followed by 2d or depthwise conv.

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    conv_type: 'string' of "std", "depth", "maxpool", or "avgpool" this selects
      the spatial conv/pooling method.
    kernel: `int` kernel size to be used for `conv2d` or max_pool2d` operations.
      Should be a positive integer.
    filters: `int` number of filters in the convolution.
    stride: 'int' temporal stride
    is_training: 'bool' specifying whether in training mode or not.
    activation_fn: 'string' the activation function to use (relu or swish)
    data_format: `str`. Only supports 'channels_last' as the data format.

  Returns:
    A `Tensor` of the same data_format
  """

  if kernel == 1:
    return inputs
  use_relu = (activation_fn == 'relu')
  if conv_type == 'std' or conv_type == 'depth':
    inputs = conv2d(inputs, 1, filters, 1, is_training, use_relu=use_relu)
    if not use_relu:
      inputs = hard_swish(inputs)

  if conv_type == 'std' or conv_type == '1std':
    inputs = conv2d(inputs, int(kernel), filters, int(stride), is_training,
                    use_relu=use_relu)
    if not use_relu:
      inputs = hard_swish(inputs)
  elif conv_type == 'depth':
    depth_multiplier = 1
    depthwise_kernel_shape = (int(kernel), int(kernel), inputs.shape[-1],
                              depth_multiplier)

    depthwise_kernel = contrib_framework.model_variable(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        dtype=tf.float32,
        initializer=contrib_layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        trainable=True)

    inputs = tf.nn.depthwise_conv2d(
        inputs,
        tf.cast(depthwise_kernel, inputs.dtype),
        strides=[1, int(stride), int(stride), 1],
        padding='SAME',
        rate=[1, 1],
        data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

    inputs = bn.batch_norm_relu(
        inputs,
        is_training,
        relu=use_relu,
        data_format=data_format)
    if not use_relu:
      inputs = hard_swish(inputs)

  elif conv_type == 'maxpool':
    inputs = tf.layers.max_pooling2d(
        inputs,
        int(kernel),
        int(stride),
        padding='same',
        data_format=data_format)

  elif conv_type == 'avgpool':
    inputs = tf.layers.average_pooling2d(
        inputs,
        int(kernel),
        int(stride),
        padding='same',
        data_format=data_format)

  return inputs


def temporal_conv(inputs,
                  conv_type,
                  kernel,
                  filters,
                  stride,
                  is_training,
                  num_frames,
                  activation_fn='relu',
                  data_format='channels_last'):
  """Performs 1D conv.

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    conv_type: 'string' of "1d",  "maxpool", or "avgpool" controlling conv type
    kernel: `int` kernel size to be used for `conv2d` or max_pool2d` operations.
      Should be a positive integer.
    filters: `int` number of filters in the convolution.
    stride: 'int' temporal stride
    is_training: 'bool' specifying whether in training mode or not.
    num_frames: int number of frames
    activation_fn: 'string' the activation function to use (relu or swish)
    data_format: `str`. Only supports 'channels_last' as the data format.

  Returns:
    A `Tensor` of the same data_format
  """

  use_relu = (activation_fn == 'relu')
  new_frames = num_frames
  if kernel == 1:
    return inputs, new_frames
  if conv_type == '1d':
    inputs, new_frames = reshape_temporal_conv1d_bn(
        inputs=inputs,
        is_training=is_training,
        filters=filters,
        kernel_size=kernel,
        stride=stride,
        num_frames=num_frames,
        use_relu=use_relu,
        data_format=data_format)
    if not use_relu:
      inputs = hard_swish(inputs)

  elif conv_type == 'maxpool' or conv_type == 'avgpool':
    feature_shape = inputs.shape
    batch_size = feature_shape[0] // num_frames

    inputs = tf.reshape(
        inputs,
        [batch_size, num_frames, feature_shape[1] * feature_shape[2], -1])

    if conv_type == 'maxpool':
      inputs = tf.layers.max_pooling2d(
          inputs, [kernel, 1], [stride, 1],
          padding='same',
          data_format=data_format)
    elif conv_type == 'avgpool':
      inputs = tf.layers.average_pooling2d(
          inputs, [kernel, 1], [stride, 1],
          padding='same',
          data_format=data_format)

    new_frames = inputs.shape[1]
    inputs = tf.reshape(
        inputs,
        [batch_size * new_frames, feature_shape[1], feature_shape[2], -1])
  return inputs, new_frames


def squeeze_and_excite(inputs, filters, ratio, num_frames):
  """Squeeze and excite layer for videos.

  Squeeze-and-Excitation Networks
  arXiv: 1709.01507

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    filters: `int` number of filters in the convolution.
    ratio: 'float' percent to squeeze
    num_frames: int number of frames

  Returns:
    A `Tensor` of the same data_format
  """

  reduced_filters = max(int(filters * ratio), 8)
  # Shape is [batch*num_frames, width, height, channels].
  feature_shape = [inputs.shape[0], 1, 1, inputs.shape[3]]
  # Reshape to [batch, num_frames*width, height, channels]
  # for memory efficient spatio-temporal squeeze-exicte layer.
  squeeze_excite_inputs = tf.reshape(inputs, [
      int(tf.compat.dimension_value(inputs.shape[0]) // num_frames),
      num_frames*inputs.shape[1],
      inputs.shape[2], -1
  ])

  # Spatio-temporal averaging.
  squeeze_excite = tf.reduce_mean(squeeze_excite_inputs, [1, 2], keepdims=True)
  squeeze_excite = tf.layers.conv2d(
      inputs=squeeze_excite,
      filters=reduced_filters,
      kernel_size=1,
      strides=1,
      padding='SAME',
      use_bias=True,
      kernel_initializer=contrib_layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_IN', uniform=False),
      data_format='channels_last')
  squeeze_excite = tf.layers.conv2d(
      inputs=tf.nn.relu(squeeze_excite),
      filters=filters,
      kernel_size=1,
      strides=1,
      padding='SAME',
      use_bias=True,
      kernel_initializer=contrib_layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_IN', uniform=False),
      data_format='channels_last')
  squeeze_excite = tf.expand_dims(tf.nn.sigmoid(squeeze_excite), 1)
  # Add in number of frames.
  pattern = tf.stack([1, num_frames, 1, 1, 1])
  # Reshape to full spatio-temporal video size.
  return tf.reshape(tf.tile(squeeze_excite, pattern), feature_shape) * inputs


def context_gate(inputs, filters, num_frames):
  """Context Gating layer for videos.

  Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video
  Classification
  arXiv:1712.04851

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    filters: `int` number of filters in the convolution.
    num_frames: int number of frames

  Returns:
    A `Tensor` of the same data_format
  """
  # Shape is [batch*num_frames, width, height, channels].
  feature_shape = [inputs.shape[0], 1, 1, inputs.shape[3]]

  # Reshape to [batch, num_frames*width, height, channels]
  # for memory efficient spatio-temporal squeeze-exicte layer.
  context_inputs = tf.reshape(inputs, [
      int(tf.compat.dimension_value(inputs.shape[0]) // num_frames),
      num_frames*inputs.shape[1],
      inputs.shape[2], -1
  ])

  feature = tf.reduce_mean(context_inputs, [1, 2], keepdims=True)
  feature = tf.layers.conv2d(
      inputs=feature,
      filters=filters,
      kernel_size=1,
      strides=1,
      padding='SAME',
      use_bias=True,
      kernel_initializer=contrib_layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_IN', uniform=False),
      data_format='channels_last')
  feature = tf.expand_dims(tf.nn.sigmoid(feature), 1)
  pattern = tf.stack([1, num_frames, 1, 1, 1])
  return tf.reshape(tf.tile(feature, pattern), feature_shape) * inputs

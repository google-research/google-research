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

"""Defines the model for resnet v2.

Adapted from https://github.com/tensorflow/models/tree/master/official/resnet
"""

from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""

  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs,
      axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=training,
      fused=True)


def group_norm(inputs, training, data_format):
  """Perform group normalization on the inputs."""
  del training  # Unused
  assert data_format == 'channels_first'

  _, c, h, w = inputs.get_shape().as_list()
  g = min(32, c)

  with tf.variable_scope(None, 'group_norm'):
    inputs_split = tf.reshape(inputs, [-1, g, c // g, h, w])
    mean, var = tf.nn.moments(inputs_split, [2, 3, 4], keep_dims=True)
    inputs_split = (inputs_split - mean) / tf.sqrt(var + 1e-6)
    # per channel gamma and beta
    gamma = tf.get_variable('gamma', [1, c, 1, 1],
                            initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable('beta', [1, c, 1, 1],
                           initializer=tf.constant_initializer(0.0))
    output = tf.reshape(inputs_split, [-1, c, h, w]) * gamma + beta

  return output


def normalize(inputs, training, data_format, kind='batch'):
  """Perform normalization of choice on the inputs.

  Group Normalization: https://arxiv.org/abs/1803.08494
  Batch Normalization: https://arxiv.org/abs/1502.03167

  Args:
    inputs: The tensor to be normalized.
    training: Whether it is at the training phase.
    data_format: Format of the data. ('channels_first' or 'channels_last')
    kind: Type of normalization. ('group' or 'batch')

  Returns:
    A normalize version of the input with the normalization of choice.
  """
  if kind == 'batch':
    return batch_norm(inputs, training, data_format)
  elif kind == 'group':
    return group_norm(inputs, training, data_format)
  else:
    raise ValueError('Unknown norm_type "%s"' % kind)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
        Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.compat.v1.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
      data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format, norm_type):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
    norm_type: The normalization function to use ('batch' or 'group')

  Returns:
    The output tensor of the block; shape should match inputs.
  """

  shortcut = inputs
  inputs = normalize(inputs, training, data_format, norm_type)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = normalize(inputs, training, data_format, norm_type)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format, norm_type):
  """A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
    norm_type: The normalization function to use ('batch' or 'group')

  Returns:
    The output tensor of the block; shape should match inputs.
  """

  shortcut = inputs
  inputs = normalize(inputs, training, data_format, norm_type)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = normalize(inputs, training, data_format, norm_type)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = normalize(inputs, training, data_format, norm_type)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, norm_type):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
    norm_type: The normalization function to use ('batch' or 'group')


  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format, norm_type)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format,
                      norm_type)

  return tf.identity(inputs, name)


################################################################################
# ResNet callable class.
################################################################################
class ResNet(object):
  """A class for creating the resnet v2 architecture.

  A callable class that builds a resnet-32 model.

  Attributes:
    num_classes: Number of classes in the training data.
    bottleneck: Use regular blocks or bottleneck blocks.
    num_classes: The number of classes used as labels.
    num_filters: The number of filters to use for the first block layer
      of the model. This number is then doubled for each subsequent block
      layer.
    kernel_size: The kernel size to use for convolution.
    conv_stride: stride size for the initial convolutional layer
    first_pool_size: Pool size to be used for the first pooling layer.
      If none, the first pooling layer is skipped.
    first_pool_stride: stride size for the first pooling layer. Not used
      if first_pool_size is None.
    block_sizes: A list containing n values, where n is the number of sets of
      block layers desired. Each value should be the number of blocks in the
      i-th set.
    block_strides: List of integers representing the desired stride size for
      each of the sets of block layers. Should be same length as block_sizes.
    pre_activation: Use BN and ReLU before the final output layer.
    weight_decay: Coefficient for l2 regularization.
    norm_type: Type of normalization. ('group' or 'batch').
    data_format: Input format ('channels_last', 'channels_first').
    variables: A list of tf.Variable's tha contains tensorflow variables within
      the model's scope sorted by the variable's name.
  """

  def __init__(
      self,
      bottleneck,
      num_filters,
      kernel_size,
      conv_stride,
      first_pool_size,
      first_pool_stride,
      block_sizes,
      block_strides,
      pre_activation,
      weight_decay,
      norm_type,
      loss_filter_fn,
      num_classes,
      scope='resnet',
      data_format='channels_first'):

    self.num_classes = num_classes
    self.bottleneck = bottleneck
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.pre_activation = pre_activation
    self.weight_decay = weight_decay
    self.norm_type = norm_type
    self.data_format = data_format
    self._loss_filter_fn = loss_filter_fn
    self._scope = scope
    self._was_called = False

  @property
  def variables(self):
    """Returns a list of the variables within model's scope.

    Only works if the object has been called.
    """
    if not self._was_called:
      raise ValueError('This model has not been called yet.')
    all_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)
    return sorted(all_vars, key=lambda x: x.name)

  def __call__(
      self,
      inputs,
      is_training,
      end_points_collection=None):

    with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE) as scope:
      self._scope = scope.name
      if end_points_collection:
        end_points_collection['inputs'] = inputs

      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters,
          kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')

      if self.first_pool_size:
        inputs = tf.compat.v1.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=_building_block_v2, blocks=num_blocks,
            strides=self.block_strides[i], training=is_training,
            name='block_layer{}'.format(i + 1),
            data_format=self.data_format,
            norm_type=self.norm_type)
        if end_points_collection:
          end_points_collection['h{}'.format(i+1)] = inputs

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = normalize(inputs, is_training, self.data_format,
                           self.norm_type)
        inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.squeeze(inputs, axes)
      inputs = tf.compat.v1.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')

      # Add weight decay to the loss.
      if is_training:
        var_list = [v for v in tf.trainable_variables(self._scope) if
                    self._loss_filter_fn(v.name)]
        reg = contrib_layers.l2_regularizer(scale=self.weight_decay)
        l2_loss_list = list(map(reg, var_list))
        l2_loss = tf.add_n(l2_loss_list)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_loss)

      self._was_called = True
      return inputs

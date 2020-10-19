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
"""Various blocks used in constructing ResNets."""

import abc

import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import

BATCH_NORM_MOMENTUM = 0.9
BATCH_NORM_EPSILON = 1e-5


class BatchNormalization(tf.layers.BatchNormalization):
  """Batch Normalization layer that supports cross replica computation on TPU.

  This class extends the keras.BatchNormalization implementation by supporting
  cross replica means and variances. The base class implementation only computes
  moments based on mini-batch per replica (TPU core).

  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

  Note that this does not support fused batch norm, so passing fused=True will
  be ignored.
  """

  def __init__(self, **kwargs):
    """Builds the batch normalization layer.

    Arguments:
      **kwargs: input augments that are forwarded to
        tf.layers.BatchNormalization.
    """
    if 'fused' in kwargs and kwargs['fused']:
      raise ValueError('The TPU version of BatchNormalization does not support '
                       'fused=True.')
    super(BatchNormalization, self).__init__(fused=False, **kwargs)

  def _cross_replica_average(self, t):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    return tf.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(BatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards and num_shards > 1:
      # Each group has multiple replicas: here we compute group mean/variance by
      # aggregating per-replica mean/variance.
      group_mean = self._cross_replica_average(shard_mean)
      group_variance = self._cross_replica_average(shard_variance)

      # Group variance needs to also include the difference between shard_mean
      # and group_mean.
      mean_distance = tf.square(group_mean - shard_mean)
      group_variance += self._cross_replica_average(mean_distance)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


def batch_norm(init_zero=False,
               data_format='channels_last',
               use_trainable_beta=True,
               batch_norm_momentum=BATCH_NORM_MOMENTUM,
               use_global_batch_norm=True):
  """Creates a BatchNormalization layer.

  Args:
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    use_trainable_beta: Whether or not to have a trainable shift parameter.
    batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.

  Returns:
    A callable BatchNormalization layer.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1

  bn_ctor = (
      BatchNormalization
      if use_global_batch_norm else tf.layers.BatchNormalization)
  return bn_ctor(
      axis=axis,
      momentum=batch_norm_momentum,
      epsilon=BATCH_NORM_EPSILON,
      center=use_trainable_beta,
      scale=True,
      gamma_initializer=gamma_initializer)


class FixedPadding(tf.layers.Layer):
  """Pads the input along the spatial dimensions independently of input size.

  Produces a padded `Tensor` of the same `data_format` with size either intact
  (if `kernel_size == 1`) or padded (if `kernel_size > 1`).

  Attributes:
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
  """

  def __init__(self,
               kernel_size,
               data_format='channels_last',
               name='FixedPadding',
               **kwargs):
    self.kernel_size = kernel_size
    self.data_format = data_format
    super(FixedPadding, self).__init__(name=name, **kwargs)

  def compute_output_shape(self, input_shape):
    fake_input = tf.zeros(
        shape=[1] + input_shape[1:],
        dtype=(self.dtype or tf.keras.backend.floatx()))
    fake_output = self._pad(fake_input)
    return [input_shape[0]] + fake_output.shape[1:]

  def call(self, inputs):
    return self._pad(inputs)

  def _pad(self, inputs):
    pad_total = self.kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if self.data_format == 'channels_first':
      padded_inputs = tf.pad(
          inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs


class Conv2DFixedPadding(tf.layers.Layer):
  """Strided 2D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.Conv2d` alone).

  Attributes:
    filters: `int` Number of filters in the convolution.
    kernel_size: `int` Size of the kernel to be used in the convolution.
    strides: `int` Strides of the convolution.
    data_format: `str` Either "channels_first" for `[batch, channels, height,
      width]` or "channels_last" for `[batch, height, width, channels]`.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides,
               data_format='channels_last',
               name='Conv2DFixedPadding',
               **kwargs):
    super(Conv2DFixedPadding, self).__init__(name=name, **kwargs)

    self.use_padding = strides > 1
    if self.use_padding:
      self.fixed_padding = FixedPadding(kernel_size, data_format)

    self.conv = tf.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('VALID' if self.use_padding else 'SAME'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

  def compute_output_shape(self, input_shape):
    shape = input_shape
    if self.use_padding:
      shape = self.fixed_padding.compute_output_shape(shape)
    return self.conv.compute_output_shape(shape)

  def call(self, inputs):
    x = inputs
    if self.use_padding:
      x = self.fixed_padding(x)

    return self.conv(x)


def _conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.

  Copied from tensorflow/python/keras/utils/conv_utils.py

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full', 'causal'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding in ['same', 'causal']:
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride


class GroupConv2D(tf.layers.Layer):
  """A grouped 2D convolution.

  Attributes:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      filters in the convolution).
    kernel_size: kernel_size: An integer specifying the height and width of the
      2D convolution window.
    strides: An integer specifying the strides of the convolution along the
      height and width.
    padding: One of "valid" or "same" (case-insensitive).
    data_format: A string, one of channels_last (default) or channels_first. The
      ordering of the dimensions in the inputs. channels_last corresponds to
      inputs with shape (batch, height, width, channels) while channels_first
      corresponds to inputs with shape (batch, channels, height, width).
    groups: Integer, the number of groups to divide the inputs into, along the
      channel dimension, before performing the convolution. The channel
      dimension of the input and `filters` must both be divisible by `groups`.
    kernel_initializer: An initializer for the convolution kernel.
    name: A name for this object.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               data_format='channels_last',
               groups=1,
               kernel_initializer=tf.variance_scaling_initializer(),
               name='GroupConv2D',
               **kwargs):
    super(GroupConv2D, self).__init__(name=name, **kwargs)

    assert filters % groups == 0, ('`filters` must be divisible by `groups`.')

    self.filters = filters
    self.kernel_size = kernel_size
    self.kernel_initializer = kernel_initializer
    self.strides = strides
    self.padding = padding
    self.data_format = data_format
    self.groups = groups

  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_last':
      new_dim1 = _conv_output_length(input_shape[1], self.kernel_size,
                                     self.padding, self.strides)
      new_dim2 = _conv_output_length(input_shape[2], self.kernel_size,
                                     self.padding, self.strides)
      return tf.TensorShape([input_shape[0], new_dim1, new_dim2, self.filters])
    else:
      new_dim2 = _conv_output_length(input_shape[2], self.kernel_size,
                                     self.padding, self.strides)
      new_dim3 = _conv_output_length(input_shape[3], self.kernel_size,
                                     self.padding, self.strides)
      return tf.TensorShape([input_shape[0], self.filters, new_dim2, new_dim3])

  def build(self, input_shape):
    in_channels = (
        input_shape[-1]
        if self.data_format == 'channels_last' else input_shape[1])
    assert in_channels is not None, (
        'The input channel dimension can not be None.')
    assert in_channels % self.groups == 0, (
        'The input channel dimension must be divisible by `groups`.')

    filter_shape = [
        self.kernel_size, self.kernel_size, in_channels // self.groups,
        self.filters
    ]

    self.kernel = self.add_weight(
        name='kernel',
        shape=filter_shape,
        initializer=self.kernel_initializer,
        trainable=True)

    super(GroupConv2D, self).build(input_shape)

  def call(self, inputs):
    channel_axis = -1 if self.data_format == 'channels_last' else 1
    groups = tf.split(inputs, self.groups, axis=channel_axis)
    kernels = tf.split(self.kernel, self.groups, axis=3)
    stride = ([1, self.strides, self.strides, 1]
              if self.data_format == 'channels_last' else
              [1, 1, self.strides, self.strides])

    outputs = []
    for group, kernel in zip(groups, kernels):
      outputs.append(
          tf.nn.conv2d(
              group,
              filter=kernel,
              strides=stride,
              padding=self.padding.upper(),
              data_format=('NHWC'
                           if self.data_format == 'channels_last' else 'NCHW')))

    output = tf.concat(outputs, channel_axis)

    return output


class BlockInterface(tf.layers.Layer, metaclass=abc.ABCMeta):
  """An interface for block layers that can be used with resnet.BlockGroup.

  Attributes:
    filters: `int` number of filters for the convolutions.
    strides: `int` block stride. If greater than 1, this block will ultimately
      downsample the input.
    use_projection: `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    final_activation: activation to be used at the end of the block
    batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.
    name: A name for this Layer.
  """

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               data_format='channels_last',
               final_activation=tf.nn.relu,
               batch_norm_momentum=BATCH_NORM_MOMENTUM,
               use_global_batch_norm=True,
               name='ResidualBlock',
               **kwargs):
    super(BlockInterface, self).__init__(name=name, **kwargs)

    self.filters = filters
    self.strides = strides
    self.use_projection = use_projection
    self.data_format = data_format
    self.final_activation = final_activation
    self.batch_norm_momentum = batch_norm_momentum
    self.use_global_batch_norm = use_global_batch_norm

  @abc.abstractmethod
  def call(self, inputs, training=None):
    pass


class ResidualBlock(BlockInterface):
  """Standard building block for ResNet V1 with BN after convolutions.

  Attributes: See `BlockInterface`
  """

  def __init__(self, name='ResidualBlock', **kwargs):
    super(ResidualBlock, self).__init__(name=name, **kwargs)

    if self.use_projection:
      self.projection_conv = Conv2DFixedPadding(
          filters=self.filters,
          kernel_size=1,
          strides=self.strides,
          data_format=self.data_format)
      self.projection_bn = batch_norm(
          data_format=self.data_format,
          batch_norm_momentum=self.batch_norm_momentum,
          use_global_batch_norm=self.use_global_batch_norm)

    self.first_conv = Conv2DFixedPadding(
        filters=self.filters,
        kernel_size=3,
        strides=self.strides,
        data_format=self.data_format)
    self.first_bn = batch_norm(
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

    self.second_conv = Conv2DFixedPadding(
        filters=self.filters,
        kernel_size=3,
        strides=1,
        data_format=self.data_format)
    self.second_bn = batch_norm(
        init_zero=True,
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

  def call(self, inputs, training=None):
    x = inputs
    shortcut = x
    if self.use_projection:
      # Projection shortcut in first layer to match filters and strides
      shortcut = self.projection_conv(x)
      shortcut = self.projection_bn(shortcut, training)

    x = self.first_conv(x)
    x = self.first_bn(x, training)
    x = tf.nn.relu(x)

    x = self.second_conv(x)
    x = self.second_bn(x, training)

    return self.final_activation(x + shortcut)


class BottleneckResidualBlock(BlockInterface):
  """Bottleneck block variant for ResNet v1 with BN after convolutions.

  Attributes: See `BlockInterface`
  """

  def __init__(self, name='BottleneckBlock', **kwargs):
    super(BottleneckResidualBlock, self).__init__(name=name, **kwargs)

    if self.use_projection:
      # Projection shortcut is only used in the first block within a group.
      # The final conv of a Bottleneck block uses 4 times the number of filters
      # as the others, so the projection needs to match that.
      self.projection_conv = Conv2DFixedPadding(
          filters=4 * self.filters,
          kernel_size=1,
          strides=self.strides,
          data_format=self.data_format)
      self.projection_bn = batch_norm(
          data_format=self.data_format,
          batch_norm_momentum=self.batch_norm_momentum,
          use_global_batch_norm=self.use_global_batch_norm)

    self.first_conv = Conv2DFixedPadding(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        data_format=self.data_format)
    self.first_bn = batch_norm(
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

    self.second_conv = Conv2DFixedPadding(
        filters=self.filters,
        kernel_size=3,
        strides=self.strides,
        data_format=self.data_format)
    self.second_bn = batch_norm(
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

    self.third_conv = Conv2DFixedPadding(
        filters=4 * self.filters,
        kernel_size=1,
        strides=1,
        data_format=self.data_format)
    self.third_bn = batch_norm(
        init_zero=True,
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

  def call(self, inputs, training=None):
    x = inputs
    shortcut = x
    if self.use_projection:
      shortcut = self.projection_conv(x)
      shortcut = self.projection_bn(shortcut, training)

    x = self.first_conv(x)
    x = self.first_bn(x, training)
    x = tf.nn.relu(x)

    x = self.second_conv(x)
    x = self.second_bn(x, training)
    x = tf.nn.relu(x)

    x = self.third_conv(x)
    x = self.third_bn(x, training)

    return tf.nn.relu(x + shortcut)


class ResNextBlock(BlockInterface):
  """ResNeXt block variant with BN after convolutions.

  Attributes: See `BlockInterface`
  """

  def __init__(self,
               name='ResNextBlock',
               **kwargs):
    super(ResNextBlock, self).__init__(name=name, **kwargs)

    if self.use_projection:
      # Projection shortcut is only used in the first block within a group.
      # The final conv of a ResNext block uses 4 times the number of filters
      # as the others, so the projection needs to match that.
      self.projection_conv = Conv2DFixedPadding(
          filters=4 * self.filters,
          kernel_size=1,
          strides=self.strides,
          data_format=self.data_format)
      self.projection_bn = batch_norm(
          data_format=self.data_format,
          batch_norm_momentum=self.batch_norm_momentum,
          use_global_batch_norm=self.use_global_batch_norm)

    self.first_conv = Conv2DFixedPadding(
        filters=2 * self.filters,
        kernel_size=1,
        strides=1,
        data_format=self.data_format)
    self.first_bn = batch_norm(
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

    self.second_conv = GroupConv2D(
        filters=2 * self.filters,
        kernel_size=3,
        strides=self.strides,
        data_format=self.data_format,
        groups=64)
    self.second_bn = batch_norm(
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

    self.third_conv = Conv2DFixedPadding(
        filters=4 * self.filters,
        kernel_size=1,
        strides=1,
        data_format=self.data_format)
    self.third_bn = batch_norm(
        init_zero=False,
        data_format=self.data_format,
        batch_norm_momentum=self.batch_norm_momentum,
        use_global_batch_norm=self.use_global_batch_norm)

  def call(self, inputs, training=None):
    x = inputs
    shortcut = inputs
    if self.use_projection:
      shortcut = self.projection_conv(x)
      shortcut = self.projection_bn(shortcut, training)

    x = self.first_conv(x)
    x = self.first_bn(x, training)
    x = tf.nn.relu(x)

    x = self.second_conv(x)
    x = self.second_bn(x, training)
    x = tf.nn.relu(x)

    x = self.third_conv(x)
    x = self.third_bn(x, training)

    return tf.nn.relu(x + shortcut)

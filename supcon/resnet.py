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

"""ResNet implementation."""

import tensorflow.compat.v1 as tf

from supcon import blocks as blocks_lib


class _BlockGroup(tf.layers.Layer):
  """A group of Residual blocks.

  A standard ResNet model has 4 block groups, with a different number of blocks
  in each one. This generalizes the concept, allowing the creation of a block
  group with any number of blocks and any block implementation, as long as it
  implements the blocks.BlockInterface interface.

  Attributes:
    block_fn: A tf.compat.v1.layers.Layer subclass that implements a block. The
      constructor must have the arguments (filters, strides, use_projection,
      data_format).
    num_block: `int` The number of blocks to create.
    filters: `int` The number of filters passed to the block implementation.
    strides: `int` The stride passed to the first block of the group. Subsequent
      blocks all get passed stride=1.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.
    name: A name for this object.
  """

  def __init__(self,
               block_fn,
               num_blocks,
               filters,
               strides,
               data_format='channels_last',
               batch_norm_momentum=blocks_lib.BATCH_NORM_MOMENTUM,
               use_global_batch_norm=True,
               name='BlockGroup',
               **kwargs):
    super(_BlockGroup, self).__init__(name=name, **kwargs)

    self.num_blocks = num_blocks

    for i in range(self.num_blocks):
      # Use setattr rather than appending to a list, since Keras only tracks
      # sublayers that are direct members of the parent layers.
      setattr(
          self, f'block_{i}',
          block_fn(
              filters=filters,
              strides=strides if i == 0 else 1,
              use_projection=i == 0,
              data_format=data_format,
              batch_norm_momentum=batch_norm_momentum,
              use_global_batch_norm=use_global_batch_norm))

  def call(self, inputs, training=None):
    x = inputs
    for i in range(self.num_blocks):
      block = getattr(self, f'block_{i}')
      x = block(x, training)
    return x


class _BaseResidualNetwork(tf.layers.Layer):
  """An base residual network implementation.

  Attributes:
    block_fn: A `blocks.BlockInterface` implementation.
    block_group_sizes: A list of integers whose length is the number of block
      groups and whose values are the number of blocks in each group.
    width: The multiplier on the number of channels in each convolution.
    first_conv_kernel_size: The kernel size of the first convolution layer in
      the network.
    first_conv_stride: The stride of the first convolution layer in the network.
    use_initial_max_pool: Whether to include a max-pool layer between the
      initial convolution and the first residual block.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.
    name: A name for this object.
  """

  def __init__(self,
               block_fn,
               block_group_sizes,
               width=1,
               first_conv_kernel_size=7,
               first_conv_stride=2,
               use_initial_max_pool=True,
               data_format='channels_last',
               batch_norm_momentum=blocks_lib.BATCH_NORM_MOMENTUM,
               use_global_batch_norm=True,
               name='AbstractResidualNetwork',
               **kwargs):
    super(_BaseResidualNetwork, self).__init__(name=name, **kwargs)

    self.data_format = data_format
    self.num_block_groups = len(block_group_sizes)

    self.initial_conv = blocks_lib.Conv2DFixedPadding(
        filters=int(64 * width),
        kernel_size=first_conv_kernel_size,
        strides=first_conv_stride,
        data_format=data_format)
    self.initial_batchnorm = blocks_lib.batch_norm(
        data_format=data_format, use_global_batch_norm=use_global_batch_norm)
    self.initial_activation = tf.keras.layers.Activation('relu')

    self.initial_max_pool = None
    if use_initial_max_pool:
      self.initial_max_pool = tf.layers.MaxPooling2D(
          pool_size=3, strides=2, padding='SAME', data_format=data_format)

    for i, num_blocks in enumerate(block_group_sizes):
      # Use setattr rather than appending to a list, since Keras only tracks
      # sublayers that are direct members of the parent layers.
      setattr(
          self, f'block_group_{i}',
          _BlockGroup(
              filters=int(64 * 2**i * width),
              block_fn=block_fn,
              num_blocks=num_blocks,
              strides=1 if i == 0 else 2,
              data_format=data_format,
              batch_norm_momentum=batch_norm_momentum,
              use_global_batch_norm=use_global_batch_norm,
              name=f'BlockGroup{i}'))

  def call(self, inputs, training):
    x = inputs
    x = self.initial_conv(x)
    x = self.initial_batchnorm(x, training)
    x = self.initial_activation(x)
    if self.initial_max_pool:
      x = self.initial_max_pool(x)

    for i in range(self.num_block_groups):
      block_group = getattr(self, f'block_group_{i}')
      x = block_group(x, training)

    if self.data_format == 'channels_first':
      pool_axis = [2, 3]
    else:
      pool_axis = [1, 2]
    x = tf.reduce_mean(x, axis=pool_axis, keepdims=True)
    x = tf.squeeze(x, pool_axis)
    return x


class ResNetV1(_BaseResidualNetwork):
  """A ResNetV1 implementation.

  Attributes:
    depth: The number of convolutions in the network. Must be one of (18, 34,
      50, 101, 152, 200).
    width: The multiplier on the number of channels in each convolution.
    first_conv_kernel_size: The kernel size of the first convolution layer in
      the network.
    first_conv_stride: The stride of the first convolution layer in the network.
    use_initial_max_pool: Whether to include a max-pool layer between the
      initial convolution and the first residual block.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.
    name: A name for this object.
  """

  def __init__(self,
               depth,
               width=1,
               first_conv_kernel_size=7,
               first_conv_stride=2,
               use_initial_max_pool=True,
               data_format='channels_last',
               batch_norm_momentum=blocks_lib.BATCH_NORM_MOMENTUM,
               use_global_batch_norm=True,
               name='ResNetV1',
               **kwargs):
    super(ResNetV1, self).__init__(
        block_fn=self.block_fn(depth),
        block_group_sizes=self.block_group_sizes(depth),
        width=width,
        first_conv_kernel_size=first_conv_kernel_size,
        first_conv_stride=first_conv_stride,
        use_initial_max_pool=use_initial_max_pool,
        data_format=data_format,
        batch_norm_momentum=batch_norm_momentum,
        use_global_batch_norm=use_global_batch_norm,
        name=name,
        **kwargs)

  def block_fn(self, depth):
    """Returns the appropriate block implementation to use for this network.

    Args:
      depth: The number of convolutions in the network.

    Returns:
      A callable block implementation.
    """
    return (blocks_lib.ResidualBlock
            if depth < 50 else blocks_lib.BottleneckResidualBlock)

  def block_group_sizes(self, depth):
    """Returns the appropriate block group sizes for this network.

    Args:
      depth: The number of convolutions in the network.

    Returns:
      A tuple of integers, corresponding to the number of blocks in each group.

    Raises:
      ValueError if `depth` is not a valid ResNetv1 depth.
    """
    sizes_map = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3),
    }
    if depth not in sizes_map:
      raise ValueError(
          f'{depth} is not a valid ResNetV1 depth. Valid values are: '
          f'{sizes_map.keys()}')
    return sizes_map[depth]


class ResNext(_BaseResidualNetwork):
  """A ResNext implementation.

  Attributes:
    depth: The number of convolutions in the network. Must be one of (50, 101,
      152, 200).
    width: The multiplier on the number of channels in each convolution.
    first_conv_kernel_size: The kernel size of the first convolution layer in
      the network.
    first_conv_stride: The stride of the first convolution layer in the network.
    use_initial_max_pool: Whether to include a max-pool layer between the
      initial convolution and the first residual block.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
      batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.
    name: A name for this object.
  """

  def __init__(self,
               depth,
               width=1,
               first_conv_kernel_size=7,
               first_conv_stride=2,
               use_initial_max_pool=True,
               data_format='channels_last',
               batch_norm_momentum=blocks_lib.BATCH_NORM_MOMENTUM,
               use_global_batch_norm=True,
               name='ResNext',
               **kwargs):
    super(ResNext, self).__init__(
        block_fn=blocks_lib.ResNextBlock,
        block_group_sizes=self.block_group_sizes(depth),
        width=width,
        first_conv_kernel_size=first_conv_kernel_size,
        first_conv_stride=first_conv_stride,
        use_initial_max_pool=use_initial_max_pool,
        data_format=data_format,
        batch_norm_momentum=batch_norm_momentum,
        use_global_batch_norm=use_global_batch_norm,
        name=name,
        **kwargs)

  def block_group_sizes(self, depth):
    """Returns the appropriate block group sizes for this network.

    Args:
      depth: The number of convolutions in the network.

    Returns:
      A tuple of integers, corresponding to the number of blocks in each group.

    Raises:
      ValueError if `depth` is not a valid ResNext depth.
    """
    sizes_map = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3),
    }
    if depth not in sizes_map:
      raise ValueError(
          f'{depth} is not a valid ResNext depth. Valid values are: '
          f'{sizes_map.keys()}')
    return sizes_map[depth]

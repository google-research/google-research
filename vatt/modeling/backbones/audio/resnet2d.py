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

"""This code is directly inheritted from:
https://github.com/google-research/simclr/blob/master/tf2/resnet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

from absl import logging
import tensorflow as tf


BACKBONE_TO_DEPTH = {
    'RESNET18': 18,
    'RESNET34': 34,
    'RESNET50': 50,
    'RESNET101': 101
}


class BatchNormRelu(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               relu=True,
               init_zero=False,
               bn_kwargs=None,
               use_xreplica_bn=True,
               data_format='channels_last',
               name='bn_relu',
               **kwargs):
    super(BatchNormRelu, self).__init__(name=name, **kwargs)
    default_bn_kwargs = {'center': True,
                         'scale': True,
                         'momentum': 0.99,
                         'epsilon': 0.001}
    if bn_kwargs is not None:
      default_bn_kwargs.update(bn_kwargs)

    self._relu = relu
    self._use_xreplica_bn = use_xreplica_bn
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()
    if data_format == 'channels_first':
      axis = 1
    else:
      axis = -1
    if self._use_xreplica_bn:
      # TODO(srbs): Set fused=True
      # Batch normalization layers with fused=True only support 4D input
      # tensors.
      self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
          axis=axis,
          gamma_initializer=gamma_initializer,
          name='batch_norm',
          **default_bn_kwargs)
    else:
      # TODO(srbs): Set fused=True
      # Batch normalization layers with fused=True only support 4D input
      # tensors.
      self.bn = tf.keras.layers.BatchNormalization(
          axis=axis,
          fused=False,
          gamma_initializer=gamma_initializer,
          name='batch_norm',
          **default_bn_kwargs)

  def call(self, inputs, training):
    inputs = self.bn(inputs, training=training)
    if self._relu:
      inputs = tf.nn.relu(inputs)
    return inputs


class DropBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               keep_prob,
               dropblock_size,
               data_format='channels_last',
               name='drop_block',
               **kwargs):
    self.keep_prob = keep_prob
    self.dropblock_size = dropblock_size
    self.data_format = data_format
    super(DropBlock, self).__init__(name=name, **kwargs)

  def call(self, net, training):
    keep_prob = self.keep_prob
    dropblock_size = self.dropblock_size
    data_format = self.data_format
    if not training or keep_prob is None:
      return net

    logging.info('Applying DropBlock: dropblock_size %s, net.shape %s',
                 dropblock_size,
                 net.shape)

    if data_format == 'channels_last':
      _, width, height, _ = net.get_shape().as_list()
    else:
      _, _, width, height = net.get_shape().as_list()
    if width != height:
      raise ValueError('Input tensor with width!=height is not supported.')

    dropblock_size = min(dropblock_size, width)
    # seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
        width - dropblock_size + 1)**2

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
    valid_block_center = tf.logical_and(
        tf.logical_and(w_i >= int(dropblock_size // 2),
                       w_i < width - (dropblock_size - 1) // 2),
        tf.logical_and(h_i >= int(dropblock_size // 2),
                       h_i < width - (dropblock_size - 1) // 2))

    valid_block_center = tf.expand_dims(valid_block_center, 0)
    valid_block_center = tf.expand_dims(
        valid_block_center, -1 if data_format == 'channels_last' else 0)

    randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
    block_pattern = (
        1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
            (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if dropblock_size == width:
      block_pattern = tf.reduce_min(
          block_pattern,
          axis=[1, 2] if data_format == 'channels_last' else [2, 3],
          keepdims=True)
    else:
      if data_format == 'channels_last':
        ksize = [1, dropblock_size, dropblock_size, 1]
      else:
        ksize = [1, 1, dropblock_size, dropblock_size]
      block_pattern = -tf.nn.max_pool(
          -block_pattern,
          ksize=ksize,
          strides=[1, 1, 1, 1],
          padding='SAME',
          data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

    percent_ones = (
        tf.cast(tf.reduce_sum((block_pattern)), tf.float32) /
        tf.cast(tf.size(block_pattern), tf.float32))

    net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
        block_pattern, net.dtype)
    return net


class FixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, kernel_size,
               data_format='channels_last',
               name='fixed_padding',
               **kwargs):
    super(FixedPadding, self).__init__(name=name, **kwargs)
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
               data_format='channels_last',
               name='conv_pad',
               **kwargs):
    super(Conv2dFixedPadding, self).__init__(name=name, **kwargs)
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
        data_format=data_format,
        name='conv2d')

  def call(self, inputs, training):
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv2d(inputs, training=training)


class IdentityLayer(tf.keras.layers.Layer):

  def call(self, inputs, training):
    return tf.identity(inputs)


class SK_Conv2D(tf.keras.layers.Layer):  # pylint: disable=invalid-name
  """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""

  def __init__(self,
               filters,
               strides,
               sk_ratio,
               min_dim=32,
               data_format='channels_last',
               bn_kwargs=None,
               use_xreplica_bn=True,
               name='sk_conv2d',
               **kwargs):
    super(SK_Conv2D, self).__init__(name=name, **kwargs)
    self.data_format = data_format
    self.filters = filters
    self.sk_ratio = sk_ratio
    self.min_dim = min_dim

    # Two stream convs (using split and both are 3x3).
    self.conv2d_fixed_padding = Conv2dFixedPadding(
        filters=2 * filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)
    self.batch_norm_relu = BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                                         bn_kwargs=bn_kwargs,
                                         data_format=data_format)

    # Mixing weights for two streams.
    mid_dim = max(int(filters * sk_ratio), min_dim)
    self.conv2d_0 = tf.keras.layers.Conv2D(
        filters=mid_dim,
        kernel_size=1,
        strides=1,
        use_bias=False,
        data_format=data_format)
    self.batch_norm_relu_1 = BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                                           bn_kwargs=bn_kwargs,
                                           data_format=data_format)
    self.conv2d_1 = tf.keras.layers.Conv2D(
        filters=2 * filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        data_format=data_format)

  def call(self, inputs, training):
    channel_axis = 1 if self.data_format == 'channels_first' else 3
    pooling_axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

    # Two stream convs (using split and both are 3x3).
    inputs = self.conv2d_fixed_padding(inputs, training=training)
    inputs = self.batch_norm_relu(inputs, training=training)
    inputs = tf.stack(tf.split(inputs, num_or_size_splits=2, axis=channel_axis))

    # Mixing weights for two streams.
    global_features = tf.reduce_mean(
        tf.reduce_sum(inputs, axis=0), pooling_axes, keepdims=True)
    global_features = self.conv2d_0(global_features, training=training)
    global_features = self.batch_norm_relu_1(global_features, training=training)
    mixing = self.conv2d_1(global_features, training=training)
    mixing = tf.stack(tf.split(mixing, num_or_size_splits=2, axis=channel_axis))
    mixing = tf.nn.softmax(mixing, axis=0)

    return tf.reduce_sum(inputs * mixing, axis=0)


class SE_Layer(tf.keras.layers.Layer):  # pylint: disable=invalid-name
  """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""

  def __init__(self,
               filters,
               se_ratio,
               data_format='channels_last',
               name='se_layer',
               **kwargs):
    super(SE_Layer, self).__init__(name=name, **kwargs)
    self.data_format = data_format
    self.se_reduce = tf.keras.layers.Conv2D(
        max(1, int(filters * se_ratio)),
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        data_format=data_format,
        use_bias=True)
    self.se_expand = tf.keras.layers.Conv2D(
        None,  # This is filled later in build().
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        data_format=data_format,
        use_bias=True)

  def build(self, input_shape):
    self.se_expand.filters = input_shape[-1]
    super(SE_Layer, self).build(input_shape)

  def call(self, inputs, training):
    spatial_dims = [2, 3] if self.data_format == 'channels_first' else [1, 2]
    se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)
    se_tensor = self.se_expand(tf.nn.relu(self.se_reduce(se_tensor)))
    return tf.sigmoid(se_tensor) * inputs


class ResidualBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               strides,
               sk_ratio=0.,
               se_ratio=0.,
               use_projection=False,
               data_format='channels_last',
               bn_kwargs=None,
               use_xreplica_bn=True,
               dropblock_keep_prob=None,
               dropblock_size=None,
               name='res_block',
               **kwargs):
    super(ResidualBlock, self).__init__(name=name, **kwargs)
    del dropblock_keep_prob
    del dropblock_size
    self.sk_ratio = sk_ratio
    self.se_ratio = se_ratio
    self.conv2d_bn_layers = []
    self.shortcut_layers = []
    if use_projection:
      if self.sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
        if strides > 1:
          self.shortcut_layers.append(FixedPadding(2, data_format))
        self.shortcut_layers.append(
            tf.keras.layers.AveragePooling2D(
                pool_size=2,
                strides=strides,
                padding='SAME' if strides == 1 else 'VALID',
                data_format=data_format))
        self.shortcut_layers.append(
            Conv2dFixedPadding(
                filters=filters,
                kernel_size=1,
                strides=1,
                data_format=data_format,
                name='proj/conv_pad'))
      else:
        self.shortcut_layers.append(
            Conv2dFixedPadding(
                filters=filters,
                kernel_size=1,
                strides=strides,
                data_format=data_format,
                name='proj/conv_pad'))
      self.shortcut_layers.append(
          BatchNormRelu(relu=False,
                        use_xreplica_bn=use_xreplica_bn,
                        bn_kwargs=bn_kwargs,
                        data_format=data_format,
                        name='proj/bn_relu'))

    self.conv2d_bn_layers.append(
        Conv2dFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
            name='conv_pad_1'))
    self.conv2d_bn_layers.append(BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                                               bn_kwargs=bn_kwargs,
                                               data_format=data_format,
                                               name='bn_relu_1'))
    self.conv2d_bn_layers.append(
        Conv2dFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=1,
            data_format=data_format,
            name='conv_pad_2'))
    self.conv2d_bn_layers.append(
        BatchNormRelu(relu=False,
                      init_zero=True,
                      use_xreplica_bn=use_xreplica_bn,
                      bn_kwargs=bn_kwargs,
                      data_format=data_format,
                      name='bn_relu_2'))
    if self.se_ratio > 0:
      self.se_layer = SE_Layer(filters, self.se_ratio, data_format=data_format)

  def call(self, inputs, training):
    shortcut = inputs
    for layer in self.shortcut_layers:
      # Projection shortcut in first layer to match filters and strides
      shortcut = layer(shortcut, training=training)

    for layer in self.conv2d_bn_layers:
      inputs = layer(inputs, training=training)

    if self.se_ratio > 0:
      inputs = self.se_layer(inputs, training=training)

    return tf.nn.relu(inputs + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
  """BottleneckBlock."""

  def __init__(self,
               filters,
               strides,
               sk_ratio=0.,
               se_ratio=0.,
               use_projection=False,
               data_format='channels_last',
               bn_kwargs=None,
               use_xreplica_bn=True,
               dropblock_keep_prob=None,
               dropblock_size=None,
               name='bott_block',
               **kwargs):
    super(BottleneckBlock, self).__init__(name=name, **kwargs)
    self.sk_ratio = sk_ratio
    self.se_ratio = se_ratio
    self.projection_layers = []
    if use_projection:
      filters_out = 4 * filters
      if self.sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
        if strides > 1:
          self.projection_layers.append(FixedPadding(2, data_format))
        self.projection_layers.append(
            tf.keras.layers.AveragePooling2D(
                pool_size=2,
                strides=strides,
                padding='SAME' if strides == 1 else 'VALID',
                data_format=data_format))
        self.projection_layers.append(
            Conv2dFixedPadding(
                filters=filters_out,
                kernel_size=1,
                strides=1,
                data_format=data_format,
                name='proj/conv_pad'))
      else:
        self.projection_layers.append(
            Conv2dFixedPadding(
                filters=filters_out,
                kernel_size=1,
                strides=strides,
                data_format=data_format,
                name='proj/conv_pad'))
      self.projection_layers.append(
          BatchNormRelu(relu=False,
                        use_xreplica_bn=use_xreplica_bn,
                        bn_kwargs=bn_kwargs,
                        data_format=data_format,
                        name='proj/bn_relu'))
    self.shortcut_dropblock = DropBlock(
        data_format=data_format,
        keep_prob=dropblock_keep_prob,
        dropblock_size=dropblock_size,
        name='drop_block_1')

    self.conv_relu_dropblock_layers = []

    self.conv_relu_dropblock_layers.append(
        Conv2dFixedPadding(
            filters=filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
            name='conv_pad_1'))
    self.conv_relu_dropblock_layers.append(
        BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                      bn_kwargs=bn_kwargs,
                      data_format=data_format,
                      name='bn_relu_1'))
    self.conv_relu_dropblock_layers.append(
        DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            name='drop_block_2'))

    if self.sk_ratio > 0:
      self.conv_relu_dropblock_layers.append(
          SK_Conv2D(filters,
                    strides,
                    self.sk_ratio,
                    bn_kwargs=bn_kwargs,
                    use_xreplica_bn=use_xreplica_bn,
                    data_format=data_format,
                    name='sk_conv_1'))
    else:
      self.conv_relu_dropblock_layers.append(
          Conv2dFixedPadding(
              filters=filters,
              kernel_size=3,
              strides=strides,
              data_format=data_format,
              name='conv_pad_2'))
      self.conv_relu_dropblock_layers.append(
          BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                        bn_kwargs=bn_kwargs,
                        data_format=data_format,
                        name='bn_relu_2'))
    self.conv_relu_dropblock_layers.append(
        DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            name='drop_block_3'))

    self.conv_relu_dropblock_layers.append(
        Conv2dFixedPadding(
            filters=4 * filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
            name='conv_pad_3'))
    self.conv_relu_dropblock_layers.append(
        BatchNormRelu(relu=False,
                      init_zero=True,
                      use_xreplica_bn=use_xreplica_bn,
                      bn_kwargs=bn_kwargs,
                      data_format=data_format,
                      name='bn_relu_3'))
    self.conv_relu_dropblock_layers.append(
        DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            name='drop_block_4'))

    if self.se_ratio > 0:
      self.conv_relu_dropblock_layers.append(
          SE_Layer(filters,
                   self.se_ratio,
                   data_format=data_format,
                   name='se_layer_1'))

  def call(self, inputs, training):
    shortcut = inputs
    for layer in self.projection_layers:
      shortcut = layer(shortcut, training=training)
    shortcut = self.shortcut_dropblock(shortcut, training=training)

    for layer in self.conv_relu_dropblock_layers:
      inputs = layer(inputs, training=training)

    return tf.nn.relu(inputs + shortcut)


class BlockGroup(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               block_fn,
               blocks,
               strides,
               data_format='channels_last',
               bn_kwargs=None,
               use_xreplica_bn=True,
               dropblock_keep_prob=None,
               dropblock_size=None,
               **kwargs):
    self._name = kwargs.get('name')
    super(BlockGroup, self).__init__(**kwargs)
    self.layers = []
    self.layers.append(
        block_fn(
            filters,
            strides,
            use_projection=True,
            data_format=data_format,
            bn_kwargs=bn_kwargs,
            use_xreplica_bn=use_xreplica_bn,
            dropblock_keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            name='block_1'))

    for n in range(1, blocks):
      self.layers.append(
          block_fn(
              filters,
              1,
              data_format=data_format,
              bn_kwargs=bn_kwargs,
              use_xreplica_bn=use_xreplica_bn,
              dropblock_keep_prob=dropblock_keep_prob,
              dropblock_size=dropblock_size,
              name='block_{}'.format(n+1)))

  def call(self, inputs, training):
    for layer in self.layers:
      inputs = layer(inputs, training=training)
    return tf.identity(inputs, self._name)


class Resnet(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               block_fn,
               layers,
               width_multiplier=1,
               cifar_stem=False,
               data_format='channels_last',
               batch_norm_decay=0.99,
               batch_norm_epsilon=0.001,
               batch_norm_scale=True,
               use_xreplica_bn=True,
               sk_ratio=0.,
               stop_gradient_point=-1,
               dropblock_keep_probs=None,
               dropblock_size=None,
               model_name='resnet2d',
               **kwargs):
    super(Resnet, self).__init__(name=model_name, **kwargs)
    self.data_format = data_format
    self.stop_gradient_point = stop_gradient_point
    if dropblock_keep_probs is None:
      dropblock_keep_probs = [None] * 4
    if not isinstance(dropblock_keep_probs,
                      list) or len(dropblock_keep_probs) != 4:
      raise ValueError('dropblock_keep_probs is not valid:',
                       dropblock_keep_probs)
    bn_kwargs = {'momentum': batch_norm_decay,
                 'epsilon': batch_norm_epsilon,
                 'scale': batch_norm_scale}
    trainable = self.stop_gradient_point == -1
    self.initial_conv_relu_max_pool = []
    if cifar_stem:
      self.initial_conv_relu_max_pool.append(
          Conv2dFixedPadding(
              filters=64 * width_multiplier,
              kernel_size=3,
              strides=1,
              data_format=data_format,
              trainable=trainable,
              name='stem/conv_pad'))
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_conv', trainable=trainable))
      self.initial_conv_relu_max_pool.append(
          BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                        bn_kwargs=bn_kwargs,
                        data_format=data_format,
                        trainable=trainable,
                        name='stem/bn_relu'))
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_max_pool', trainable=trainable))
    else:
      if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(
                filters=64 * width_multiplier // 2,
                kernel_size=3,
                strides=2,
                data_format=data_format,
                trainable=trainable,
                name='stem/sk_conv_pad_1'))
        self.initial_conv_relu_max_pool.append(
            BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                          bn_kwargs=bn_kwargs,
                          data_format=data_format,
                          trainable=trainable,
                          name='stem/sk_bn_relu_1'))
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(
                filters=64 * width_multiplier // 2,
                kernel_size=3,
                strides=1,
                data_format=data_format,
                trainable=trainable,
                name='stem/sk_conv_pad_2'))
        self.initial_conv_relu_max_pool.append(
            BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                          bn_kwargs=bn_kwargs,
                          data_format=data_format,
                          trainable=trainable,
                          name='stem/sk_bn_relu_2'))
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(
                filters=64 * width_multiplier,
                kernel_size=3,
                strides=1,
                data_format=data_format,
                trainable=trainable,
                name='stem/sk_conv_pad_3'))
      else:
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(
                filters=64 * width_multiplier,
                kernel_size=7,
                strides=2,
                data_format=data_format,
                trainable=trainable,
                name='stem/conv_pad'))
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_conv', trainable=trainable))
      self.initial_conv_relu_max_pool.append(
          BatchNormRelu(use_xreplica_bn=use_xreplica_bn,
                        bn_kwargs=bn_kwargs,
                        data_format=data_format,
                        trainable=trainable,
                        name='stem/bn_relu'))

      self.initial_conv_relu_max_pool.append(
          tf.keras.layers.MaxPooling2D(
              pool_size=3,
              strides=2,
              padding='SAME',
              data_format=data_format,
              trainable=trainable))
      self.initial_conv_relu_max_pool.append(
          IdentityLayer(name='initial_max_pool', trainable=trainable))

    self.block_groups = []
    # TODO(srbs): This impl is different from the original one in the case where
    # fine_tune_after_block != 4. In that case earlier BN stats were getting
    # updated. Now they will not be. Check with Ting to make sure this is ok.
    if self.stop_gradient_point == 0:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=64 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[0],
            strides=1,
            name='block_group_1',
            data_format=data_format,
            bn_kwargs=bn_kwargs,
            use_xreplica_bn=use_xreplica_bn,
            dropblock_keep_prob=dropblock_keep_probs[0],
            dropblock_size=dropblock_size,
            trainable=trainable))

    if self.stop_gradient_point == 1:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=128 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[1],
            strides=2,
            name='block_group_2',
            data_format=data_format,
            bn_kwargs=bn_kwargs,
            use_xreplica_bn=use_xreplica_bn,
            dropblock_keep_prob=dropblock_keep_probs[1],
            dropblock_size=dropblock_size,
            trainable=trainable))

    if self.stop_gradient_point == 2:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=256 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[2],
            strides=2,
            name='block_group_3',
            data_format=data_format,
            bn_kwargs=bn_kwargs,
            use_xreplica_bn=use_xreplica_bn,
            dropblock_keep_prob=dropblock_keep_probs[2],
            dropblock_size=dropblock_size,
            trainable=trainable))

    if self.stop_gradient_point == 3:
      trainable = True

    self.block_groups.append(
        BlockGroup(
            filters=512 * width_multiplier,
            block_fn=block_fn,
            blocks=layers[3],
            strides=2,
            name='block_group_4',
            data_format=data_format,
            bn_kwargs=bn_kwargs,
            use_xreplica_bn=use_xreplica_bn,
            dropblock_keep_prob=dropblock_keep_probs[3],
            dropblock_size=dropblock_size,
            trainable=trainable))

    if self.stop_gradient_point == 4:
      # This case doesn't really matter.
      trainable = True

    pool_dims = self._get_pool_dims(data_format)
    self.pool = functools.partial(tf.reduce_mean, axis=pool_dims)

  def _get_pool_dims(self, data_format):
    if data_format == 'channels_last':
      return [1, 2]
    else:
      return [2, 3]

  def call(self, inputs, training):
    end_points = {}
    for layer in self.initial_conv_relu_max_pool:
      inputs = layer(inputs, training=training)

    for i, layer in enumerate(self.block_groups):
      if self.stop_gradient_point == i:
        inputs = tf.stop_gradient(inputs)
      inputs = layer(inputs, training=training)
      end_points[f'unit_out_{i}'] = inputs
    if self.stop_gradient_point == 4:
      inputs = tf.stop_gradient(inputs)
    end_points['last_conv'] = inputs

    features_pooled = self.pool(inputs)
    return features_pooled, end_points


def Resnet2dBase(backbone='resnet50',
                 width_multiplier=1,
                 cifar_stem=False,
                 data_format='channels_last',
                 stop_gradient_point=-1,
                 batch_norm_decay=0.99,
                 batch_norm_epsilon=0.001,
                 batch_norm_scale=True,
                 use_xreplica_bn=True,
                 dropblock_keep_probs=None,
                 dropblock_size=None,
                 **kwargs):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {
          'block': ResidualBlock,
          'layers': [2, 2, 2, 2]
      },
      34: {
          'block': ResidualBlock,
          'layers': [3, 4, 6, 3]
      },
      50: {
          'block': BottleneckBlock,
          'layers': [3, 4, 6, 3]
      },
      101: {
          'block': BottleneckBlock,
          'layers': [3, 4, 23, 3]
      },
      152: {
          'block': BottleneckBlock,
          'layers': [3, 8, 36, 3]
      },
      200: {
          'block': BottleneckBlock,
          'layers': [3, 24, 36, 3]
      }
  }
  resnet_depth = BACKBONE_TO_DEPTH[backbone.upper()]
  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)
  model_name = 'resnet_{}'.format(resnet_depth)

  return Resnet(
      block_fn=model_params[resnet_depth]['block'],
      layers=model_params[resnet_depth]['layers'],
      width_multiplier=width_multiplier,
      cifar_stem=cifar_stem,
      dropblock_keep_probs=dropblock_keep_probs,
      dropblock_size=dropblock_size,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon,
      batch_norm_scale=batch_norm_scale,
      use_xreplica_bn=use_xreplica_bn,
      stop_gradient_point=stop_gradient_point,
      data_format=data_format,
      model_name=model_name)

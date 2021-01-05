# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""ResNet Model.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import tensorflow as tf

MOVING_AVERAGE_DECAY = 0.9
EPSILON = 1e-5


def fixed_padding(inputs, kernel_size, data_format='channels_first'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
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


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format='channels_first'):
  """Updating batch norm to work with regularization."""
  inputs = tf.compat.v1.layers.batch_normalization(
      inputs=inputs,
      axis=1 if data_format == 'channels_first' else 3,
      momentum=MOVING_AVERAGE_DECAY,
      epsilon=EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=False,
      gamma_initializer=(tf.zeros_initializer()
                         if init_zero else tf.ones_initializer()))

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_first',
                         name=None):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs:  Input tensor, float32 or bfloat16 of size [batch, channels, height,
      width].
    filters: Int specifying number of filters for the first two convolutions.
    kernel_size: Int designating size of kernel to be used in the convolution.
    strides: Int specifying the stride. If stride >1, the input is downsampled.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor of size [batch, filters, height_out, width_out]

  Raises:
    ValueError: If the data_format provided is not a valid string.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.compat.v1.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      data_format=data_format,
      name=name)


def residual_block_(inputs,
                    filters,
                    is_training,
                    strides,
                    use_projection=False,
                    data_format='channels_first',
                    name=''):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs:  Input tensor, float32 or bfloat16 of size [batch, channels, height,
      width].
    filters: Int specifying number of filters for the first two convolutions.
    is_training: Boolean specifying whether the model is training.
    strides: Int specifying the stride. If stride >1, the input is downsampled.
    use_projection: Boolean for whether the layer should use a projection
      shortcut Often, use_projection=True for the first block of a block group.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    end_point = 'residual_projection_%s' % name
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        name=end_point)
    shortcut = batch_norm_relu(
        shortcut, is_training, relu=False, data_format=data_format)

  end_point = 'residual_1_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      name=end_point)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  end_point = 'residual_2_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format,
      name=end_point)
  inputs = batch_norm_relu(
      inputs, is_training, relu=False, init_zero=True, data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block_(inputs,
                      filters,
                      is_training,
                      strides,
                      use_projection=False,
                      data_format='channels_first',
                      name=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: Input tensor, float32 or bfloat16 of size [batch, channels, height,
      width].
    filters: Int specifying number of filters for the first two convolutions.
    is_training: Boolean specifying whether the model is training.
    strides: Int specifying the stride. If stride >1, the input is downsampled.
    use_projection: Boolean for whether the layer should use a projection
      shortcut Often, use_projection=True for the first block of a block group.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor.
  """
  shortcut = inputs

  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    end_point = 'bottleneck_projection_%s' % name
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        name=end_point)
    shortcut = batch_norm_relu(
        shortcut, is_training, relu=False, data_format=data_format)

  end_point = 'bottleneck_1_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      name=end_point)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  end_point = 'bottleneck_2_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      name=end_point)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  end_point = 'bottleneck_3_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      name=end_point)
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
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
      greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: String specifying the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  with tf.name_scope(name):
    end_point = 'block_group_projection_%s' % name
    # Only the first block per block_group uses projection shortcut and strides.
    inputs = block_fn(
        inputs,
        filters,
        is_training,
        strides,
        use_projection=True,
        data_format=data_format,
        name=end_point)

    for n in range(1, blocks):
      with tf.name_scope('block_group_%d' % n):
        end_point = '%s_%d_1' % (name, n)
        inputs = block_fn(
            inputs,
            filters,
            is_training,
            1,
            data_format=data_format,
            name=end_point)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn,
                        num_blocks,
                        num_classes,
                        data_format='channels_first',
                        name=None):
  """Generator for ResNet v1 models.

  Args:
    block_fn: String that defines whether to use a `residual_block` or
      `bottleneck_block`.
    num_blocks: list of Ints that denotes number of blocks to include in each
      block group. Each group consists of blocks that take inputs of the same
      resolution.
    num_classes: Int number of possible classes for image classification.
    data_format: String either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    name: String that specifies name for model layer.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """

  def model(inputs, is_training):
    """Creation of the model graph."""
    with tf.compat.v1.variable_scope(name, 'resnet_model'):
      inputs = conv2d_fixed_padding(
          inputs=inputs,
          filters=64,
          kernel_size=7,
          strides=2,
          data_format=data_format,
          name='initial_conv')
      inputs = tf.identity(inputs, 'initial_conv')
      inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

      inputs = tf.compat.v1.layers.max_pooling2d(
          inputs=inputs,
          pool_size=3,
          strides=2,
          padding='SAME',
          data_format=data_format,
          name='initial_max_pool')
      inputs = tf.identity(inputs, 'initial_max_pool')

      inputs = block_group(
          inputs=inputs,
          filters=64,
          block_fn=block_fn,
          blocks=num_blocks[0],
          strides=1,
          is_training=is_training,
          name='block_group1',
          data_format=data_format)
      inputs = block_group(
          inputs=inputs,
          filters=128,
          block_fn=block_fn,
          blocks=num_blocks[1],
          strides=2,
          is_training=is_training,
          name='block_group2',
          data_format=data_format)
      inputs = block_group(
          inputs=inputs,
          filters=256,
          block_fn=block_fn,
          blocks=num_blocks[2],
          strides=2,
          is_training=is_training,
          name='block_group3',
          data_format=data_format)
      inputs = block_group(
          inputs=inputs,
          filters=512,
          block_fn=block_fn,
          blocks=num_blocks[3],
          strides=2,
          is_training=is_training,
          name='block_group4',
          data_format=data_format)

      pool_size = (inputs.shape[1], inputs.shape[2])
      inputs = tf.compat.v1.layers.average_pooling2d(
          inputs=inputs,
          pool_size=pool_size,
          strides=1,
          padding='VALID',
          data_format=data_format,
          name='final_avg_pool')
      inputs = tf.identity(inputs, 'final_avg_pool')
      inputs = tf.reshape(inputs,
                          [-1, 2048 if block_fn is bottleneck_block_ else 512])
      inputs = tf.compat.v1.layers.dense(
          inputs=inputs,
          units=num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=.01),
          name='final_dense')
      inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def resnet_v1_(resnet_depth, num_classes, data_format='channels_first'):
  """Returns the ResNet model for a given size and number of output classes.

  Args:
    resnet_depth: Int number of blocks in the architecture.
    num_classes: Int number of possible classes for image classification.
    data_format: String specifying either "channels_first" for `[batch,
      channels, height, width]` or "channels_last for `[batch, height, width,
      channels]`.

  Raises:
    ValueError: If the resnet_depth int is not in the model_params dictionary.
  """
  model_params = {
      18: {
          'block': residual_block_,
          'layers': [2, 2, 2, 2]
      },
      34: {
          'block': residual_block_,
          'layers': [3, 4, 6, 3]
      },
      50: {
          'block': bottleneck_block_,
          'layers': [3, 4, 6, 3]
      },
      101: {
          'block': bottleneck_block_,
          'layers': [3, 4, 23, 3]
      },
      152: {
          'block': bottleneck_block_,
          'layers': [3, 8, 36, 3]
      },
      200: {
          'block': bottleneck_block_,
          'layers': [3, 24, 36, 3]
      }
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return resnet_v1_generator(params['block'], params['layers'], num_classes,
                             data_format)

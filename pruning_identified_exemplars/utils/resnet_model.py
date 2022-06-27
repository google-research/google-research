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

"""This is a ResNet-50 model.

"""
from six.moves import range
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.pruning_tools import pruning_layer
from tensorflow.python.ops import variables as tf_variables  # pylint: disable=g-direct-tensorflow-import


def get_model_variables(getter,
                        name,
                        rename=None,
                        shape=None,
                        dtype=None,
                        initializer=None,
                        regularizer=None,
                        trainable=True,
                        collections=None,
                        caching_device=None,
                        partitioner=None,
                        use_resource=None,
                        **_):
  """This ensures variables are retrieved in a consistent way for core layers."""
  name_components = name.split('/')
  short_name = name_components[-1]

  # rename is an optional dict of strings defining alteration of tensor names
  if rename and short_name in rename:
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return tf_variables.model_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      collections=collections,
      trainable=trainable,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=getter,
      use_resource=use_resource)


def variable_getter(rename=None):
  """Ensures scope is respected and consistently used."""

  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return get_model_variables(getter, *args, **kwargs)

  return layer_variable_getter


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
      momentum=0.9,
      epsilon=1e-5,
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
                         pruning_method=None,
                         data_format='channels_first',
                         is_training=True,
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
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove. Default of None to indicate no pruning.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    is_training: boolean for whether model is in training or eval mode.
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor of size [batch, filters, height_out, width_out]

  Raises:
    ValueError: If the data_format provided is not a valid string.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  layer_variable_getter = variable_getter({
      'bias': 'biases',
      'kernel': 'weights',
  })
  if pruning_method:
    with tf.variable_scope(
        name, 'Conv', [pruning_method],
        custom_getter=layer_variable_getter) as sc:
      if data_format == 'channels_last':
        data_format_channels = 'NHWC'
      elif data_format == 'channels_first':
        data_format_channels = 'NCHW'
      else:
        raise ValueError('Not a valid channel string:', data_format)
      return pruning_layer.masked_conv2d(
          inputs=inputs,
          num_outputs=filters,
          kernel_size=kernel_size,
          stride=strides,
          padding=('SAME' if strides == 1 else 'VALID'),
          data_format=data_format_channels,
          rate=1,
          activation_fn=None,
          weights_initializer=tf.variance_scaling_initializer(),
          weights_regularizer=None,
          normalizer_fn=None,
          normalizer_params=None,
          biases_initializer=None,
          biases_regularizer=None,
          outputs_collections=None,
          trainable=is_training,
          scope=sc)
  else:
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        name=name)


def residual_block_(inputs,
                    filters,
                    is_training,
                    strides,
                    use_projection=False,
                    pruning_method=None,
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
   pruning_method: String that specifies the pruning method used to identify
      which weights to remove. Default of None to indicate no pruning.
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
        pruning_method=pruning_method,
        is_training=is_training,
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
      is_training=is_training,
      pruning_method=pruning_method,
      data_format=data_format,
      name=end_point)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  end_point = 'residual_2_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      is_training=is_training,
      pruning_method=pruning_method,
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
                      pruning_method=None,
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
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove. Default of None to indicate no pruning.
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
        pruning_method=pruning_method,
        data_format=data_format,
        is_training=is_training,
        name=end_point)
    shortcut = batch_norm_relu(
        shortcut, is_training, relu=False, data_format=data_format)

  end_point = 'bottleneck_1_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      pruning_method=pruning_method,
      data_format=data_format,
      is_training=is_training,
      name=end_point)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  end_point = 'bottleneck_2_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      pruning_method=pruning_method,
      data_format=data_format,
      is_training=is_training,
      name=end_point)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  end_point = 'bottleneck_3_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      pruning_method=pruning_method,
      data_format=data_format,
      is_training=is_training,
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
                pruning_method=None,
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
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove. Default of None to indicate no pruning.
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
        pruning_method=pruning_method,
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
            pruning_method=pruning_method,
            data_format=data_format,
            name=end_point)

  return tf.identity(inputs, name)


def resnet_50_generator(block_fn,
                        lst_layers,
                        num_classes,
                        pruning_method=None,
                        data_format='channels_first',
                        name=None):
  """Generator for ResNet v1 models.

  Args:
    block_fn: String that defines whether to use a `residual_block` or
      `bottleneck_block`.
    lst_layers: list of Ints that denotes number of blocks to include in each
      block group. Each group consists of blocks that take inputs of the same
      resolution.
    num_classes: Int number of possible classes for image classification.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    data_format: String either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    name: String that specifies name for model layer.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """

  def model(inputs, is_training):
    """Creation of the model graph."""
    with tf.variable_scope(name, 'resnet_model'):
      inputs = conv2d_fixed_padding(
          inputs=inputs,
          filters=64,
          kernel_size=7,
          strides=2,
          pruning_method=pruning_method,
          data_format=data_format,
          name='initial_conv')
      inputs = tf.identity(inputs, 'initial_conv')
      inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

      inputs = tf.layers.max_pooling2d(
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
          blocks=lst_layers[0],
          strides=1,
          is_training=is_training,
          name='block_group1',
          pruning_method=pruning_method,
          data_format=data_format)
      inputs = block_group(
          inputs=inputs,
          filters=128,
          block_fn=block_fn,
          blocks=lst_layers[1],
          strides=2,
          is_training=is_training,
          name='block_group2',
          pruning_method=pruning_method,
          data_format=data_format)
      inputs = block_group(
          inputs=inputs,
          filters=256,
          block_fn=block_fn,
          blocks=lst_layers[2],
          strides=2,
          is_training=is_training,
          name='block_group3',
          pruning_method=pruning_method,
          data_format=data_format)
      inputs = block_group(
          inputs=inputs,
          filters=512,
          block_fn=block_fn,
          blocks=lst_layers[3],
          strides=2,
          is_training=is_training,
          name='block_group4',
          pruning_method=pruning_method,
          data_format=data_format)

      pool_size = (inputs.shape[1], inputs.shape[2])
      inputs = tf.layers.average_pooling2d(
          inputs=inputs,
          pool_size=pool_size,
          strides=1,
          padding='VALID',
          data_format=data_format,
          name='final_avg_pool')
      inputs = tf.identity(inputs, 'final_avg_pool')
      inputs = tf.reshape(inputs, [-1, 2048])
      inputs = tf.layers.dense(
          inputs=inputs,
          units=num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=.01),
          name='final_dense')
      inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def resnet_50(num_classes, data_format='channels_first', pruning_method=None):
  """Returns the ResNet model for a given size and number of output classes."""
  return resnet_50_generator(
      block_fn=bottleneck_block_,
      lst_layers=[3, 4, 6, 3],
      num_classes=num_classes,
      pruning_method=pruning_method,
      data_format=data_format)

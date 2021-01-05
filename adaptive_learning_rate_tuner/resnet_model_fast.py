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

"""Defines the resnet model.

Adapted from
https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet.
The following code is based on its v1 version.

"""

import tensorflow.compat.v1 as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
NUM_CLASSES = 10


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format, name=''):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.compat.v1.layers.batch_normalization(
      inputs=inputs,
      axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=training,
      fused=True,
      name=name)


# add name later if necessary
def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
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
    padded_inputs = tf.pad(
        tensor=inputs,
        paddings=[[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        tensor=inputs,
        paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format,
                         name):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.compat.v1.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      reuse=tf.AUTO_REUSE,
      kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
      data_format=data_format,
      name=name)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format, name):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference mode.
      Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
    name: Block name.

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  first_name = name + 'first'
  inputs = batch_norm(
      inputs, training, data_format, name=first_name + 'batch_norm')
  inputs = tf.nn.relu(inputs, name=first_name + 'relu')

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs, name=first_name + 'proj')

  second_name = name + 'second'
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      name=second_name + 'input')

  inputs = batch_norm(
      inputs, training, data_format, name=second_name + 'batch_norm')
  inputs = tf.nn.relu(inputs, name=second_name + 'relu')

  third_name = name + 'third'
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format,
      name=third_name + 'input')

  return inputs + shortcut


def block_layer(inputs,
                filters,
                bottleneck,
                block_fn,
                blocks,
                strides,
                training,
                name,
                data_format,
                shortcut=True):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or [batch,
      height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the model.
      Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
    shortcut: Whether to use projection shortcut in the first block.

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs, name):
    return conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        name=name)

  # Only the first block per block_layer uses projection_shortcut and strides.
  # Skip the projection shortcut in the first block layer.
  shortcut_fn = projection_shortcut if shortcut else None
  inputs = block_fn(
      inputs,
      filters,
      training,
      shortcut_fn,
      strides,
      data_format,
      name=name + 'input')

  for j in range(1, blocks):
    inputs = block_fn(
        inputs,
        filters,
        training,
        None,
        1,
        data_format,
        name=name + 'block' + str(j))

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self,
               resnet_size,
               bottleneck,
               num_classes,
               num_filters,
               kernel_size,
               conv_stride,
               first_pool_size,
               first_pool_stride,
               block_sizes,
               block_strides,
               resnet_version=DEFAULT_VERSION,
               data_format=None,
               dtype=DEFAULT_DTYPE):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer of the
        model. This number is then doubled for each subsequent block layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer. If
        none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used if
        first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None). If
        set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = ('channels_first'
                     if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.dtype = dtype
    self.pre_activation = resnet_version == 2

  def _custom_dtype_getter(self,  # pylint: disable=keyword-arg-before-vararg
                           getter,
                           name,
                           shape=None,
                           dtype=DEFAULT_DTYPE,
                           *args,
                           **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.compat.v1.variable_scope(
        'resnet_model',
        custom_getter=self._custom_dtype_getter,
        reuse=tf.AUTO_REUSE)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

      inputs = conv2d_fixed_padding(
          inputs=inputs,
          filters=self.num_filters,
          kernel_size=self.kernel_size,
          strides=self.conv_stride,
          data_format=self.data_format,
          name='initial_input')
      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

      if self.first_pool_size:
        inputs = tf.compat.v1.layers.max_pooling2d(
            inputs=inputs,
            pool_size=self.first_pool_size,
            strides=self.first_pool_stride,
            padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        # We now have 4 block layers, but the last does not
        # double the number of filters.
        # We also skip the projection shortcut in the first block layer.
        num_filters = self.num_filters * min((2**i), 4)
        shortcut = i != 0
        inputs = block_layer(
            inputs=inputs,
            filters=num_filters,
            bottleneck=self.bottleneck,
            block_fn=self.block_fn,
            blocks=num_blocks,
            strides=self.block_strides[i],
            training=training,
            name='block_layer{}'.format(i + 1),
            data_format=self.data_format,
            shortcut=shortcut)

      # Skip the last BN+relu.
      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      # if self.pre_activation:
      #   inputs = batch_norm(inputs, training, self.data_format,
      #     name='pre_act'+'batch_norm')
      #   inputs = tf.nn.relu(inputs,name='pre_act'+'relu')

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      # Also perform max-pooling, and concat results.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      avg_pooled = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
      avg_pooled = tf.squeeze(avg_pooled, axes)
      max_pooled = tf.reduce_max(input_tensor=inputs, axis=axes, keepdims=True)
      max_pooled = tf.squeeze(max_pooled, axes)
      inputs = tf.concat([avg_pooled, max_pooled], axis=1)
      inputs = tf.identity(inputs, 'final_pooling')

      inputs = tf.compat.v1.layers.dense(
          inputs=inputs, units=self.num_classes, reuse=tf.AUTO_REUSE)
      inputs = tf.identity(inputs, 'final_dense')
      return inputs


###############################################################################
# Running the model
###############################################################################
class FastCifar10Model(Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self,
               resnet_size,
               data_format=None,
               num_classes=NUM_CLASSES,
               resnet_version=DEFAULT_VERSION,
               dtype=DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """

    # 4 block layers, so change to 8n+2.
    if resnet_size % 8 != 2:
      raise ValueError('resnet_size must be 8n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 8

    # Switch to 4 block layers. Use 64, 128, 256, 256 filters.
    super(FastCifar10Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 4,
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype)

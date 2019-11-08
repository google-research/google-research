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

"""ResNet modified to including pruning layers if specified.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import flags
import tensorflow as tf
from state_of_sparsity.sparse_rn50.pruning_layers import sparse_conv2d
from state_of_sparsity.sparse_rn50.pruning_layers import sparse_fully_connected
from tensorflow.contrib import layers as contrib_layers
from tensorflow.python.ops import init_ops  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
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
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
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
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


class SparseConvVarianceScalingInitializer(init_ops.Initializer):
  """Define an initializer for an already sparse layer."""

  def __init__(self, sparsity, seed=None, dtype=tf.float32):
    if sparsity < 0. or sparsity >= 1.:
      raise ValueError('sparsity must be in the range [0., 1.).')

    self.sparsity = sparsity
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    if partition_info is not None:
      raise ValueError('partition_info not supported.')
    if dtype is None:
      dtype = self.dtype

    # Calculate number of non-zero weights
    nnz = 1.
    for d in shape:
      nnz *= d
    nnz *= (1. - self.sparsity)

    input_channels = shape[-2]
    n = nnz / input_channels

    variance = (2. / n)**.5

    return tf.random_normal(shape, 0, variance, dtype, seed=self.seed)

  def get_config(self):
    return {
        'seed': self.seed,
        'dtype': self.dtype.name,
    }


class SparseFCVarianceScalingInitializer(init_ops.Initializer):
  """Define an initializer for an already sparse layer."""

  def __init__(self, sparsity, seed=None, dtype=tf.float32):
    if sparsity < 0. or sparsity >= 1.:
      raise ValueError('sparsity must be in the range [0., 1.).')

    self.sparsity = sparsity
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    if partition_info is not None:
      raise ValueError('partition_info not supported.')
    if dtype is None:
      dtype = self.dtype

    if len(shape) != 2:
      raise ValueError('Weights must be 2-dimensional.')

    fan_in = shape[0]
    fan_out = shape[1]

    # Calculate number of non-zero weights
    nnz = 1.
    for d in shape:
      nnz *= d
    nnz *= (1. - self.sparsity)

    limit = math.sqrt(6. / (nnz / fan_out + nnz / fan_in))

    return tf.random_uniform(shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
        'seed': self.seed,
        'dtype': self.dtype.name,
    }


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         pruning_method='baseline',
                         init_method='baseline',
                         data_format='channels_first',
                         end_sparsity=0.,
                         weight_decay=0.,
                         clip_log_alpha=8.,
                         log_alpha_threshold=3.,
                         is_training=False,
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
      which weights to remove.
    init_method: ('baseline', 'sparse') Whether to use standard initialization
      or initialization that takes into the existing sparsity of the layer.
      'sparse' only makes sense when combined with pruning_method == 'scratch'.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.
    clip_log_alpha: Value at which to clip log_alpha (if pruning_method ==
      'variational_dropout') during training.
    log_alpha_threshold: Threshold at which to zero weights based on log_alpha
      (if pruning_method == 'variational_dropout') during eval.
    is_training: boolean for whether model is in training or eval mode.
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor of size [batch, filters, height_out, width_out]

  Raises:
    ValueError: If the data_format provided is not a valid string.
  """
  if strides > 1:
    inputs = fixed_padding(
        inputs, kernel_size, data_format=data_format)
  padding = 'SAME' if strides == 1 else 'VALID'

  kernel_initializer = tf.variance_scaling_initializer()
  if pruning_method == 'threshold' and init_method == 'sparse':
    kernel_initializer = SparseConvVarianceScalingInitializer(end_sparsity)
  if pruning_method != 'threshold' and init_method == 'sparse':
    raise ValueError(
        'Unsupported combination of flags, init_method must be baseline when '
        'pruning_method is not threshold.')

  # Initialize log-alpha s.t. the dropout rate is 10%
  log_alpha_initializer = tf.random_normal_initializer(
      mean=2.197, stddev=0.01, dtype=tf.float32)
  kernel_regularizer = contrib_layers.l2_regularizer(weight_decay)
  return sparse_conv2d(
      x=inputs,
      units=filters,
      activation=None,
      kernel_size=[kernel_size, kernel_size],
      use_bias=False,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_initializer=None,
      biases_regularizer=None,
      sparsity_technique=pruning_method,
      log_sigma2_initializer=tf.constant_initializer(-15., dtype=tf.float32),
      log_alpha_initializer=log_alpha_initializer,
      normalizer_fn=None,
      strides=[strides, strides],
      padding=padding,
      threshold=log_alpha_threshold,
      clip_alpha=clip_log_alpha,
      data_format=data_format,
      is_training=is_training,
      name=name)


def residual_block_(inputs,
                    filters,
                    is_training,
                    strides,
                    use_projection=False,
                    pruning_method='baseline',
                    init_method='baseline',
                    data_format='channels_first',
                    end_sparsity=0.,
                    weight_decay=0.,
                    clip_log_alpha=8.,
                    log_alpha_threshold=3.,
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
      which weights to remove.
    init_method: ('baseline', 'sparse') Whether to use standard initialization
      or initialization that takes into the existing sparsity of the layer.
      'sparse' only makes sense when combined with pruning_method == 'scratch'.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.
    clip_log_alpha: Value at which to clip log_alpha (if pruning_method ==
      'variational_dropout') during training.
    log_alpha_threshold: Threshold at which to zero weights based on log_alpha
      (if pruning_method == 'variational_dropout') during eval.
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
        init_method=init_method,
        data_format=data_format,
        end_sparsity=end_sparsity,
        weight_decay=weight_decay,
        clip_log_alpha=clip_log_alpha,
        log_alpha_threshold=log_alpha_threshold,
        is_training=is_training,
        name=end_point)
    shortcut = batch_norm_relu(
        shortcut, is_training, relu=False, data_format=data_format)

  end_point = 'residual_1_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      pruning_method=pruning_method,
      init_method=init_method,
      data_format=data_format,
      end_sparsity=end_sparsity,
      weight_decay=weight_decay,
      clip_log_alpha=clip_log_alpha,
      log_alpha_threshold=log_alpha_threshold,
      is_training=is_training,
      name=end_point)
  inputs = batch_norm_relu(
      inputs, is_training, data_format=data_format)

  end_point = 'residual_2_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      pruning_method=pruning_method,
      init_method=init_method,
      data_format=data_format,
      end_sparsity=end_sparsity,
      weight_decay=weight_decay,
      clip_log_alpha=clip_log_alpha,
      log_alpha_threshold=log_alpha_threshold,
      is_training=is_training,
      name=end_point)
  inputs = batch_norm_relu(
      inputs, is_training, relu=False, init_zero=True, data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block_(inputs,
                      filters,
                      is_training,
                      strides,
                      use_projection=False,
                      pruning_method='baseline',
                      init_method='baseline',
                      data_format='channels_first',
                      end_sparsity=0.,
                      weight_decay=0.,
                      clip_log_alpha=8.,
                      log_alpha_threshold=3.,
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
      which weights to remove.
    init_method: ('baseline', 'sparse') Whether to use standard initialization
      or initialization that takes into the existing sparsity of the layer.
      'sparse' only makes sense when combined with pruning_method == 'scratch'.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.
    clip_log_alpha: Value at which to clip log_alpha (if pruning_method ==
      'variational_dropout') during training.
    log_alpha_threshold: Threshold at which to zero weights based on log_alpha
      (if pruning_method == 'variational_dropout') during eval.
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
        init_method=init_method,
        data_format=data_format,
        end_sparsity=end_sparsity,
        weight_decay=weight_decay,
        clip_log_alpha=clip_log_alpha,
        log_alpha_threshold=log_alpha_threshold,
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
      init_method=init_method,
      data_format=data_format,
      end_sparsity=end_sparsity,
      weight_decay=weight_decay,
      clip_log_alpha=clip_log_alpha,
      log_alpha_threshold=log_alpha_threshold,
      is_training=is_training,
      name=end_point)
  inputs = batch_norm_relu(
      inputs, is_training, data_format=data_format)

  end_point = 'bottleneck_2_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      pruning_method=pruning_method,
      init_method=init_method,
      data_format=data_format,
      end_sparsity=end_sparsity,
      weight_decay=weight_decay,
      clip_log_alpha=clip_log_alpha,
      log_alpha_threshold=log_alpha_threshold,
      is_training=is_training,
      name=end_point)
  inputs = batch_norm_relu(
      inputs, is_training, data_format=data_format)

  end_point = 'bottleneck_3_%s' % name
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      pruning_method=pruning_method,
      init_method=init_method,
      data_format=data_format,
      end_sparsity=end_sparsity,
      weight_decay=weight_decay,
      clip_log_alpha=clip_log_alpha,
      log_alpha_threshold=log_alpha_threshold,
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
                pruning_method='baseline',
                init_method='baseline',
                data_format='channels_first',
                end_sparsity=0.,
                weight_decay=0.,
                clip_log_alpha=8.,
                log_alpha_threshold=3.):
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
      which weights to remove.
    init_method: ('baseline', 'sparse') Whether to use standard initialization
      or initialization that takes into the existing sparsity of the layer.
      'sparse' only makes sense when combined with pruning_method == 'scratch'.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.
    clip_log_alpha: Value at which to clip log_alpha (if pruning_method ==
      'variational_dropout') during training.
    log_alpha_threshold: Threshold at which to zero weights based on log_alpha
      (if pruning_method == 'variational_dropout') during eval.

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
        init_method=init_method,
        data_format=data_format,
        end_sparsity=end_sparsity,
        weight_decay=weight_decay,
        clip_log_alpha=clip_log_alpha,
        log_alpha_threshold=log_alpha_threshold,
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
            init_method=init_method,
            data_format=data_format,
            end_sparsity=end_sparsity,
            weight_decay=weight_decay,
            clip_log_alpha=clip_log_alpha,
            log_alpha_threshold=log_alpha_threshold,
            name=end_point)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn,
                        num_blocks,
                        num_classes,
                        pruning_method='baseline',
                        init_method='baseline',
                        width=1.,
                        prune_first_layer=True,
                        prune_last_layer=True,
                        data_format='channels_first',
                        end_sparsity=0.,
                        weight_decay=0.,
                        clip_log_alpha=8.,
                        log_alpha_threshold=3.,
                        name=None):
  """Generator for ResNet v1 models.

  Args:
    block_fn: String that defines whether to use a `residual_block` or
      `bottleneck_block`.
    num_blocks: list of Ints that denotes number of blocks to include in each
      block group. Each group consists of blocks that take inputs of the same
      resolution.
    num_classes: Int number of possible classes for image classification.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    init_method: ('baseline', 'sparse') Whether to use standard initialization
      or initialization that takes into the existing sparsity of the layer.
      'sparse' only makes sense when combined with pruning_method == 'scratch'.
    width: Float that scales the number of filters in each layer.
    prune_first_layer: Whether or not to prune the first layer.
    prune_last_layer: Whether or not to prune the last layer.
    data_format: String either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.
    clip_log_alpha: Value at which to clip log_alpha (if pruning_method ==
      'variational_dropout') during training.
    log_alpha_threshold: Threshold at which to zero weights based on log_alpha
      (if pruning_method == 'variational_dropout') during eval.
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
          filters=int(64 * width),
          kernel_size=7,
          strides=2,
          pruning_method=pruning_method if prune_first_layer else 'baseline',
          init_method=init_method if prune_first_layer else 'baseline',
          is_training=is_training,
          data_format=data_format,
          end_sparsity=end_sparsity,
          weight_decay=weight_decay,
          clip_log_alpha=clip_log_alpha,
          log_alpha_threshold=log_alpha_threshold,
          name='initial_conv')

      inputs = tf.identity(inputs, 'initial_conv')
      inputs = batch_norm_relu(
          inputs, is_training, data_format=data_format)

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
          filters=int(64 * width),
          block_fn=block_fn,
          blocks=num_blocks[0],
          strides=1,
          is_training=is_training,
          name='block_group1',
          pruning_method=pruning_method,
          init_method=init_method,
          data_format=data_format,
          end_sparsity=end_sparsity,
          weight_decay=weight_decay,
          clip_log_alpha=clip_log_alpha,
          log_alpha_threshold=log_alpha_threshold)
      inputs = block_group(
          inputs=inputs,
          filters=int(128 * width),
          block_fn=block_fn,
          blocks=num_blocks[1],
          strides=2,
          is_training=is_training,
          name='block_group2',
          pruning_method=pruning_method,
          init_method=init_method,
          data_format=data_format,
          end_sparsity=end_sparsity,
          weight_decay=weight_decay,
          clip_log_alpha=clip_log_alpha,
          log_alpha_threshold=log_alpha_threshold)
      inputs = block_group(
          inputs=inputs,
          filters=int(256 * width),
          block_fn=block_fn,
          blocks=num_blocks[2],
          strides=2,
          is_training=is_training,
          name='block_group3',
          pruning_method=pruning_method,
          init_method=init_method,
          data_format=data_format,
          end_sparsity=end_sparsity,
          weight_decay=weight_decay,
          clip_log_alpha=clip_log_alpha,
          log_alpha_threshold=log_alpha_threshold)
      inputs = block_group(
          inputs=inputs,
          filters=int(512 * width),
          block_fn=block_fn,
          blocks=num_blocks[3],
          strides=2,
          is_training=is_training,
          name='block_group4',
          pruning_method=pruning_method,
          init_method=init_method,
          data_format=data_format,
          end_sparsity=end_sparsity,
          weight_decay=weight_decay,
          clip_log_alpha=clip_log_alpha,
          log_alpha_threshold=log_alpha_threshold)

      pool_size = (inputs.shape[1], inputs.shape[2])
      inputs = tf.layers.average_pooling2d(
          inputs=inputs,
          pool_size=pool_size,
          strides=1,
          padding='VALID',
          data_format=data_format,
          name='final_avg_pool')
      inputs = tf.identity(inputs, 'final_avg_pool')
      multiplier = 4 if block_fn is bottleneck_block_ else 1
      fc_units = multiplier * int(512 * width)
      inputs = tf.reshape(inputs, [-1, fc_units])

      kernel_initializer = tf.random_normal_initializer(stddev=.01)
      if (pruning_method == 'threshold' and init_method == 'sparse' and
          prune_last_layer):
        kernel_initializer = SparseFCVarianceScalingInitializer(end_sparsity)
      if pruning_method != 'threshold' and init_method == 'sparse':
        raise ValueError(
            'Unsupported combination of flags, init_method must be baseline '
            'when pruning_method is not threshold.')

      # Initialize log-alpha s.t. the dropout rate is 10%
      log_alpha_initializer = tf.random_normal_initializer(
          mean=2.197, stddev=0.01, dtype=tf.float32)
      kernel_regularizer = contrib_layers.l2_regularizer(weight_decay)
      inputs = sparse_fully_connected(
          x=inputs,
          units=num_classes,
          sparsity_technique=pruning_method if prune_last_layer else 'baseline',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          log_sigma2_initializer=tf.constant_initializer(
              -15., dtype=tf.float32),
          log_alpha_initializer=log_alpha_initializer,
          clip_alpha=clip_log_alpha,
          threshold=log_alpha_threshold,
          is_training=is_training,
          name='final_dense')

      inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def resnet_v1_(resnet_depth,
               num_classes,
               pruning_method='baseline',
               init_method='baseline',
               width=1.,
               prune_first_layer=True,
               prune_last_layer=True,
               data_format='channels_first',
               end_sparsity=0.,
               weight_decay=0.,
               clip_log_alpha=8.,
               log_alpha_threshold=3.):
  """Returns the ResNet model for a given size and number of output classes.

  Args:
    resnet_depth: Int number of blocks in the architecture.
    num_classes: Int number of possible classes for image classification.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    init_method: ('baseline', 'sparse') Whether to use standard initialization
      or initialization that takes into the existing sparsity of the layer.
      'sparse' only makes sense when combined with pruning_method == 'scratch'.
    width: Float multiplier of the number of filters in each layer.
    prune_first_layer: Whether or not to prune the first layer.
    prune_last_layer: Whether or not to prune the last layer.
    data_format: String specifying either "channels_first" for `[batch,
      channels, height, width]` or "channels_last for `[batch, height, width,
      channels]`.
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.
    clip_log_alpha: Value at which to clip log_alpha (if pruning_method ==
      'variational_dropout') during training.
    log_alpha_threshold: Threshold at which to zero weights based on log_alpha
      (if pruning_method == 'variational_dropout') during eval.

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
  return resnet_v1_generator(
      params['block'], params['layers'], num_classes, pruning_method,
      init_method, width, prune_first_layer, prune_last_layer, data_format,
      end_sparsity, weight_decay, clip_log_alpha, log_alpha_threshold)

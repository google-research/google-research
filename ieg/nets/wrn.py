# coding=utf-8
"""Builds the Wide-ResNet Model."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow.compat.v1 as tf

from ieg import utils
from ieg.models import custom_ops as ops
from ieg.models.custom_ops import decay_weights
from ieg.models.networks import StrategyNetBase


def residual_block(x,
                   in_filter,
                   out_filter,
                   stride,
                   is_training,
                   activate_before_residual=False):
  """Adds residual connection to `x` in addition to applying BN->ReLU->3x3 Conv.

  Args:
    x: Tensor that is the output of the previous layer in the model.
    in_filter: Number of filters `x` has.
    out_filter: Number of filters that the output of this layer will have.
    stride: Integer that specified what stride should be applied to `x`.
    is_training: Bool for training mode.
    activate_before_residual: Boolean on whether a BN->ReLU should be applied to
      x before the convolution is applied.

  Returns:
    A Tensor that is the result of applying two sequences of BN->ReLU->3x3 Conv
    and then adding that Tensor to `x`.
  """

  if activate_before_residual:  # Pass up RELU and BN activation for resnet
    with tf.variable_scope('shared_activation'):
      x = ops.batch_norm(x, scope='init_bn', is_training=is_training)
      x = tf.nn.relu(x)
      orig_x = x
  else:
    orig_x = x

  block_x = x
  if not activate_before_residual:
    with tf.variable_scope('residual_only_activation'):
      block_x = ops.batch_norm(
          block_x, scope='init_bn', is_training=is_training)
      block_x = tf.nn.relu(block_x)

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(block_x, out_filter, 3, stride=stride, scope='conv1')

  with tf.variable_scope('sub2'):
    block_x = ops.batch_norm(block_x, scope='bn2', is_training=is_training)
    block_x = tf.nn.relu(block_x)
    block_x = ops.conv2d(block_x, out_filter, 3, stride=1, scope='conv2')

  with tf.variable_scope(
      'sub_add'):  # If number of filters do not agree then zero pad them
    if in_filter != out_filter:
      orig_x = ops.avg_pool(orig_x, stride, stride)
      orig_x = ops.zero_pad(orig_x, in_filter, out_filter)
  x = orig_x + block_x
  return x


def _res_add(in_filter, out_filter, stride, x, orig_x):
  """Adds `x` with `orig_x`, both of which are layers in the model.

  Args:
    in_filter: Number of filters in `orig_x`.
    out_filter: Number of filters in `x`.
    stride: Integer specifying the stide that should be applied `orig_x`.
    x: Tensor that is the output of the previous layer.
    orig_x: Tensor that is the output of an earlier layer in the network.

  Returns:
    A Tensor that is the result of `x` and `orig_x` being added after
    zero padding and striding are applied to `orig_x` to get the shapes
    to match.
  """
  if in_filter != out_filter:
    orig_x = ops.avg_pool(orig_x, stride, stride)
    orig_x = ops.zero_pad(orig_x, in_filter, out_filter)
  x = x + orig_x
  orig_x = x
  return x, orig_x


class WRN(StrategyNetBase):
  """Wide-ResNet parallel version."""

  def __init__(self, num_classes, wrn_size=160, weight_decay_rate=5e-4):
    super(WRN, self).__init__()

    self.num_classes = num_classes
    self.wrn_size = wrn_size
    self.wd = weight_decay_rate

  def get_partial_variables(self, level=-1):
    vs = []
    if level == 0:
      # only get last fc layer
      for v in self.trainable_variables:
        if 'FC' in v.name:
          vs.append(v)
    elif level < 0:
      vs = self.trainable_variables
    else:
      raise ValueError
    assert vs, 'Length of obtained partial variable is 0'
    return vs

  def __call__(self,
               images,
               name,
               reuse=True,
               training=True,
               custom_getter=None):
    """Builds the WRN model.

    Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.

    Args:
      images: Tensor of images that will be fed into the Wide ResNet Model.
      name: Name of the model as scope
      reuse: If True, reuses the parameters.
      training: If True, for training stage.
      custom_getter: custom_getter function for variable_scope.

    Returns:
      The logits of the Wide ResNet model.
    """
    num_classes = self.num_classes
    wrn_size = self.wrn_size

    kernel_size = wrn_size
    filter_size = 3
    num_blocks_per_resnet = 4
    filters = [
        min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4
    ]
    strides = [1, 2, 2]  # stride for each resblock

    with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

      # Run the first conv
      with tf.variable_scope('init'):
        x = images
        output_filters = filters[0]
        x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')

      first_x = x  # Res from the beginning
      orig_x = x  # Res from previous block

      for block_num in range(1, 4):
        with tf.variable_scope('unit_{}_0'.format(block_num)):
          activate_before_residual = True if block_num == 1 else False
          x = residual_block(
              x,
              filters[block_num - 1],
              filters[block_num],
              strides[block_num - 1],
              activate_before_residual=activate_before_residual,
              is_training=training)
        for i in range(1, num_blocks_per_resnet):
          with tf.variable_scope('unit_{}_{}'.format(block_num, i)):
            x = residual_block(
                x,
                filters[block_num],
                filters[block_num],
                1,
                activate_before_residual=False,
                is_training=training)
        x, orig_x = _res_add(filters[block_num - 1], filters[block_num],
                             strides[block_num - 1], x, orig_x)
      final_stride_val = np.prod(strides)
      x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
      with tf.variable_scope('unit_last'):
        x = ops.batch_norm(x, scope='final_bn', is_training=training)
        x = tf.nn.relu(x)
        x = ops.global_avg_pool(x)
        logits = ops.fc(x, num_classes)

      if not isinstance(reuse, bool) or not reuse:
        self.regularization_loss = decay_weights(
            self.wd, utils.get_var(tf.trainable_variables(), name))
        self.init(name, with_name='moving', outputs=logits)
        self.count_parameters(name)
    return logits

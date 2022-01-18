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

"""Builds the Wide-ResNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import custom_ops as ops
import tensorflow as tf


def activation_fn(x, hparams, layer=None):
  """Define the activation function."""
  if hparams.use_gamma_swish:
    assert layer is not None
    b = tf.contrib.framework.get_variables_by_suffix(
        'swish_beta_layer_{}'.format(layer))[0]
    gamma = tf.contrib.framework.get_variables_by_suffix(
        'swish_gamma_layer_{}'.format(layer))[0]
    assert b.name == 'model/swish_beta_layer_{}:0'.format(layer)
    assert gamma.name == 'model/swish_gamma_layer_{}:0'.format(layer)
    tf.logging.info('beta shape: {}'.format(b))
    tf.logging.info('gamma shape: {}'.format(gamma))
    return gamma * x * tf.nn.sigmoid(b * x)
  else:
    return tf.nn.relu(x)


def residual_block(
    x, in_filter, out_filter, stride, hparams=None, layer=None):
  """Adds residual connection to `x` in addition to applying BN ->Act->3x3 Conv.

  Args:
    x: Tensor that is the output of the previous layer in the model.
    in_filter: Number of filters `x` has.
    out_filter: Number of filters that the output of this layer will have.
    stride: Integer that specified what stride should be applied to `x`.
    hparams: hparams
    layer: layer

  Returns:
    A Tensor that is the result of applying two sequences of BN->Act->3x3 Conv
    and then adding that Tensor to `x`.
  """
  if layer is not None:
    tf.logging.info('layer: {}'.format(layer))
  if hparams is None:
    hparams = {'use_swish': 0, 'beta_swish': 0, 'use_BN': 0}

  with tf.variable_scope('residual_activation'):
    block_x = ops.maybe_normalize(x, scope='init_bn')
    block_x = activation_fn(block_x, hparams=hparams, layer=layer)

    if in_filter != out_filter:
      x = ops.conv2d(block_x, out_filter, 3, stride=stride, scope='conva')

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(
        block_x, out_filter, 3, stride=stride, scope='conv1',
        position='residual')

  with tf.variable_scope('sub2'):
    block_x = ops.maybe_normalize(block_x, scope='bn2')
    block_x = activation_fn(block_x, hparams=hparams, layer=layer)
    block_x = ops.conv2d(
        block_x, out_filter, 3, stride=1, scope='conv2',
        position='residual_last')

  return x + block_x


def build_wrn_model(images, num_classes, hparams):
  """Builds the WRN model.

  Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.

  Args:
    images: Tensor of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    hparams: hparams.

  Returns:
    The logits of the Wide ResNet model.
  """
  kernel_size = hparams.wrn_size
  filter_size = 3
  num_blocks_per_resnet = 4
  filters = [
      min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4
  ]
  strides = [1, 2, 2]  # stride for each resblock

  # Run the first conv
  with tf.variable_scope('init'):
    x = images
    output_filters = filters[0]
    x = ops.conv2d(x, output_filters, filter_size, scope='init_conv',
                   position='input')

  layer = 0
  for block_num in range(1, 4):
    with tf.variable_scope('unit_{}_0'.format(block_num)):
      x = residual_block(
          x,
          filters[block_num - 1],
          filters[block_num],
          strides[block_num - 1],
          hparams=hparams,
          layer=layer)
      layer += 1
    for i in range(1, num_blocks_per_resnet):
      with tf.variable_scope('unit_{}_{}'.format(block_num, i)):
        x = residual_block(
            x,
            filters[block_num],
            filters[block_num],
            1,
            hparams=hparams,
            layer=layer)
        layer += 1

  with tf.variable_scope('unit_last'):
    x = ops.maybe_normalize(x, scope='final_bn')
    x = activation_fn(x, hparams=hparams, layer=layer)
    hiddens = ops.global_avg_pool(x)
    logits = ops.fc(hiddens, num_classes)
  return logits, hiddens

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

# Lint as: python2, python3
"""ResNet V1 implementation for UQ experiments.

Mostly derived from Keras documentation:
  https://keras.io/examples/cifar10_resnet/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import logging

from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from uq_benchmark_2019 import uq_utils
keras = tf.keras
tfd = tfp.distributions


def _resnet_layer(inputs,
                  num_filters=16,
                  kernel_size=3,
                  strides=1,
                  activation='relu',
                  depth=20,
                  batch_norm=True,
                  conv_first=True,
                  variational=False,
                  std_prior_scale=1.5,
                  eb_prior_fn=None,
                  always_on_dropout_rate=None,
                  examples_per_epoch=None):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs (tensor): input tensor from input image or previous layer
    num_filters (int): Conv2D number of filters
    kernel_size (int): Conv2D square kernel dimensions
    strides (int): Conv2D square stride dimensions
    activation (string): Activation function string.
    depth (int): ResNet depth; used for initialization scale.
    batch_norm (bool): whether to include batch normalization
    conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
    variational (bool): Whether to use a variational convolutional layer.
    std_prior_scale (float): Scale for log-normal hyperprior.
    eb_prior_fn (callable): Empirical Bayes prior for use with TFP layers.
    always_on_dropout_rate (float): Dropout rate (active in train and test).
    examples_per_epoch (int): Number of examples per epoch for variational KL.

  Returns:
      x (tensor): tensor as input to the next layer
  """
  if variational:
    divergence_fn = uq_utils.make_divergence_fn_for_empirical_bayes(
        std_prior_scale, examples_per_epoch)

    def fixup_init(shape, dtype=None):
      """Fixup initialization; see https://arxiv.org/abs/1901.09321."""
      return keras.initializers.he_normal()(shape, dtype=dtype) * depth**(-1/4)

    conv = tfp.layers.Convolution2DFlipout(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_prior_fn=eb_prior_fn,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=fixup_init),
        kernel_divergence_fn=divergence_fn)
  else:
    conv = keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(1e-4))

  def apply_conv(net):
    logging.info('Applying conv layer; always_on_dropout=%s.',
                 always_on_dropout_rate)
    if always_on_dropout_rate:
      net = keras.layers.Dropout(always_on_dropout_rate)(net, training=True)
    return conv(net)

  x = inputs
  x = apply_conv(x) if conv_first else x
  x = (keras.layers.BatchNormalization()(x)
       if batch_norm and not variational else x)
  x = keras.layers.Activation(activation)(x) if activation is not None else x
  x = x if conv_first else apply_conv(x)
  return x


def build_resnet_v1(input_layer, depth,
                    variational,
                    std_prior_scale,
                    always_on_dropout_rate,
                    examples_per_epoch,
                    eb_prior_fn=None,
                    no_first_layer_dropout=False,
                    num_filters=16):
  """ResNet Version 1 Model builder [a]."""
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  num_res_blocks = int((depth - 2) / 6)

  activation = 'selu' if variational else 'relu'
  resnet_layer = functools.partial(
      _resnet_layer,
      activation=activation,
      depth=depth,
      std_prior_scale=std_prior_scale,
      always_on_dropout_rate=always_on_dropout_rate,
      examples_per_epoch=examples_per_epoch,
      eb_prior_fn=eb_prior_fn)

  logging.info('Starting ResNet build.')
  x = resnet_layer(
      inputs=input_layer,
      num_filters=num_filters,
      always_on_dropout_rate=(None if no_first_layer_dropout
                              else always_on_dropout_rate))
  # Instantiate the stack of residual units
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
      y = resnet_layer(inputs=y, num_filters=num_filters, activation=None,
                       variational=variational)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = resnet_layer(inputs=x,
                         num_filters=num_filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         batch_norm=False)
      x = keras.layers.add([x, y])
      x = keras.layers.Activation(activation)(x)
    num_filters *= 2

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = keras.layers.AveragePooling2D(pool_size=8)(x)
  return keras.layers.Flatten()(x)

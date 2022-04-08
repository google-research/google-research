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

"""Contains models used in the experiments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v2 as tf

from cold_posterior_bnn.core import frn
from cold_posterior_bnn.imdb import imdb_model


def build_cnnlstm(num_words, sequence_length, pfac):
  model = imdb_model.cnn_lstm_nd(pfac, num_words, sequence_length)
  return model


def build_resnet_v1(input_shape, depth, num_classes, pfac, use_frn=False,
                    use_internal_bias=True):
  """Builds ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    num_classes: Number of output classes.
    pfac: priorfactory.PriorFactory class.
    use_frn: if True, then use Filter Response Normalization (FRN) instead of
      batchnorm.
    use_internal_bias: if True, use biases in all Conv layers.
      If False, only use a bias in the final Dense layer.

  Returns:
    tf.keras.Model.
  """
  def resnet_layer(inputs,
                   filters,
                   kernel_size=3,
                   strides=1,
                   activation=None,
                   pfac=None,
                   use_frn=False,
                   use_bias=True):
    """2D Convolution-Batch Normalization-Activation stack builder.

    Args:
      inputs: tf.Tensor.
      filters: Number of filters for Conv2D.
      kernel_size: Kernel dimensions for Conv2D.
      strides: Stride dimensinons for Conv2D.
      activation: tf.keras.activations.Activation.
      pfac: prior.PriorFactory object.
      use_frn: if True, use Filter Response Normalization (FRN) layer
      use_bias: if True, use biases in Conv layers.

    Returns:
      tf.Tensor.
    """
    x = inputs
    logging.info('Applying conv layer.')
    x = pfac(tf.keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=use_bias))(x)

    if use_frn:
      x = pfac(frn.FRN())(x)
    else:
      x = tf.keras.layers.BatchNormalization()(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
    return x

  # Main network code
  num_res_blocks = (depth - 2) // 6
  filters = 16
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

  logging.info('Starting ResNet build.')
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = resnet_layer(inputs,
                   filters=filters,
                   activation='relu',
                   pfac=pfac,
                   use_frn=use_frn,
                   use_bias=use_internal_bias)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(x,
                       filters=filters,
                       strides=strides,
                       activation='relu',
                       pfac=pfac,
                       use_frn=use_frn,
                       use_bias=use_internal_bias)
      y = resnet_layer(y,
                       filters=filters,
                       activation=None,
                       pfac=pfac,
                       use_frn=use_frn,
                       use_bias=use_internal_bias)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = resnet_layer(x,
                         filters=filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         pfac=pfac,
                         use_frn=use_frn,
                         use_bias=use_internal_bias)
      x = tf.keras.layers.add([x, y])
      if use_frn:
        x = pfac(frn.TLU())(x)
      else:
        x = tf.keras.layers.Activation('relu')(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = pfac(tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal'))(x)

  logging.info('ResNet successfully built.')
  return tf.keras.models.Model(inputs=inputs, outputs=x)

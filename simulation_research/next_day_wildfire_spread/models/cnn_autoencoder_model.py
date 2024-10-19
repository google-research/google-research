# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""CNN autoencoder model definition."""
from typing import Sequence

import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras

from simulation_research.next_day_wildfire_spread.models import model_utils


def encoder(
    input_tensor,
    layers_list,
    pool_list,
    dropout = model_utils.DROPOUT_DEFAULT,
    batch_norm = model_utils.BATCH_NORM_DEFAULT,
    l1_regularization = model_utils.L1_REGULARIZATION_DEFAULT,
    l2_regularization = model_utils.L2_REGULARIZATION_DEFAULT
):
  """Performs a series of downsamples to the input.

  Args:
    input_tensor: Input to the encoder.
    layers_list: Filters for successive layers.
    pool_list: Values for maxpooling `pool_size`, stride, and upsampling size.
    dropout: Dropout rate.
    batch_norm: Controls batch normalization layers.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Output of the encoder.
  """
  net = input_tensor
  net = model_utils.conv2d_layer(
      filters=layers_list[0],
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(
          net)
  if batch_norm == 'all':
    net = keras.layers.BatchNormalization()(net)
  net = keras.layers.LeakyReLU()(net)
  net = keras.layers.Dropout(dropout)(net)
  net = model_utils.conv2d_layer(
      filters=layers_list[0],
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(
          net)
  shortcut = model_utils.conv2d_layer(
      filters=layers_list[0],
      kernel_size=model_utils.RES_SHORTCUT_KERNEL_SIZE,
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(
          input_tensor)
  if batch_norm == 'all':
    shortcut = keras.layers.BatchNormalization()(shortcut)
  shortcut = keras.layers.Dropout(dropout)(shortcut)

  net = shortcut + net

  for i in range(len(layers_list[1:])):
    filters = layers_list[1 + i]
    pool = pool_list[1 + i]
    net = model_utils.res_block(
        net,
        filters=(filters, filters),
        strides=(pool, 1),
        pool_size=pool,
        dropout=dropout,
        batch_norm=batch_norm,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)
  return net


def decoder_block(
    input_tensor,
    filters,
    pool_size,
    dropout = model_utils.DROPOUT_DEFAULT,
    batch_norm = model_utils.BATCH_NORM_DEFAULT,
    l1_regularization = model_utils.L1_REGULARIZATION_DEFAULT,
    l2_regularization = model_utils.L2_REGULARIZATION_DEFAULT
):
  """Creates a decoder block that performs upsampling.

  Args:
    input_tensor: Input to the decoder block.
    filters: Number of filters.
    pool_size: Maxpool's `pool_size`.
    dropout: Dropout rate.
    batch_norm: Controls batch normalization layers.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Output of the decoder block.
  """
  net = keras.layers.UpSampling2D(
      size=(pool_size, pool_size), interpolation='nearest')(
          input_tensor)
  net = model_utils.res_block(
      net,
      filters=(filters, filters),
      strides=model_utils.RES_DECODER_STRIDES,
      pool_size=pool_size,
      dropout=dropout,
      batch_norm=batch_norm,
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)
  return net


def decoder(
    input_tensor,
    layers_list,
    pool_list,
    dropout = model_utils.DROPOUT_DEFAULT,
    batch_norm = model_utils.BATCH_NORM_DEFAULT,
    l1_regularization = model_utils.L1_REGULARIZATION_DEFAULT,
    l2_regularization = model_utils.L2_REGULARIZATION_DEFAULT
):
  """Performs a series of upsamples to the input.

  Args:
    input_tensor: Input to the decoder.
    layers_list: Filters for successive layers of the decoder, usually reverse
      of the encoder `layers_list`.
    pool_list: Values for maxpooling pool_size and upsampling size.
    dropout: Dropout rate.
    batch_norm: Controls batch normalization layers.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Output of the decoder.
  """
  net = input_tensor
  for layer, pool in zip(layers_list, pool_list):
    net = decoder_block(net, layer, pool, dropout, batch_norm,
                        l1_regularization, l2_regularization)
  return net


def create_model(
    input_tensor,
    num_out_channels,
    encoder_layers,
    decoder_layers,
    encoder_pools,
    decoder_pools,
    dropout = model_utils.DROPOUT_DEFAULT,
    batch_norm = model_utils.BATCH_NORM_DEFAULT,
    l1_regularization = model_utils.L1_REGULARIZATION_DEFAULT,
    l2_regularization = model_utils.L2_REGULARIZATION_DEFAULT
):
  """Create a CNN autoencoder model.

  Args:
    input_tensor: Input to the model.
    num_out_channels: Number of output channels.
    encoder_layers: Filters for successive layers of the encoder.
    decoder_layers: Filters for successive layers of the decoder; usually the
      reverse of `encoder_layers`, but is shorter for coarse segmentation.
    encoder_pools: List of size and stride of the maxpool and of the
      corresponding downsampling. Entries equal to `1` effectively remove
      maxpool and downsampling; can be used to keep layer depth and bottleneck
      height and width dimensions while decreasing the input and output size by
      a factor of `2`. Should be same length as `encoder_layers`.
    decoder_pools: List of size and stride of the maxpool and of the
      corresponding upsampling. Usually the reverse of `encoder_pools`, but is
      shorter for coarse segmentation. Should be same length as
      `decoder_layers`.
    dropout: Dropout rate.
    batch_norm: Controls batch normalization layers.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Output of the model.
  """
  if len(encoder_layers) != len(encoder_pools):
    raise ValueError(
        'Length of encoder_layers and encoder_pools should be equal.')
  if len(decoder_layers) != len(decoder_pools):
    raise ValueError(
        'Length of decoder_layers and decoder_pools should be equal.')
  if len(decoder_layers) > len(encoder_layers):
    raise ValueError(
        'Length of decoder_layers should be <= length of encoder_layers.')
  bottleneck_x = encoder(input_tensor, encoder_layers, encoder_pools, dropout,
                         batch_norm, l1_regularization, l2_regularization)

  x = model_utils.res_block(
      bottleneck_x,
      filters=(encoder_layers[-1], encoder_layers[-1]),
      dropout=dropout,
      batch_norm=batch_norm,
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)
  x = decoder(x, decoder_layers, decoder_pools, dropout, batch_norm,
              l1_regularization, l2_regularization)
  x = model_utils.conv2d_layer(
      filters=num_out_channels,
      kernel_size=model_utils.RES_SHORTCUT_KERNEL_SIZE,
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(x)
  return x

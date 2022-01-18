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

"""Implements a custom U-Net.

Reference:
  Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for
  Biomedical Image Segmentation, MICCAI 2015.
  https://doi.org/10.1007/978-3-319-24574-4_28
"""
from typing import Sequence, Tuple

import tensorflow as tf


def _down_block(x,
                depth,
                name_prefix = 'down'):
  """Applies a U-Net downscaling block to the previous stage's output.

  Args:
    x: Output from the previous stage, with shape [B, H, W, C].
    depth: Number of channels in the output tensor.
    name_prefix: Prefix to each layer's name. Each block's prefix must be unique
      in the same model.

  Returns:
    Two tensors:
    - Output of the Conv2D layer used for the skip connection. Has shape [B, H,
      W, `depth`].
    - Output of the MaxPool2D layer used as the input to the next block. Has
      shape [B, H/2, W/2, `depth`].
  """
  conv = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=3,
      padding='same',
      activation='relu',
      name=f'{name_prefix}_conv1')(
          x)
  skip = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=3,
      padding='same',
      activation='relu',
      name=f'{name_prefix}_conv2')(
          conv)
  down_2x = tf.keras.layers.MaxPool2D(
      pool_size=(2, 2), name=f'{name_prefix}_pool')(
          skip)
  return skip, down_2x


def _up_block(x,
              skip,
              depth,
              interpolation = 'bilinear',
              name_prefix = 'up'):
  """Applies a U-Net upscaling block to the previous stage's output.

  Args:
    x: Output from the previous stage, with shape [B, H, W, C].
    skip: Output from the corresponding downscaling block, with shape [B, 2H,
      2W, C']. Normally C' = C / 2.
    depth: Number of channels in the output tensor.
    interpolation: Interpolation method. Must be "neareat" or "bilinear".
    name_prefix: Prefix to each layer's name. Each block's prefix must be unique
      in the same model.

  Returns:
    Output of the upscaling block. Has shape [B, 2H, 2W, `depth`].
  """
  up_2x = tf.keras.layers.UpSampling2D(
      size=(2, 2), interpolation=interpolation, name=f'{name_prefix}_2x')(
          x)
  up_2x = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=2,
      padding='same',
      activation='relu',
      name=f'{name_prefix}_2xconv')(
          up_2x)
  concat = tf.keras.layers.concatenate([up_2x, skip],
                                       name=f'{name_prefix}_concat')
  conv = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=3,
      padding='same',
      activation='relu',
      name=f'{name_prefix}_conv1')(
          concat)
  conv = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=3,
      padding='same',
      activation='relu',
      name=f'{name_prefix}_conv2')(
          conv)
  return conv


def get_model(input_shape = (512, 512, 3),
              scales = 4,
              bottleneck_depth = 1024,
              bottleneck_layers = 2):
  """Builds a U-Net with given parameters.

  The output of this model has the same shape as the input tensor.

  Args:
    input_shape: Shape of the input tensor, without the batch dimension. For a
      typical RGB image, this should be [height, width, 3].
    scales: Number of downscaling/upscaling blocks in the network. The width and
      height of the input tensor are 2**`scales` times those of the bottleneck.
      0 means no rescaling is applied and a simple feed-forward network is
      returned.
    bottleneck_depth: Number of channels in the bottleneck tensors.
    bottleneck_layers: Number of Conv2D layers in the bottleneck.

  Returns:
    A Keras model instance representing a U-Net.
  """
  input_layer = tf.keras.Input(shape=input_shape, name='input')
  previous_output = input_layer

  # Downscaling arm. Produces skip connections.
  skips = []
  depths = [bottleneck_depth // 2**i for i in range(scales, 0, -1)]
  for depth in depths:
    skip, previous_output = _down_block(
        previous_output, depth, name_prefix=f'down{depth}')
    skips.append(skip)

  # Bottleneck.
  for i in range(bottleneck_layers):
    previous_output = tf.keras.layers.Conv2D(
        filters=bottleneck_depth,
        kernel_size=3,
        padding='same',
        activation='relu',
        name=f'bottleneck_conv{i + 1}')(
            previous_output)

  # Upscaling arm. Consumes skip connections.
  for depth, skip in zip(reversed(depths), reversed(skips)):
    previous_output = _up_block(
        previous_output, skip, depth, name_prefix=f'up{depth}')

  # Squash output to (0, 1).
  output_layer = tf.keras.layers.Conv2D(
      filters=input_shape[-1],
      kernel_size=1,
      activation='sigmoid',
      name='output')(
          previous_output)

  return tf.keras.Model(input_layer, output_layer, name='unet')

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

"""Unprocessing neural network architecture.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf


def conv(features, num_channels, activation=tf.nn.leaky_relu):
  """Applies a 3x3 conv layer."""
  return tf.layers.conv2d(features, num_channels, 3, padding='same',
                          activation=activation)


def conv_block(features, num_channels):
  """Applies 3x conv layers."""
  with tf.name_scope(None, 'conv_block'):
    features = conv(features, num_channels)
    features = conv(features, num_channels)
    features = conv(features, num_channels)
    return features


def downsample_2x(features):
  """Applies a 2x spatial downsample via max pooling."""
  with tf.name_scope(None, 'downsample_2x'):
    return tf.layers.max_pooling2d(features, 2, 2, padding='same')


def upsample_2x(features):
  """Applies a 2x spatial upsample via bilinear interpolation."""
  with tf.name_scope(None, 'upsample_2x'):
    shape = tf.shape(features)
    shape = [shape[1] * 2, shape[2] * 2]
    features = tf.image.resize_bilinear(features, shape)
    return features


def inference(noisy_img, variance):
  """Residual U-Net with skip connections.

  Expects four input channels for the Bayer color filter planes (e.g. RGGB).
  This is the format of real raw images before they are processed, and an
  effective time to denoise images in an image processing pipelines.

  Args:
    noisy_img: Tensor of shape [B, H, W, 4].
    variance: Tensor of shape [B, H, W, 4].

  Returns:
    Denoised image in Tensor of shape [B, H, W, 4].
  """

  noisy_img = tf.identity(noisy_img, 'noisy_img')
  noisy_img.set_shape([None, None, None, 4])
  variance = tf.identity(variance, 'variance')
  variance.shape.assert_is_compatible_with(noisy_img.shape)
  variance.set_shape([None, None, None, 4])

  features = tf.concat([noisy_img, variance], axis=-1)
  skip_connections = []

  with tf.name_scope(None, 'encoder'):
    for num_channels in (32, 64, 128, 256):
      features = conv_block(features, num_channels)
      skip_connections.append(features)
      features = downsample_2x(features)
    features = conv_block(features, 512)

  with tf.name_scope(None, 'decoder'):
    for num_channels in (256, 128, 64, 32):
      features = upsample_2x(features)
      with tf.name_scope(None, 'skip_connection'):
        features = tf.concat([features, skip_connections.pop()], axis=-1)
      features = conv_block(features, num_channels)

  residual = conv(features, 4, None)
  return tf.identity(noisy_img + residual, 'denoised_img')

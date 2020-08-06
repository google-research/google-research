# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# -*- coding: utf-8 -*-
"""Network definitions for learning disparity and pose."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def apply_harmonic_bias(channels, num_layers):
  """Offset network outputs to ensure harmonic distribution of initial alpha.

  The first num_layers-1 channels are the ones that will become the alpha
  channels for layers [1, N-1]. (There is no channel corresponding to the alpha
  of the back layer because is it always 1.0, i.e. fully opaque.)

  We adjust these first num_layers-1 channels so that instead of all layer
  alphas having an initial mean of 0.5, the Nth layer from the back has an
  initial mean of 1/N. This harmonic distribution allows each layer to
  contribute equal weight when the layers are composed.

  Args:
    channels: [..., N] Network output before final tanh activation.
    num_layers: How many layers we are predicting an MPI for.

  Returns:
    [..., N] Adjusted output.
  """
  # The range below begins at 2 because the back layer is not predicted, as it's
  # always fully opaque.
  alpha = 1.0 / tf.range(2, num_layers + 1, dtype=tf.float32)
  # Convert to desired offset before activation and scaling:
  shift = tf.atanh(2.0 * alpha - 1.0)

  # Remaining channels are left as is.
  no_shift = tf.zeros([tf.shape(channels)[-1] - (num_layers - 1)])
  shift = tf.concat([shift, no_shift], axis=-1)
  return channels + shift


def mpi_network(image, num_channels, num_layers):
  """Predict num_channels full-resolution output channels from input image.

  This is a U-Net style architecture with skip connections.

  Args:
    image: [B, H, W, C] input image.
    num_channels: number of output channels required.
    num_layers: number of MPI layers we're going to derive from it.

  Returns:
    [B, H, W, num_channels] output from network, in the range (0, 1).
  """
  stack = []

  def down(t):
    stack.append(t)
    return tf.keras.layers.MaxPooling2D(2)(t)

  def up(t):
    doubled = tf.repeat(tf.repeat(t, 2, axis=-2), 2, axis=-3)
    return tf.concat([doubled, stack.pop()], axis=-1)

  def conv(kernel, c, name):
    return tf.keras.layers.Conv2D(
        c,
        kernel,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer='l2',
        name=name)

  spec = [
      conv(7, 32, 'down1'),
      conv(7, 32, 'down1b'), down,
      conv(5, 64, 'down2'),
      conv(5, 64, 'down2b'), down,
      conv(3, 128, 'down3'),
      conv(3, 128, 'down3b'), down,
      conv(3, 256, 'down4'),
      conv(3, 256, 'down4b'), down,
      conv(3, 512, 'down5'),
      conv(3, 512, 'down5b'), down,
      conv(3, 512, 'down6'),
      conv(3, 512, 'down6b'), down,
      conv(3, 512, 'down7'),
      conv(3, 512, 'down7b'), down,
      conv(3, 512, 'mid1'),
      conv(3, 512, 'mid2'), up,
      conv(3, 512, 'up7'),
      conv(3, 512, 'up7b'), up,
      conv(3, 512, 'up6'),
      conv(3, 512, 'up6b'), up,
      conv(3, 512, 'up5'),
      conv(3, 512, 'up5b'), up,
      conv(3, 256, 'up4'),
      conv(3, 256, 'up4b'), up,
      conv(3, 128, 'up3'),
      conv(3, 128, 'up3b'), up,
      conv(3, 64, 'up2'),
      conv(3, 64, 'up2b'), up,
      conv(3, 64, 'post1'),
      conv(3, 64, 'post2'),
      conv(3, 64, 'up1'),
      conv(3, 64, 'up1b')
  ]

  t = image
  for item in spec:
    t = item(t)

  output = tf.keras.layers.Conv2D(
      num_channels,
      3,
      padding='same',
      kernel_initializer='glorot_normal',
      kernel_regularizer='l2',
      name='output')(
          t)

  output = apply_harmonic_bias(output, num_layers)

  # This is equal to sigmoid(2 * output), but written as below it's clearer
  # that apply_harmonic_bias is using the correct inverse activation.
  output = (tf.tanh(output) + 1.0) / 2.0

  return output


def mpi_from_image(image):
  """A network to predict MPI layers from single images.

  Args:
    image: [B, H, W, 3] input image.

  Returns:
    [B, 32, H, W, 4] 32 RGBA layers, back to front
  """

  output = mpi_network(image, 34, 32)
  # The first 31 channels give us layer alpha, and the first (i.e. back) layer
  # is fully opaque.
  alpha = tf.transpose(output, [0, 3, 1, 2])[:, :-3, Ellipsis, tf.newaxis]
  layer_alpha = tf.concat([tf.ones_like(alpha[:, 0:1]), alpha], axis=1)

  # Color is a blend of foreground (input image) and background (predicted):
  foreground = image[:, tf.newaxis]
  background = output[:, tf.newaxis, Ellipsis, -3:]
  blend = tf.math.cumprod(
      1.0 - layer_alpha, axis=1, exclusive=True, reverse=True)
  layer_rgb = blend * foreground + (1.0 - blend) * background

  layers = tf.concat([layer_rgb, layer_alpha], axis=-1)
  return layers

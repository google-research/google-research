# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Network definitions for predicting flow basis."""

import tensorflow as tf
import utils


def unet_style_network(image, num_channels, regularization=1e-06):
  """Predict num_channels full-resolution output channels from input image.

  This is a U-Net style architecture with skip connections.

  Args:
    image: [B, H, W, C] input image.
    num_channels: number of output channels required.
    regularization: weight for per-layer regularization.

  Returns:
    [B, H, W, num_channels] output from network, in the range (0, 1).
    [] regularization_loss: sum of regularization losses like kernel weight l2
  """
  stack = []
  regularization_losses = []

  def down(t):
    stack.append(t)
    return tf.keras.layers.MaxPooling2D(2)(t)

  def up(t):
    return tf.concat([utils.double_size(t), stack.pop()], axis=-1)

  def conv(kernel, c, name):
    conv_layer = tf.keras.layers.Conv2D(
        c,
        kernel,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.L2(regularization),
        name=name)

    def process(x):
      output = conv_layer(x)
      regularization_losses.extend(conv_layer.losses)
      output = tf.keras.activations.relu(output)
      return output
    return process

  def convmin(kernel, c, name):
    return conv(kernel, max(c, num_channels), name)

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
      convmin(3, 512, 'mid2'), up,
      convmin(3, 512, 'up7'),
      convmin(3, 512, 'up7b'), up,
      convmin(3, 512, 'up6'),
      convmin(3, 512, 'up6b'), up,
      convmin(3, 512, 'up5'),
      convmin(3, 512, 'up5b'), up,
      convmin(3, 256, 'up4'),
      convmin(3, 256, 'up4b'), up,
      convmin(3, 128, 'up3'),
      convmin(3, 128, 'up3b'), up,
      convmin(3, 64, 'up2'),
      convmin(3, 64, 'up2b'), up,
      convmin(3, 64, 'post1'),
      convmin(3, 64, 'post2'),
      convmin(3, 32, 'up1'),
      convmin(3, 32, 'up1b')
  ]

  with tf.compat.v1.variable_scope('unet'):
    t = image
    for item in spec:
      t = item(t)

    output_layer = tf.keras.layers.Conv2D(
        num_channels,
        3,
        padding='same',
        kernel_initializer='glorot_normal',
        kernel_regularizer=tf.keras.regularizers.L2(regularization),
        name='output')

    regularization_losses.extend(output_layer.losses)
    output = output_layer(t)

  return output, tf.add_n(regularization_losses)


def _pad_to_multiple(image, factor, mode='SYMMETRIC'):
  """Pad image to a round size, for later unpadding.

  Args:
    image: [B, H, W, C].
    factor: Integer, padded dimensions will be multiples of this.
    mode: Padding mode to use (see tf.pad).
  Returns:
    (padded_image, unpad).
    padded_image is the input padded up to [B, H', W', C], H' and W' being the
    smallest multiples of factor not less than H and W.
    unpad is a function which takes an image [B, H', W', C'] (the network output
    will typically have a different number of channels from the input) and
    returns the central crop [B, H, W, C'] (i.e. it removes exactly the padding
    that was added, across all channels).
  """
  height = image.shape[1]
  width = image.shape[2]
  pad_x = (-width) % factor
  pad_y = (-height) % factor
  pad_left = pad_x // 2
  pad_right = pad_x - pad_left
  pad_top = pad_y // 2
  pad_bottom = pad_y - pad_top

  padded = tf.pad(
      image, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
      mode=mode)

  def unpad(data):
    return data[:, pad_top:(pad_top + height), pad_left:(pad_left + width), :]

  return padded, unpad


def predict_scene_representation(image, embedding_dimension):
  """A network to predict disparity, and optionally an object-instance embedding.

  The disparity output is in the range (0, 1), and the embedding is normalized
  to be a unit vector at each pixel.

  Args:
    image: [B, H, W, C] input image (RGB, range [0.0–1.0])
    embedding_dimension: Number of embedding dimensions to predict

  Returns:
    [B, H, W, 1] disparity.
    [B, embedding_dimension, H, W] embeddings, or None if
                                   embedding_dimension is 0
    {} losses, a dictionary of losses
  """
  losses = {}
  total_channels = 1 + embedding_dimension

  # The typical normalization used for imagenet images to make them roughly
  # standard-normal. See e.g.
  # https://paperswithcode.github.io/torchbench/imagenet/
  imagenet_mean = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 1, 3])
  imagenet_std = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 1, 3])
  image = (image - imagenet_mean) / imagenet_std

  padded, unpad = _pad_to_multiple(image, 128)
  predictions, regularization_loss = unet_style_network(padded, total_channels)
  losses['regularization'] = regularization_loss
  predictions = unpad(predictions)
  [disparity, embeddings] = tf.split(
      predictions, [1, embedding_dimension], axis=-1)

  disparity_activation_hinge = 5.0
  losses['disparity_activation'] = utils.batched_mean(
      tf.maximum(disparity - disparity_activation_hinge, 0))
  # Put disparity in range [0, 1].
  disparity = tf.keras.activations.sigmoid(disparity)

  if embedding_dimension:
    # embeddings is shape [B H W A]
    # Normalize and convert to [B A H W]
    embedding_norm = tf.norm(embeddings, ord=2, axis=-1, keepdims=True)
    losses['embedding_activation'] = utils.batched_mean(
        tf.maximum(tf.square(embedding_norm) - 1, 0))

    embeddings = embeddings/embedding_norm
    embeddings = tf.transpose(embeddings, [0, 3, 1, 2])
  else:
    embeddings = None

  return disparity, embeddings, losses

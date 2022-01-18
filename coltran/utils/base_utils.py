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

"""Various utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf


def nats_to_bits(nats):
  return nats / np.log(2)


def act_to_func(act):
  cond_act_map = {
      'relu': tf.nn.relu,
      'sigmoid': tf.math.sigmoid,
      'tanh': tf.math.tanh,
      'identity': lambda x: x}
  return cond_act_map[act]


def roll_channels_to_batch(tensor):
  # Switch from [B, H, W, C, D] to [B, C, H, W, D]
  return tf.transpose(tensor, perm=[0, 3, 1, 2, 4])


def roll_channels_from_batch(tensor):
  # Switch from [B, C, H, W, D] to [B, H, W, C, D]
  return tf.transpose(tensor, perm=[0, 2, 3, 1, 4])


def image_to_hist(image, num_symbols):
  """Returns a per-channel histogram of intensities.

  Args:
    image: 4-D Tensor, shape=(B, H, W, C), dtype=tf.int
    num_symbols: int
  Returns:
    hist: 3-D Tensor, shape=(B, C, num_symbols)
  """
  _, height, width, channels = image.shape
  image = tf.one_hot(image, depth=num_symbols)
  image = tf.reshape(image, shape=[-1, height*width, channels, num_symbols])
  # average spatially.
  image = tf.reduce_mean(image, axis=1)

  # smooth
  eps = 1e-8
  image = (image + eps) / (1 + num_symbols*eps)
  return image


def get_bw_and_color(inputs, colorspace):
  """Returns gray-scale and colored channels.

  Inputs are assumed to be in the RGB colorspace.

  Args:
    inputs: 4-D Tensor with 3 channels.
    colorspace: 'rgb' or 'ycbcr'
  Returns:
    grayscale: 4-D Tensor with 1 channel.
    inputs: 4=D Tensor with 3 channels.
  """
  if colorspace == 'rgb':
    grayscale = tf.image.rgb_to_grayscale(inputs)
  elif colorspace == 'ycbcr':
    inputs = rgb_to_ycbcr(inputs)
    grayscale, inputs = inputs[Ellipsis, :1], inputs[Ellipsis, 1:]
  return grayscale, inputs


def rgb_to_ycbcr(rgb):
  """Map from RGB to YCbCr colorspace."""
  rgb = tf.cast(rgb, dtype=tf.float32)
  r, g, b = tf.unstack(rgb, axis=-1)
  y = r * 0.299 + g * 0.587 + b * 0.114
  cb = r * -0.1687 - g * 0.3313 + b * 0.5
  cr = r * 0.5 - g * 0.4187 - b * 0.0813
  cb += 128.0
  cr += 128.0

  ycbcr = tf.stack((y, cb, cr), axis=-1)
  ycbcr = tf.clip_by_value(ycbcr, 0, 255)
  ycbcr = tf.cast(ycbcr, dtype=tf.int32)
  return ycbcr


def ycbcr_to_rgb(ycbcr):
  """Map from YCbCr to Colorspace."""
  ycbcr = tf.cast(ycbcr, dtype=tf.float32)
  y, cb, cr = tf.unstack(ycbcr, axis=-1)

  cb -= 128.0
  cr -= 128.0

  r = y * 1. + cb * 0. + cr * 1.402
  g = y * 1. - cb * 0.34414 - cr * 0.71414
  b = y * 1. + cb * 1.772 + cr * 0.

  rgb = tf.stack((r, g, b), axis=-1)
  rgb = tf.clip_by_value(rgb, 0, 255)
  rgb = tf.cast(rgb, dtype=tf.int32)
  return rgb


def convert_bits(x, n_bits_out=8, n_bits_in=8):
  """Quantize / dequantize from n_bits_in to n_bits_out."""
  if n_bits_in == n_bits_out:
    return x
  x = tf.cast(x, dtype=tf.float32)
  x = x / 2**(n_bits_in - n_bits_out)
  x = tf.cast(x, dtype=tf.int32)
  return x


def get_patch(upscaled, window, normalize=True):
  """Extract patch of size from upscaled.shape[1]//window from upscaled."""
  upscaled = tf.cast(upscaled, dtype=tf.float32)

  # pool + quantize + normalize
  patch = tf.nn.avg_pool2d(
      upscaled, ksize=window, strides=window, padding='VALID')

  if normalize:
    patch = tf.cast(patch, dtype=tf.float32)
    patch /= 256.0
  else:
    patch = tf.cast(patch, dtype=tf.int32)
  return patch


def labels_to_bins(labels, num_symbols_per_channel):
  """Maps each (R, G, B) channel triplet to a unique bin.

  Args:
    labels: 4-D Tensor, shape=(batch_size, H, W, 3).
    num_symbols_per_channel: number of symbols per channel.

  Returns:
    labels: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
  """
  labels = tf.cast(labels, dtype=tf.float32)
  channel_hash = [num_symbols_per_channel**2, num_symbols_per_channel, 1.0]
  channel_hash = tf.constant(channel_hash)
  labels = labels * channel_hash

  labels = tf.reduce_sum(labels, axis=-1)
  labels = tf.cast(labels, dtype=tf.int32)
  return labels


def bins_to_labels(bins, num_symbols_per_channel):
  """Maps back from each bin to the (R, G, B) channel triplet.

  Args:
    bins: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
    num_symbols_per_channel: number of symbols per channel.
  Returns:
    labels: 4-D Tensor, shape=(batch_size, H, W, 3)
  """
  labels = []
  factor = int(num_symbols_per_channel**2)

  for _ in range(3):
    channel = tf.math.floordiv(bins, factor)
    labels.append(channel)

    bins = tf.math.floormod(bins, factor)
    factor = factor // num_symbols_per_channel
  return tf.stack(labels, axis=-1)


# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Implementation of RAFT."""

import tensorflow as tf
from tensorflow_addons import image as tfa_image


def compute_corr(input_tensor):
  fmap1, fmap2 = input_tensor

  b, h, w, c = tf.unstack(tf.shape(fmap1))
  fmap1 = tf.reshape(fmap1, (b, h * w, c))
  fmap2 = tf.reshape(fmap2, (b, h * w, c))

  corr = tf.linalg.matmul(fmap1, tf.transpose(fmap2, [0, 2, 1]))
  corr = tf.reshape(corr, (b, h, w, 1, h, w))
  return corr / tf.math.sqrt(tf.cast(c, dtype=tf.float32))


# pylint:disable=missing-function-docstring
def corr_pyramid(input_tensor, num_levels=4, bidirectional=False):
  fmap1, fmap2 = input_tensor
  corr_pyramid_out = []
  # all pairs correlation
  corr_fw = compute_corr([fmap1, fmap2])
  b, h1, w1, c, h2, w2 = tf.unstack(tf.shape(corr_fw))

  if bidirectional:
    corr_bw = tf.transpose(corr_fw, [0, 4, 5, 3, 1, 2])
    corr_bw = tf.reshape(corr_bw, (b * h2 * w2, h1, w1, c))

  # Compute forward pyramid.
  corr_fw = tf.reshape(corr_fw, (b * h1 * w1, h2, w2, c))
  corr_pyramid_out.append(corr_fw)
  for _ in range(num_levels):
    corr_fw = tf.nn.avg_pool(corr_fw, ksize=2, strides=2, padding='VALID')
    corr_pyramid_out.append(corr_fw)

  # Compute backward pyramid.
  if bidirectional:
    corr_pyramid_out_bw = []
    corr_pyramid_out_bw.append(corr_bw)
    for _ in range(num_levels):
      corr_bw = tf.nn.avg_pool(corr_bw, ksize=2, strides=2, padding='VALID')
      corr_pyramid_out_bw.append(corr_bw)
    return {'fw': corr_pyramid_out, 'bw': corr_pyramid_out_bw}
  return {'fw': corr_pyramid_out}


def corr_block(corr_pyramid_inst, coords, radius):
  r = int(radius)
  b, h1, w1, _ = tf.unstack(tf.shape(coords))
  out_pyramid = []
  for i, corr in enumerate(corr_pyramid_inst):
    start = tf.cast(-r, dtype=tf.float32)
    stop = tf.cast(r, dtype=tf.float32)
    num = tf.cast(2 * r + 1, tf.int32)

    dx = tf.linspace(start, stop, num)
    dy = tf.linspace(start, stop, num)
    delta = tf.stack(tf.meshgrid(dy, dx), axis=-1)

    centroid_lvl = tf.reshape(coords, (b * h1 * w1, 1, 1, 2)) / 2**i
    delta_lvl = tf.reshape(delta, (1, 2 * r + 1, 2 * r + 1, 2))
    coords_lvl = tf.cast(
        centroid_lvl, dtype=tf.float32) + tf.cast(
            delta_lvl, dtype=tf.float32)

    corr = tfa_image.resampler(corr, coords_lvl)

    channel_dim = (2 * r + 1) * (2 * r + 1)
    corr = tf.reshape(corr, (b, h1, w1, channel_dim))
    out_pyramid.append(corr)
  out = tf.concat(out_pyramid, axis=-1)
  return out

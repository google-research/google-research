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


def coords_grid(b, h, w):
  coords = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
  coords = tf.cast(tf.stack(coords[::-1], axis=-1), dtype=tf.float32)
  coords = tf.expand_dims(coords, axis=0)
  coords = tf.repeat(coords, b, axis=0)
  return coords


def initialize_flow(b, h, w, division=8):
  coords0 = coords_grid(b, h // division, w // division)
  coords1 = coords_grid(b, h // division, w // division)
  return coords0, coords1


# size is (h, w)
def compute_upsample_flow(flow, size):
  upsampled_flow = tf.image.resize(flow, size)
  upsampled_x = upsampled_flow[:, :, :, 0] * tf.cast(
      size[1], dtype=tf.float32) / tf.cast(
          tf.shape(flow)[2], dtype=tf.float32)
  upsampled_y = upsampled_flow[:, :, :, 1] * tf.cast(
      size[0], dtype=tf.float32) / tf.cast(
          tf.shape(flow)[1], dtype=tf.float32)
  return tf.stack((upsampled_x, upsampled_y), axis=-1)

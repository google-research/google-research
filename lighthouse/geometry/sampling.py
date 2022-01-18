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

"""Bilinear and trilinear sampling gather functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def trilerp_gather(vol, inds, bad_inds=None):
  """Trilinear interpolation dense gather from volume at query inds."""

  inds_b = inds[Ellipsis, 0]
  inds_x = inds[Ellipsis, 1]
  inds_y = inds[Ellipsis, 2]
  inds_z = inds[Ellipsis, 3]

  inds_x_0 = tf.floor(inds_x)
  inds_x_1 = inds_x_0 + 1
  inds_y_0 = tf.floor(inds_y)
  inds_y_1 = inds_y_0 + 1
  inds_z_0 = tf.floor(inds_z)
  inds_z_1 = inds_z_0 + 1

  # store invalid indices to implement correct out-of-bounds conditions
  invalid_x = tf.logical_or(
      tf.less(inds_x_0, 0.0),
      tf.greater(inds_x_1, tf.to_float(tf.shape(vol)[2] - 1)))
  invalid_y = tf.logical_or(
      tf.less(inds_y_0, 0.0),
      tf.greater(inds_y_1, tf.to_float(tf.shape(vol)[1] - 1)))
  invalid_z = tf.logical_or(
      tf.less(inds_z_0, 0.0),
      tf.greater(inds_z_1, tf.to_float(tf.shape(vol)[3] - 1)))
  if bad_inds is not None:
    invalid_inds = tf.logical_or(
        tf.logical_or(tf.logical_or(invalid_x, invalid_y), invalid_z), bad_inds)
  else:
    invalid_inds = tf.logical_or(tf.logical_or(invalid_x, invalid_y), invalid_z)

  inds_x_0 = tf.clip_by_value(inds_x_0, 0.0, tf.to_float(tf.shape(vol)[2] - 2))
  inds_x_1 = tf.clip_by_value(inds_x_1, 0.0, tf.to_float(tf.shape(vol)[2] - 1))
  inds_y_0 = tf.clip_by_value(inds_y_0, 0.0, tf.to_float(tf.shape(vol)[1] - 2))
  inds_y_1 = tf.clip_by_value(inds_y_1, 0.0, tf.to_float(tf.shape(vol)[1] - 1))
  inds_z_0 = tf.clip_by_value(inds_z_0, 0.0, tf.to_float(tf.shape(vol)[3] - 2))
  inds_z_1 = tf.clip_by_value(inds_z_1, 0.0, tf.to_float(tf.shape(vol)[3] - 1))

  # compute interp weights
  w_x_0 = 1.0 - (inds_x - inds_x_0)
  w_x_1 = 1.0 - w_x_0
  w_y_0 = 1.0 - (inds_y - inds_y_0)
  w_y_1 = 1.0 - w_y_0
  w_z_0 = 1.0 - (inds_z - inds_z_0)
  w_z_1 = 1.0 - w_z_0

  w_0_0_0 = w_y_0 * w_x_0 * w_z_0
  w_1_0_0 = w_y_1 * w_x_0 * w_z_0
  w_0_1_0 = w_y_0 * w_x_1 * w_z_0
  w_0_0_1 = w_y_0 * w_x_0 * w_z_1
  w_1_1_0 = w_y_1 * w_x_1 * w_z_0
  w_0_1_1 = w_y_0 * w_x_1 * w_z_1
  w_1_0_1 = w_y_1 * w_x_0 * w_z_1
  w_1_1_1 = w_y_1 * w_x_1 * w_z_1

  # gather for interp
  inds_0_0_0 = tf.to_int32(
      tf.stack([inds_b, inds_y_0, inds_x_0, inds_z_0], axis=-1))
  inds_1_0_0 = tf.to_int32(
      tf.stack([inds_b, inds_y_1, inds_x_0, inds_z_0], axis=-1))
  inds_0_1_0 = tf.to_int32(
      tf.stack([inds_b, inds_y_0, inds_x_1, inds_z_0], axis=-1))
  inds_0_0_1 = tf.to_int32(
      tf.stack([inds_b, inds_y_0, inds_x_0, inds_z_1], axis=-1))
  inds_1_1_0 = tf.to_int32(
      tf.stack([inds_b, inds_y_1, inds_x_1, inds_z_0], axis=-1))
  inds_0_1_1 = tf.to_int32(
      tf.stack([inds_b, inds_y_0, inds_x_1, inds_z_1], axis=-1))
  inds_1_0_1 = tf.to_int32(
      tf.stack([inds_b, inds_y_1, inds_x_0, inds_z_1], axis=-1))
  inds_1_1_1 = tf.to_int32(
      tf.stack([inds_b, inds_y_1, inds_x_1, inds_z_1], axis=-1))

  vol_0_0_0 = tf.gather_nd(vol, inds_0_0_0) * w_0_0_0[Ellipsis, tf.newaxis]
  vol_1_0_0 = tf.gather_nd(vol, inds_1_0_0) * w_1_0_0[Ellipsis, tf.newaxis]
  vol_0_1_0 = tf.gather_nd(vol, inds_0_1_0) * w_0_1_0[Ellipsis, tf.newaxis]
  vol_0_0_1 = tf.gather_nd(vol, inds_0_0_1) * w_0_0_1[Ellipsis, tf.newaxis]
  vol_1_1_0 = tf.gather_nd(vol, inds_1_1_0) * w_1_1_0[Ellipsis, tf.newaxis]
  vol_0_1_1 = tf.gather_nd(vol, inds_0_1_1) * w_0_1_1[Ellipsis, tf.newaxis]
  vol_1_0_1 = tf.gather_nd(vol, inds_1_0_1) * w_1_0_1[Ellipsis, tf.newaxis]
  vol_1_1_1 = tf.gather_nd(vol, inds_1_1_1) * w_1_1_1[Ellipsis, tf.newaxis]

  out_vol = vol_0_0_0 + vol_1_0_0 + vol_0_1_0 + vol_0_0_1 + \
            vol_1_1_0 + vol_0_1_1 + vol_1_0_1 + vol_1_1_1

  # boundary conditions for invalid indices
  invalid_inds = tf.tile(invalid_inds[:, :, :, :, tf.newaxis],
                         [1, 1, 1, 1, tf.shape(vol)[4]])
  out_vol = tf.where(invalid_inds, tf.zeros_like(out_vol), out_vol)

  return out_vol


def bilerp_gather(img, inds):
  """Bilinear interpolation dense gather from image at query inds."""

  inds_b, _, _, = tf.meshgrid(
      tf.range(tf.shape(img)[0]),
      tf.range(tf.shape(img)[1]),
      tf.range(tf.shape(img)[2]),
      indexing='ij')

  inds_b = tf.to_float(inds_b)
  inds_x = inds[Ellipsis, 0]
  inds_y = inds[Ellipsis, 1]

  inds_x_0 = tf.floor(inds_x)
  inds_x_1 = inds_x_0 + 1
  inds_y_0 = tf.floor(inds_y)
  inds_y_1 = inds_y_0 + 1

  # store invalid indices to implement correct out-of-bounds conditions
  invalid_x = tf.logical_or(
      tf.less(inds_x_0, 0.0),
      tf.greater(inds_x_1, tf.to_float(tf.shape(img)[2] - 1)))
  invalid_y = tf.logical_or(
      tf.less(inds_y_0, 0.0),
      tf.greater(inds_y_1, tf.to_float(tf.shape(img)[1] - 1)))
  invalid_inds = tf.logical_or(invalid_x, invalid_y)

  inds_x_0 = tf.clip_by_value(inds_x_0, 0.0, tf.to_float(tf.shape(img)[2] - 2))
  inds_x_1 = tf.clip_by_value(inds_x_1, 0.0, tf.to_float(tf.shape(img)[2] - 1))
  inds_y_0 = tf.clip_by_value(inds_y_0, 0.0, tf.to_float(tf.shape(img)[1] - 2))
  inds_y_1 = tf.clip_by_value(inds_y_1, 0.0, tf.to_float(tf.shape(img)[1] - 1))

  # compute interp weights
  w_x_0 = 1.0 - (inds_x - inds_x_0)
  w_x_1 = 1.0 - w_x_0
  w_y_0 = 1.0 - (inds_y - inds_y_0)
  w_y_1 = 1.0 - w_y_0

  w_0_0 = w_y_0 * w_x_0
  w_1_0 = w_y_1 * w_x_0
  w_0_1 = w_y_0 * w_x_1
  w_1_1 = w_y_1 * w_x_1

  # gather for interp
  inds_0_0 = tf.to_int32(tf.stack([inds_b, inds_y_0, inds_x_0], axis=-1))
  inds_1_0 = tf.to_int32(tf.stack([inds_b, inds_y_1, inds_x_0], axis=-1))
  inds_0_1 = tf.to_int32(tf.stack([inds_b, inds_y_0, inds_x_1], axis=-1))
  inds_1_1 = tf.to_int32(tf.stack([inds_b, inds_y_1, inds_x_1], axis=-1))

  img_0_0 = tf.gather_nd(img, inds_0_0) * w_0_0[Ellipsis, tf.newaxis]
  img_1_0 = tf.gather_nd(img, inds_1_0) * w_1_0[Ellipsis, tf.newaxis]
  img_0_1 = tf.gather_nd(img, inds_0_1) * w_0_1[Ellipsis, tf.newaxis]
  img_1_1 = tf.gather_nd(img, inds_1_1) * w_1_1[Ellipsis, tf.newaxis]

  out_img = img_0_0 + img_1_0 + img_0_1 + img_1_1

  # boundary conditions for invalid indices
  invalid_inds = tf.tile(invalid_inds[:, :, :, tf.newaxis],
                         [1, 1, 1, tf.shape(img)[3]])

  out_img = tf.where(invalid_inds, tf.zeros_like(out_img), out_img)

  return out_img

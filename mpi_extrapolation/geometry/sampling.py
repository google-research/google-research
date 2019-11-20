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

"""Bilinear sampling gather functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def bilinear_wrapper(imgs, coords):
  """Wrapper around bilinear sampling function, handles arbitrary input sizes.

  Args:
    imgs: are [B,H,W,C]
    coords: [B,H,W,2] indicating the source pixels to copy from
  Returns:
    [B,H,W,C] images after bilinear sampling from input.
  """
  # the bilinear sampling code only handles 4D input, so we'll need to reshape
  init_dims = tf.shape(imgs)[:-3]
  end_dims_img = tf.shape(imgs)[-3:]
  end_dims_coords = tf.shape(coords)[-3:]

  prod_init_dims = tf.reduce_prod(init_dims)

  imgs = tf.reshape(
      imgs, tf.concat([prod_init_dims[tf.newaxis], end_dims_img], axis=0))
  coords = tf.reshape(
      coords, tf.concat([prod_init_dims[tf.newaxis], end_dims_coords], axis=0))
  imgs_sampled = tf.contrib.resampler.resampler(imgs, coords)
  imgs_sampled = tf.reshape(
      imgs_sampled, tf.concat([init_dims, tf.shape(imgs_sampled)[-3:]], axis=0))
  return imgs_sampled

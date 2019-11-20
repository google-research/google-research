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

"""Image preprocessing functions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import numpy as np
import sonnet as snt
import tensorflow as tf

from stacked_capsule_autoencoders.capsules.tensor_ops import ensure_length


def pad_and_shift(img, output_size, shift=None, pad_kwargs=None):
  """Pads and shifts the image.

  Args:
    img: img [H, W, C] or a batch of images [B, H, W, C].
    output_size: int or a tuple of ints.
    shift: Maximum shift along y and x axes. None, int or a tuple of ints. If
      None, the shift for each axis is computed as output_size[i] - img_size[i].
    pad_kwargs: dict of kwargs passed to tf.pad.

  Returns:
    Padded images.
  """

  output_size = ensure_length(output_size, 2)
  shift = ensure_length(shift, 2)

  if img.shape.ndims == 4:
    func = functools.partial(
        pad_and_shift,
        output_size=output_size,
        shift=shift,
        pad_kwargs=pad_kwargs)

    return tf.map_fn(func, img)

  for i, z in enumerate(img.shape[:2].as_list()):
    if shift[i] is None:
      shift[i] = output_size[i] - z

  y_shift, x_shift = shift
  x = tf.random.uniform([], -x_shift, x_shift + 1, dtype=tf.int32)
  y = tf.random.uniform([], -y_shift, y_shift + 1, dtype=tf.int32)
  y1, y2 = abs(tf.minimum(y, 0)), tf.maximum(y, 0)
  x1, x2 = abs(tf.minimum(x, 0)), tf.maximum(x, 0)

  height, width = output_size
  h = int(img.shape[0]) + abs(y)
  y_pad = tf.maximum(height - h, 0)
  y1 += y_pad // 2 + y_pad % 2
  y2 += y_pad // 2

  w = int(img.shape[1]) + abs(x)
  x_pad = tf.maximum(width - w, 0)
  x1 += x_pad // 2 + x_pad % 2
  x2 += x_pad // 2

  if pad_kwargs is None:
    pad_kwargs = dict()

  img = tf.pad(img, [(y1, y2), (x1, x2), (0, 0)], **pad_kwargs)
  img.set_shape([height, width, img.shape[-1]])

  return img[:height, :width]



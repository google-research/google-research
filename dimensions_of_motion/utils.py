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
"""Utility functions."""

import tensorflow as tf


def collapse_dim(tensor, axis):
  """Collapses one axis of a tensor into the preceding axis.

  This is a fast operation since it just involves reshaping the
  tensor.

  Example:
    a = [[[1,2], [3,4]], [[5,6], [7,8]]]

    collapse_dim(a, -1) = [[1,2,3,4], [5,6,7,8]]
    collapse_dim(a, 1) = [[1,2], [3,4], [5,6], [7,8]]

  Args:
    tensor: a tensor of shape [..., Di-1, Di, ...]
    axis: the axis to collapse, i, in the range (-n, n). The first axis may not
      be collapsed.

  Returns:
    a tensor of shape [..., Di-1 * Di, ...] containing the same values.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.shape(tensor)
  # We want to extract the parts of the shape that should remain unchanged.
  # Naively one would write shape[:axis-1] or shape[axis+1:] for this, but
  # this will be wrong if, for example, axis is -1. So the safe way is to
  # first slice using [:axis] or [axis:] and then remove an additional element.
  newshape = tf.concat([shape[:axis][:-1], [-1], shape[axis:][1:]], 0)
  return tf.reshape(tensor, newshape)


def batched_mean(tensor):
  """Reduce a tensor to its mean, retaining the first (batch) dimension."""
  axes = tf.range(1, tf.rank(tensor))
  return tf.reduce_mean(tensor, axis=axes)


def flatten_batch(tensor, axes):
  """Reshape a tensor to collapse multiple axes into a single batch axis.

  This is useful when you are working with multiple layers of batching, but you
  need to call functions that assume only one layer of batching, and then
  convert the output back to the shape with multiple layers of batching.

  Args:
    tensor: a tensor of shape [D0, ... Dn-1].
    axes: the number of initial axes i to collapse. i <= n.

  Returns:
    output: A tensor which contains the same values as input, but which has
      shape [P, Di, Di+1, ... Dn-1] where P is the product D0 * D1 * .. Di-1.
      The sizes D0, ... Di-1 must be statically known.
    unflatten: A function which can be applied to any tensor of known shape
      [P, ...] to convert back to shape [D0, D1, ... Di-1, ...].

  Raises:
    ValueError: if you attempt to flatten_batch tensor of insufficiently known
      shape, or unflatten a tensor with incompatible shape.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.shape(tensor)
  prefix = shape[:axes]
  rest = shape[axes:]
  static_shape = tensor.shape.as_list()
  product = 1
  for size in static_shape[:axes]:
    if size is None:
      raise ValueError(
          'flatten_batch requires batch dimensions to be statically known (%s).'
          % static_shape[:axes])
    product *= size
  output = tf.reshape(tensor, tf.concat([tf.constant([product]), rest], 0))

  def unflatten(flattened):
    flattened_shape = tf.shape(flattened)
    return tf.reshape(flattened, tf.concat([prefix, flattened_shape[1:]], 0))

  return output, unflatten


def double_size(image):
  """Double the size of an image or batch of images.

  This just duplicates each pixel into a 2x2 block – i.e. nearest-neighbor
  upsampling. The result is identical to using tf.image.resize_area to double
  the size, with the addition that we can take the gradient.

  Args:
    image: [..., H, W, C]

  Returns:
    [..., H*2, W*2, C] scaled up.
  """
  image = tf.convert_to_tensor(image)
  shape = image.shape.as_list()
  multiples = [1] * (len(shape) - 2) + [2, 2]
  tiled = tf.tile(image, multiples)
  newshape = shape[:-3] + [shape[-3] * 2, shape[-2] * 2, shape[-1]]
  return tf.reshape(tiled, newshape)

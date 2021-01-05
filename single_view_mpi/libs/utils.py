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

# -*- coding: utf-8 -*-
"""Utility functions for parallax learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def name_scope(target):
  """A decorator to add a tf name scope for a function."""
  name = target.__name__

  def wrapper(*args, **kwargs):
    with tf.name_scope(name):
      return target(*args, **kwargs)
  return wrapper


def broadcast_to_match(a, b, ignore_axes=0):
  """Returns (a', b') which are the inputs broadcast up to have the same shape.

  Suppose you want to apply an operation to tensors a and b but it doesn't
  support broadcasting. As an example maybe we have tensors of these shapes:
    a    [5, 1, 3, 4]
    b [2, 1, 8, 4, 2]
  Considering the last two dimensions as matrices, we may want to multiply
  a by b to get a tensor [2, 5, 8, 3, 2] of (2x3) matrices. However, tf.matmul
  doesn't support this because the outer dimensions don't match. Calling
  tf.matmul(a, b) directly will fail.

  However, the dimensions do match under broadcasting, so we can do the
  multiplication like this:
    a, b = broadcast_to_match(a, b, ignore_axes=2)
    c = tf.matmul(a, b)
  The ignore_axes parameter tells us to ignore the last two dimensions of a
  and b and just make the rest match.

  Args:
    a: Any shape
    b: Any shape
    ignore_axes: If present, broadcasting will not apply to the final this many
      axes. For example, if you are planning to call tf.matmul(a, b) on the
      result, then set ignore_axes=2 because tf.matmul operates on the last two
      axes, only the rest need to match. To ignore a different number of axes
      for inputs a and b, pass a pair of number to ignore_axes.

  Returns:
    a', b': Identical to the two inputs except tiled so that the shapes
        match. See https://www.tensorflow.org/performance/xla/broadcasting.
        If the shapes already match, no tensorflow graph operations are added,
        so this is cheap.
  """
  a = tf.convert_to_tensor(a)
  b = tf.convert_to_tensor(b)
  a_shape = a.shape.as_list()
  b_shape = b.shape.as_list()
  # Extract the part of the shape that is required to match.
  if isinstance(ignore_axes, tuple) or isinstance(ignore_axes, list):
    ignore_a = ignore_axes[0]
    ignore_b = ignore_axes[1]
  else:
    ignore_a = ignore_axes
    ignore_b = ignore_axes
  if ignore_a:
    a_shape = a_shape[:-ignore_a]
  if ignore_b:
    b_shape = b_shape[:-ignore_b]
  if a_shape == b_shape:
    return (a, b)
  # Addition supports broadcasting, so add a tensor of zeroes.
  za = tf.zeros(a_shape + [1] * ignore_b, dtype=b.dtype)
  zb = tf.zeros(b_shape + [1] * ignore_a, dtype=a.dtype)
  a += zb
  b += za

  a_new_shape = a.shape.as_list()
  b_new_shape = b.shape.as_list()
  if ignore_a:
    a_new_shape = a_new_shape[:-ignore_a]
  if ignore_b:
    b_new_shape = b_new_shape[:-ignore_b]
  assert a_new_shape == b_new_shape
  return (a, b)


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


def split_dim(tensor, axis, factor):
  """Splits a dimension into two dimensions.

  Opposite of collapse_dim.

  Args:
    tensor: an n-dimensional tensor of shape [..., Di, ...]
    axis: the axis to split, i, in the range [-n, n)
    factor: the size of the first of the two resulting axes. Must divide Di.

  Returns:
    an (n+1)-dimensional tensor of shape [..., factor, Di / factor, ...]
    containing the same values as the input tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.shape(tensor)
  newshape = tf.concat(
      [shape[:axis], [factor, shape[axis] // factor], shape[axis:][1:]], 0)
  return tf.reshape(tensor, newshape)


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
          'flatten_batch requires batch dimensions to be statically known.' %
          static_shape[:axes])
    product *= size
  output = tf.reshape(tensor, tf.concat([tf.constant([product]), rest], 0))

  def unflatten(flattened):
    flattened_shape = tf.shape(flattened)
    return tf.reshape(flattened, tf.concat([prefix, flattened_shape[1:]], 0))

  return output, unflatten

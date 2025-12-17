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

"""Provides `engine`: instance of `ComputeEngine` wraps TensorFlow functions."""
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from sparse_deferred.implicit import matrix

Tensor = tf.Tensor
Shape = tf.TensorShape|list[int]
DType = tf.DType|str


class _TFEngine(matrix.ComputeEngine):
  """Implements tensorflow as a `ComputeEngine`."""

  def where(
      self, condition, val_if_true, val_if_false
  ):
    return tf.where(condition, val_if_true, val_if_false)

  def assert_equal(self, tensor1, tensor2):
    return tf.assert_equal(tensor1, tensor2)

  def assert_greater(self, tensor1, tensor2):
    return tf.assert_greater(tensor1, tensor2)

  def ones(self, sizes, dtype = 'float32'):
    return tf.ones(sizes, dtype=dtype)

  def abs(self, tensor):
    return tf.abs(tensor)

  def sign(self, tensor):
    return tf.sign(tensor)

  def rsqrt(self, tensor):
    return tf.math.rsqrt(tensor)

  def ones_like(self, tensor):
    return tf.ones_like(tensor)

  def transpose(self, tensor):
    return tf.transpose(tensor)

  def einsum(self, notation, a, b):
    return tf.einsum(notation, a, b)

  def add_n(self, tensors):
    return tf.add_n(tensors)

  def shape(self, tensor):
    return tf.shape(tensor)

  def eye(self, num_rows, dtype='float32'):
    return tf.eye(num_rows, dtype=dtype)

  def cast(self, tensor, dtype = 'float32'):
    return tf.cast(tensor, dtype)

  def cumsum(self, tensor, axis = 0):
    return tf.cumsum(tensor, axis)

  def minimum(self, x, y):
    return tf.minimum(x, y)

  def argmax(self, tensor, axis = None):
    return tf.argmax(tensor, axis=axis)

  def argsort(self, tensor, axis=-1, direction='ASCENDING'):
    return tf.argsort(tensor, axis=axis, direction=direction)

  def all(
      self, tensor, axis = None, keepdims=False
  ):
    return tf.reduce_all(tensor, axis=axis, keepdims=keepdims)

  def gather(self, tensor, indices, axis = 0):
    return tf.gather(tensor, indices, axis=axis)

  def gather_nd(self, tensor, indices):
    return tf.gather_nd(tensor, indices)

  def unsorted_segment_sum(
      self, data, segment_ids, num_segments
  ):
    return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)

  def concat(self, tensors, axis):
    return tf.concat(tensors, axis=axis)

  def stack(self, tensors, axis):
    return tf.stack(tensors, axis=axis)

  def zeros(self, shape, dtype = 'float32'):
    return tf.zeros(shape, dtype=dtype)

  def reshape(self, tensor, shape):
    return tf.reshape(tensor, shape)

  def boolean_mask(self, tensor, mask):
    return tf.boolean_mask(tensor, mask)

  def reduce_all(
      self,
      tensor,
      axis = None,
      keepdims = False,
  ):
    return tf.reduce_all(tensor, axis=axis, keepdims=keepdims)

  def reduce_any(
      self,
      tensor,
      axis = None,
      keepdims = False,
  ):
    return tf.reduce_any(tensor, axis=axis, keepdims=keepdims)

  def reduce_max(
      self,
      tensor,
      axis = None,
      keepdims = False,
  ):
    return tf.reduce_max(tensor, axis=axis, keepdims=keepdims)

  def maximum(self, x, y):
    return tf.math.maximum(x, y)

  def range(self, up_to, dtype = 'float32'):
    return tf.range(up_to, dtype=dtype)

  def one_hot(self, tensor, num_classes):
    return tf.one_hot(tensor, num_classes)

  def random_normal(
      self,
      shape,
      mean = 0.0,
      stddev = 1.0,
      dtype = 'float32',
      seed = None,
  ):
    return tf.random.normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )

  def matrix_rank(self, a, tol = None):
    return tf.linalg.matrix_rank(a, tol=tol)

  def matmul(self, a, b):
    return tf.matmul(a, b)

  def svd(
      self, a, full_matrices = False
  ):
    s, u, v = tf.linalg.svd(a, full_matrices=full_matrices)
    return u, s, v

  def qr(self, a):
    return tf.linalg.qr(a)

  def to_cpu(self, tensor):
    """Brings a tensor to the CPU, so that python can access it."""
    return np.array(tensor)

  def fill_padding(self, tensor, bigger_padding):
    """Scatter_nd_update is equivalent to bigger_padding[indices] = tensor."""
    if tf.size(tensor) == 0:
      return bigger_padding
    range_shape = tf.shape(tensor)
    indices = tf.range(range_shape[0], dtype=tf.int32)[:, tf.newaxis]
    return tf.tensor_scatter_nd_update(bigger_padding, indices, tensor)


engine = _TFEngine()

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

"""Defines `NumpyMatrix` and its associated `engine: matrix.ComputeEngine`."""

from typing import Optional

import numpy as np
import sparse_deferred as sd

Tensor = np.ndarray
Shape = sd.Shape
DType = str|np.dtype


class NumpyMatrix(sd.Matrix):
  """Wraps numpy arrays. Multiplication is achieved by reshape-mult-reshape."""

  def __init__(self, np_arr):
    super().__init__()
    self.set_engine(engine)
    assert len(np_arr.shape) == 2
    self._np_arr = np_arr

  def matmul(self, mat):
    first_dim = mat.shape[0]
    other_shape = list(mat.shape[1:])
    return self._np_arr.dot(np.reshape(mat, (first_dim, -1))).reshape(
        [-1] + other_shape)

  def rmatmul(self, mat):
    first_dims = list(mat.shape[:-1])
    last_dim = mat.shape[-1]
    return np.reshape(mat, [-1, last_dim]).dot(self._np_arr).reshape(
        first_dims + [-1])

  @property
  def shape(self):
    return self._np_arr.shape


class _NumpyEngine(sd.ComputeEngine):
  """Implements Numpy as a `ComputeEngine`."""

  def where(self, condition, val_if_true,
            val_if_false):
    return np.where(condition, val_if_true, val_if_false)

  def assert_equal(self, array1, array2):
    assert np.allclose(array1, array2)

  def assert_greater(self, array1, array2):
    assert (array1 > array2).all()

  def ones(self, sizes, dtype = 'float32'):
    return np.ones(sizes, dtype=dtype)

  def abs(self, array):
    return np.abs(array)

  def rsqrt(self, array):
    return 1 / np.sqrt(array)

  def ones_like(self, array):
    return np.ones_like(array)

  def transpose(self, array):
    return array.T

  def einsum(self, notation, a, b):
    return np.einsum(notation, a, b)

  def add_n(self, arrays):
    return np.sum(arrays, axis=0)

  def shape(self, array):
    return np.shape(array)

  def eye(self, num_rows, dtype='float32'):
    return np.eye(num_rows, dtype=dtype)

  def cast(self, tensor, dtype = 'float32'):
    if dtype == 'string':
      dtype = 'object'
    return np.array(tensor, dtype=dtype)

  def minimum(self, x, y):
    return np.minimum(x, y)

  def argsort(self, tensor, axis = -1,
              direction = 'ASCENDING'):
    if direction != 'ASCENDING':
      tensor *= -1
    return np.argsort(tensor, axis=axis)

  def all(self, tensor, axis = None,
          keepdims = False):
    return np.all(tensor, axis=axis, keepdims=keepdims)

  def to_cpu(self, tensor):
    return tensor  # Already on CPU!

  def gather(self, tensor, indices, axis = 0
             ):
    return np.take(tensor, indices, axis=axis)

  def unsorted_segment_sum(self, data, segment_ids,
                           num_segments):
    sum_tensor = np.zeros([num_segments] + list(data.shape[1:]))
    np.add.at(sum_tensor, segment_ids, data)
    return sum_tensor

  def concat(self, tensors, axis):
    return np.concatenate(tensors, axis=axis)

  def zeros(self, shape, dtype = 'float32'):
    return np.zeros(shape, dtype=dtype)

  def reshape(self, tensor, shape):
    return np.reshape(tensor, shape)

  def boolean_mask(self, tensor, mask):
    return tensor[mask]

  def reduce_all(self, tensor, axis = None,
                 keepdims = False):
    return np.all(tensor, axis=axis, keepdims=keepdims)

  def reduce_any(self, tensor, axis = None,
                 keepdims = False):
    return np.any(tensor, axis=axis, keepdims=keepdims)

  def maximum(self, x, y):
    return np.maximum(x, y)

  def range(self, up_to, dtype = 'float32'):
    return np.arange(up_to, dtype=dtype)

  def one_hot(self, tensor, num_classes):
    onehot = np.zeros((tensor.shape[0], num_classes), dtype='float32')
    r = np.arange(tensor.shape[0], dtype='int32')
    r = r[tensor >= 0]
    tensor = tensor[tensor >= 0]
    onehot[r, tensor] = 1.0
    return onehot

  def cumsum(self, tensor, axis = 0):
    return np.cumsum(tensor, axis)

engine = _NumpyEngine()

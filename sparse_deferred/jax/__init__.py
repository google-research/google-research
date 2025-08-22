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

"""Defines `JaxMatrix` and its associated `engine: matrix.ComputeEngine`."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import sparse_deferred as sd

Tensor = jax.Array
Shape = sd.Shape
DType = str|jnp.dtype


class JaxMatrix(sd.Matrix):
  """Wraps jax arrays. Multiplication is achieved by reshape-mult-reshape."""

  def __init__(self, jax_arr):
    super().__init__()
    self.set_engine(engine)
    assert len(jax_arr.shape) == 2
    self._jax_arr = jax_arr

  def matmul(self, mat):
    first_dim = mat.shape[0]
    other_shape = list(mat.shape[1:])
    return jnp.reshape(
        jnp.dot(self._jax_arr, jnp.reshape(mat, (first_dim, -1))),
        [-1] + other_shape,
    )

  def rmatmul(self, mat):
    first_dims = list(mat.shape[:-1])
    last_dim = mat.shape[-1]
    return jnp.reshape(
        jnp.dot(jnp.reshape(mat, [-1, last_dim]), self._jax_arr),
        first_dims + [-1],
    )

  @property
  def shape(self):
    return self._jax_arr.shape


class _JaxEngine(sd.ComputeEngine):
  """Implements JAX as a `ComputeEngine`."""

  def where(self, condition, val_if_true,
            val_if_false):
    return jnp.where(condition, val_if_true, val_if_false)

  def assert_equal(self, array1, array2):
    return np.allclose(array1, array2)

  def assert_greater(self, array1, array2):
    assert (array1 > array2).all()

  def ones(self, sizes, dtype = 'float32'):
    return jnp.ones(sizes, dtype=dtype)

  def abs(self, array):
    return jnp.abs(array)

  def rsqrt(self, array):
    return 1 / jnp.sqrt(array)

  def ones_like(self, array):
    return jnp.ones_like(array)

  def transpose(self, array):
    return jnp.transpose(array)

  def einsum(self, notation, a, b):
    return jnp.einsum(notation, a, b)

  def add_n(self, arrays):
    return jnp.sum(jnp.array(arrays), axis=0)

  def shape(self, array):
    return np.shape(array)

  def eye(self, num_rows, dtype='float32'):
    return jnp.eye(num_rows, dtype=dtype)

  def cast(self, tensor, dtype = 'float32'):
    return jnp.array(tensor, dtype=dtype)

  def cumsum(self, tensor, axis = 0):
    return jnp.cumsum(tensor, axis)

  def minimum(self, x, y):
    return jnp.minimum(x, y)

  def argsort(self, tensor, axis = -1,
              direction = 'ASCENDING'):
    if direction != 'ASCENDING':
      tensor *= -1
    return jnp.argsort(tensor, axis=axis)

  def all(self, tensor, axis = None,
          keepdims = False):
    return jnp.all(tensor, axis=axis, keepdims=keepdims)

  def to_cpu(self, tensor):
    return tensor  # Already on CPU!

  def gather(self, tensor, indices, axis = 0
             ):
    return jnp.take(tensor, indices, axis=axis)

  def unsorted_segment_sum(self, data, segment_ids,
                           num_segments):
    return jax.ops.segment_sum(data, segment_ids, num_segments)

  def concat(self, tensors, axis):
    return jnp.concatenate(tensors, axis=axis)

  def zeros(self, shape, dtype = 'float32'):
    return jnp.zeros(shape, dtype=dtype)

  def reshape(self, tensor, shape):
    return jnp.reshape(tensor, shape)

  def boolean_mask(self, tensor, mask):
    return tensor[mask]

  def reduce_all(self, tensor, axis = None,
                 keepdims = False):
    return jnp.all(tensor, axis=axis, keepdims=keepdims)

  def reduce_any(self, tensor, axis = None,
                 keepdims = False):
    return jnp.any(tensor, axis=axis, keepdims=keepdims)

  def maximum(self, x, y):
    return jnp.maximum(x, y)

  def range(self, up_to, dtype = 'float32'):
    return jnp.arange(up_to, dtype=dtype)

  def one_hot(self, tensor, num_classes):
    return jax.nn.one_hot(tensor, num_classes)

engine = _JaxEngine()

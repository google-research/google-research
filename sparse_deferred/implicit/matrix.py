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

"""`Matrix` objects and functions to construct linear deferred computation.

The computation is a directed acyclic graph. Leaf nodes are data matrices, e.g.,
adjacency matrix `A`, feature matrix `X`, and root node representing a
calculation of interest, e.g., `U @ X`, where `U` is a singular basis of a
linear function of `A`.

To be able to compute products of `Matrix` instance, e.g., `U @ X`, you must
provide `ComputeEngine` instance. We have `sparse_deferred.tf.engine` as the
only implementation, but we also plan to add `sparse_deferred.jax.engine`.
"""
import abc
from typing import Any, Optional

EPSILON = 1e-6  # To prevent division by 0.

Tensor = Any
Shape = list[int]|Any
DType = str|Any


class ComputeEngine(abc.ABC):
  """Interface promises functionality from a compute engines.

  For the most part, each function should probably be a single statament.
  """

  @abc.abstractmethod
  def where(self, condition, val_if_true,
            val_if_false):
    """`result[i] == condition[i] ? val_if_true[i] : val_if_false[i]`."""

  @abc.abstractmethod
  def assert_equal(self, tensor1, tensor2):
    """Should assert that 2 `Tensor`s are equal -- can be deferred on device."""

  @abc.abstractmethod
  def assert_greater(self, a, b):
    """Asserts `a` is greater than `b` everywhere. Can be deferred on device."""

  @abc.abstractmethod
  def ones(self, sizes, dtype = 'float32'):
    """Returns all-ones `Tensor` with given `sizes`."""

  @abc.abstractmethod
  def abs(self, tensor):
    """Finds element-wise absolute value."""

  @abc.abstractmethod
  def rsqrt(self, tensor):
    """Returns element-wise `tensor ** -0.5`."""

  @abc.abstractmethod
  def ones_like(self, tensor):
    """Returns all-ones `Tensor` with same shape and dtype as `tensor`."""

  @abc.abstractmethod
  def transpose(self, tensor):
    """Reverses dimensions of `tensor`."""

  @abc.abstractmethod
  def einsum(self, notation, a, b):
    """Computes product (`a . b`) summing on axes per einsum `notation`."""

  @abc.abstractmethod
  def add_n(self, tensors):
    """Adds list of tensors."""

  @abc.abstractmethod
  def shape(self, tensor):
    """Returns shape of `tensor`."""

  @abc.abstractmethod
  def eye(self, num_rows, dtype = 'float32'):
    """Identity matrix `Tensor` with shape (num_rows, num_rows)."""

  @abc.abstractmethod
  def minimum(self, x, y):
    """Returns `result[i] = min(x[i], y[i])`. where `x, y` are of same shape."""

  @abc.abstractmethod
  def cast(self, tensor, dtype = 'float32'):
    """Casts `tensor` to `dtype`."""

  @abc.abstractmethod
  def cumsum(self, x, axis = 0):
    """Compute the cumulative sum of the tensor along `axis`."""

  @abc.abstractmethod
  def argsort(self, tensor, axis = -1,
              direction = 'ASCENDING'):
    """Returns argsort for a vector."""

  @abc.abstractmethod
  def all(self, tensor, axis = None,
          keepdims = False):
    """Reduce boolean tensor."""

  @abc.abstractmethod
  def to_cpu(self, tensor):
    """Brings a tensor to the CPU, so that python can access it."""

  @abc.abstractmethod
  def gather(self, tensor, indices, axis = 0):
    """Returns tensor of shape concat(indices.shape, tensor.shape[1:])."""

  @abc.abstractmethod
  def concat(self, tensors, axis):
    """Concatenate a list of tensors along the given axis."""

  @abc.abstractmethod
  def unsorted_segment_sum(self, data, segment_ids,
                           num_segments):
    """Like `tf.math.unsorted_segment_sum`."""

  @abc.abstractmethod
  def zeros(self, shape, dtype = 'float32'):
    """Returns tensor of all-zeros of shape `shape`."""

  @abc.abstractmethod
  def reshape(self, tensor, shape):
    """Reshapes `tensor` into shape `shape`."""

  @abc.abstractmethod
  def boolean_mask(self, tensor, mask):
    """`mask` should be vector of size `tensor.shape[0]`."""

  @abc.abstractmethod
  def reduce_all(self, tensor, axis = None,
                 keepdims = False):
    """Does logical-and (across axis) of a tensor."""

  def reduce_any(self, tensor, axis = None,
                 keepdims = False):
    """Does logical-or (across axis) of a tensor."""

  @abc.abstractmethod
  def maximum(self, x, y):
    """Element-wise maximum between two tensors."""

  @abc.abstractmethod
  def range(self, up_to, dtype = 'float32'):
    """Like `np.arange` or `tf.range`."""

  @abc.abstractmethod
  def one_hot(self, tensor, num_classes):
    """Returns one-hot encoding of a tensor."""

  def deferred_diag(self, vec):
    """Returns Diagonal Matrix under this `ComputeEngine`.

    Args:
      vec: Vector to place on diagonal of (deferred) Matrix.
    """
    return DiagMatrix(vec, self)


class Matrix(abc.ABC):
  """Holds an (implicit) matrix that can be multiplied with dense matrices."""
  _transpose: Optional['Matrix'] = None
  _engine: Optional['ComputeEngine'] = None

  def matmul(self, mat):
    """Computes `self @ mat`."""
    raise NotImplementedError()

  def rmatmul(self, mat):
    """Computes `mat @ self`."""
    raise NotImplementedError()

  @property
  def shape(self):
    raise NotImplementedError()

  def __matmul__(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    self._engine.assert_equal(self.shape[1], _shape(mat, self.engine)[0])
    return self.matmul(mat)

  def __rmatmul__(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    self._engine.assert_equal(_shape(mat, self.engine)[-1], self.shape[0])
    return self.rmatmul(mat)

  def __add__(self, mat):
    return Sum(self, mat)

  def transpose(self):
    if self._transpose is None:
      self._transpose = Transpose(self)
    return self._transpose

  @property
  def T(self):  # pylint: disable=invalid-name -- to mimic numpy.
    return self.transpose()

  def add_eye(self, diag_weight = float(1.0)):
    assert self._engine is not None, 'Compute engine is not set.'
    self._engine.assert_equal(self.shape[0], self.shape[1])
    return Sum(
        self,
        self.diag(diag_weight * self._engine.ones([self.shape[0]])))

  def rowsums(self, replace_if_0 = None):
    """Returns vector with shape `num_rows = [self.shape[0]]` that sums rows.

    Args:
      replace_if_0: If None, returns the actual sum, leaving zero-entries as-is.
        Otherwise, zero-entries will be replaced by this value.
    """
    assert self._engine is not None, 'Compute engine is not set.'
    y = self @ self._engine.ones([self.shape[1]])  # M . 1

    if replace_if_0 is not None:
      y = self._engine.where(self._engine.abs(y) < EPSILON,
                             replace_if_0 * self._engine.ones_like(y), y)
    return y

  def colsums(self, replace_if_0 = None):
    """Returns vector with shape `num_cols = [self.shape[1]]` that sums columns.

    Args:
      replace_if_0: If None, returns the actual sum, leaving zero-entries as-is.
        Otherwise, zero-entries will be replaced by this value.
    """
    assert self._engine is not None, 'Compute engine is not set.'
    # 1^T M  [shape=[cols]]
    y = self.__rmatmul__(self._engine.ones([self.shape[0]]))

    if replace_if_0 is not None:
      y = self._engine.where(self._engine.abs(y) < EPSILON,
                             replace_if_0 * self._engine.ones_like(y), y)
    return y

  def normalize_left(self):
    """Returns a left-stochastic matrix."""
    return Product(self, self.diag(1 / self.colsums(1.0)))

  def normalize_right(self):
    """Returns a right-stochastic matrix."""
    return Product(self.diag(1 / self.rowsums(1.0)), self)

  def normalize_leftright(self):
    assert self._engine is not None, 'Compute engine is not set.'
    return Product(
        self.diag(self._engine.rsqrt(self.rowsums(1.0))),
        self,
        self.diag(self._engine.rsqrt(self.colsums(1.0))),
    )

  def normalize_symmetric(self):
    assert self._engine is not None, 'Compute engine is not set.'
    inv_sqrt_degree = self.diag(self._engine.rsqrt(self.colsums(1.0)))
    return Product(inv_sqrt_degree, self, inv_sqrt_degree)

  def diag(self, vec):
    """Returns diagonal matrix with diagonal entries `vec`."""
    assert self._engine is not None, 'Compute engine is not set.'
    return self._engine.deferred_diag(vec)

  def set_engine(self, engine):
    self._engine = engine
    return self

  @property
  def engine(self):
    assert self._engine is not None, 'Compute engine is not set.'
    return self._engine


class SparseMatrix(Matrix):
  """Concrete Matrix implementation that can be initialized from indices."""

  def __init__(
      self, engine, *,
      indices,
      dense_shape,
      values = None):
    """Constructor.

    Args:
      engine: Compute engine that will be used for `gather` and
        `unsorted_segment_sum`.
      indices: pair of vector tensors: (row_ids, col_ids), which must be of
        equal length. Matrix[row_ids[i], col_ids[i]] == values[i].
      dense_shape: pair of ints: (num rows, num columns). It should be no less
        than `(max(row_ids)+1, max(col_ids)+1)`.
      values: If not set, it is assumed an all-ones vector (i.e., matrix would
        be binary). If set, it must be same length as `row_ids` (and `col_ids`).
    """
    self.set_engine(engine)
    num_rows, num_cols = dense_shape
    row_ids, col_ids = indices
    if num_rows is None:
      raise ValueError('num_rows (== dense_shape[0]) is None.')
    if num_cols is None:
      raise ValueError('num_cols (== dense_shape[1]) is None.')
    if row_ids is None:
      raise ValueError('row_ids (== indices[0]) is None.')
    if col_ids is None:
      raise ValueError('col_ids (== indices[1]) is None.')
    self.row_ids = row_ids
    self.col_ids = col_ids
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.values = values

  @property
  def shape(self):
    """Shape is (size of receiver node set, size of sender node set)."""
    return (self.num_rows, self.num_cols)

  def matmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    rows_of_mat = self._engine.gather(mat, self.col_ids)
    if self.values is not None:
      rows_of_mat *= self._broadcast_and_cast(self.values, rows_of_mat)
    return self._engine.unsorted_segment_sum(
        rows_of_mat, self.row_ids, self.num_rows)

  def rmatmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    # mat @ self == (mat @ self)^TT = (self^T @ mat^T)^T
    mat_t = self._engine.transpose(mat)
    rows_of_mat_t = self._engine.gather(mat_t, self.row_ids)
    if self.values is not None:
      rows_of_mat_t *= self._broadcast_and_cast(self.values, rows_of_mat_t)
    return self._engine.transpose(
        self._engine.unsorted_segment_sum(
            rows_of_mat_t, self.col_ids, self.num_cols))

  def _broadcast_and_cast(self, vector, tensor):
    """Broadcasts and casts `vector` to match dtype and shape of `tensor`."""
    need_extra_dims = len(tensor.shape) - len(vector.shape)
    if need_extra_dims > 0:
      broadcast_shape = self._engine.concat(
          [self._engine.shape(vector), [1]*need_extra_dims], axis=0)
      vector = self._engine.reshape(vector, broadcast_shape)
    vector = self._engine.cast(vector, tensor.dtype)
    return vector


class Transpose(Matrix):
  """Defines matrix transpose."""

  def __init__(self, mat):
    self._mat = mat
    self._engine = mat.engine

  def matmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    # (M'X) == (X'M)'
    return self._engine.transpose(
        self._mat.rmatmul(self._engine.transpose(mat)))

  def rmatmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    # (XM') == (XM')'' == (M X')'
    return self._engine.transpose(self._mat.matmul(self._engine.transpose(mat)))

  @property
  def shape(self):
    transpose_shape = self._mat.shape
    return (transpose_shape[1], transpose_shape[0])

  def transpose(self):
    return self._mat


class DiagMatrix(Matrix):
  """Defines diagonal matrix."""

  def __init__(self, diag_vector, engine):
    assert len(diag_vector.shape) == 1, 'Must be a vector.'
    self.set_engine(engine)
    self._diag_vector = diag_vector
    self._vec_shape = _shape(diag_vector, self.engine)[0]

  def matmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    return self._engine.einsum('i,i...->i...', self._diag_vector, mat)

  def rmatmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    return self._engine.einsum('i,...i->...i', self._diag_vector, mat)

  @property
  def shape(self):
    return (self._vec_shape, self._vec_shape)


class Product(Matrix):
  """Defines product of implicit matrices."""

  def __init__(self, *mats):
    assert mats
    assert mats[0].engine
    self.set_engine(mats[0].engine)
    for i in range(1, len(mats)):
      self._engine.assert_equal(
          mats[i - 1].shape[1], mats[i].shape[0])

    self._mats = mats

  def matmul(self, mat):
    for m in self._mats[::-1]:
      mat = m.__matmul__(mat)
    return mat

  def rmatmul(self, mat):
    for m in self._mats:
      mat = m.__rmatmul__(mat)
    return mat

  @property
  def shape(self):
    return (self._mats[0].shape[0], self._mats[-1].shape[1])


class Sum(Matrix):
  """Defines sum of implicit matrices."""

  def __init__(self, *mats):
    assert mats
    assert mats[0].engine
    self.set_engine(mats[0].engine)
    for i in range(1, len(mats)):
      self._engine.assert_equal(mats[i].shape[0], mats[0].shape[0])
      self._engine.assert_equal(mats[i].shape[1], mats[0].shape[1])
    self._mats = mats

  def matmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    return self._engine.add_n([m @ mat for m in self._mats])

  def rmatmul(self, mat):
    assert self._engine is not None, 'Compute engine is not set.'
    return self._engine.add_n([mat @ m for m in self._mats])

  @property
  def shape(self):
    return self._mats[0].shape


def _shape(tensor, engine
           ):
  """Helper function returns shape of eager or symbolic tensors."""
  if any([s is None for s in tensor.shape]):
    return engine.shape(tensor)
  else:
    return tensor.shape

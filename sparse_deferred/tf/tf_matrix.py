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

"""Wraps `tf.Tensor` and `tf.sparse.SparseTensor` with `matrix.Matrix`."""

import tensorflow as tf
from sparse_deferred.implicit import matrix
from sparse_deferred.tf import tf_engine


class DenseMatrix(matrix.Matrix):
  """Wraps tf matrix (rank 2 `tf.Tensor`) as deferred Matrix."""

  def __init__(self, dense_matrix):
    super().__init__()
    self.set_engine(tf_engine.engine)
    if len(dense_matrix.shape) != 2:
      raise ValueError('Expecting matrix but found shape = ' + str(
          dense_matrix.shape))
    self._dense_matrix = dense_matrix

  def matmul(self, mat):
    return tf.matmul(self._dense_matrix, mat)

  def rmatmul(self, mat):
    return tf.matmul(mat, self._dense_matrix)

  @property
  def shape(self):
    return self._dense_matrix.shape


class SparseMatrix(matrix.Matrix):
  """Wraps tf sparse matrix (rank 2 `SparseTensor`) as deferred Matrix."""

  def __init__(self, sparse_matrix):
    super().__init__()
    self.set_engine(tf_engine.engine)
    if len(sparse_matrix.shape) != 2:
      raise ValueError('Expecting matrix but found shape = ' + str(
          sparse_matrix.shape))
    self._sparse_matrix = sparse_matrix
    self._sparse_matrix_transpose = tf.sparse.transpose(sparse_matrix)

  def matmul(self, mat):
    mat = tf.cast(mat, self._sparse_matrix.dtype)
    out_shape = tf.concat([
        tf.ones(1, dtype=tf.int32) * -1,
        tf.shape(mat, out_type=tf.int32)[1:]
    ], axis=0)
    return tf.reshape(
        tf.sparse.sparse_dense_matmul(
            self._sparse_matrix, tf.reshape(mat, [tf.shape(mat)[0], -1])),
        out_shape)

  def rmatmul(self, mat):
    mat = tf.cast(mat, self._sparse_matrix.dtype)
    # (mat @ self) = (mat @ self)^TT = (self^T @ mat^T)^T
    out_shape = tf.concat([
        tf.shape(mat, out_type=tf.int32)[:-1],
        tf.ones(1, dtype=tf.int32) * -1,
    ], axis=0)
    return tf.reshape(
        tf.transpose(
            tf.sparse.sparse_dense_matmul(
                self._sparse_matrix_transpose,
                tf.reshape(mat, [-1, tf.shape(mat)[-1]]),
                adjoint_b=True)),
        out_shape)

  @property
  def shape(self):
    return self._sparse_matrix.shape


class GatherScatterSparseMatrix(matrix.SparseMatrix):
  """Equiv to `SparseMatrix` but uses gather/scatter instead of tf.sparse."""

  def __init__(self, sparse_matrix):
    indices = tf.transpose(sparse_matrix.indices)
    dense_shape = tf.cast(sparse_matrix.dense_shape, dtype=tf.int32)
    values = sparse_matrix.values
    super().__init__(
        engine=tf_engine.engine,
        indices=(indices[0], indices[1]),
        dense_shape=(dense_shape[0], dense_shape[1]),
        values=values)

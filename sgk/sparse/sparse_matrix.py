# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Defines primitive sparse matrix type for use with sparse ops."""
import numpy as np
import tensorflow.compat.v1 as tf

from sgk.sparse import connectors
from sgk.sparse import initializers

SPARSE_MATRIX_COLLECTION = "sparse_matrices"


def get_trainable_sparse_matrices():
  """Returns a list of all trainable sparse matrices."""
  return tf.get_collection(SPARSE_MATRIX_COLLECTION)


def track_trainable_sparse_matrix(sm):
  """Adds a sparse matrix to the collection."""
  assert sm.trainable
  if sm not in tf.get_collection_ref(SPARSE_MATRIX_COLLECTION):
    tf.add_to_collection(SPARSE_MATRIX_COLLECTION, sm)


def _dense_to_sparse(matrix):
  """Converts dense numpy matrix to a csr sparse matrix."""
  assert len(matrix.shape) == 2

  # Extract the nonzero values.
  values = matrix.compress((matrix != 0).flatten())

  # Calculate the offset of each row.
  mask = (matrix != 0).astype(np.int32)
  row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))),
                               axis=0)

  # Create the row indices and sort them.
  row_indices = np.argsort(-1 * np.diff(row_offsets))

  # Extract the column indices for the nonzero values.
  x = mask * (np.arange(matrix.shape[1]) + 1)
  column_indices = x.compress((x != 0).flatten())
  column_indices = column_indices - 1

  # Cast the desired precision.
  values = values.astype(np.float32)
  row_indices, row_offsets, column_indices = [
      x.astype(np.uint32) for x in
      [row_indices, row_offsets, column_indices]
  ]
  return values, row_indices, row_offsets, column_indices


class SparseTopology(object):
  """Describes a sparse matrix, with no values."""

  def __init__(self,
               name,
               shape=None,
               mask=None,
               connector=connectors.Uniform(0.8),
               dtype=tf.float32):
    if mask is None:
      assert shape is not None and len(shape) == 2
      mask = connector(np.ones(shape))
      self._shape = shape
    else:
      assert shape is None
      assert len(mask.shape) == 2
      self._shape = mask.shape
    self._name = name
    self._dtype = dtype
    self._sparsity = 1.0 - np.count_nonzero(mask) / mask.size

    # Create a numpy version of the sparse mask.
    _, row_indices_, row_offsets_, column_indices_ = _dense_to_sparse(mask)

    # Create tensors for the mask shape on the host. These are for internal
    # use and should generally not be used by end-user. Use the normal python
    # 'shape' property instead.
    with tf.device("cpu"):
      self._rows = tf.get_variable(
          initializer=self._shape[0],
          trainable=False,
          name=self._name + "_rows",
          dtype=tf.int32)
      self._columns = tf.get_variable(
          initializer=self._shape[1],
          trainable=False,
          name=self._name + "_columns",
          dtype=tf.int32)

    # Convert the sparse mask to TensorFlow variables.
    self._row_indices = tf.get_variable(
        initializer=row_indices_,
        trainable=False,
        name=self._name + "_row_indices",
        dtype=tf.uint32)
    self._row_offsets = tf.get_variable(
        initializer=row_offsets_,
        trainable=False,
        name=self._name + "_row_offsets",
        dtype=tf.uint32)
    self._column_indices = tf.get_variable(
        initializer=column_indices_,
        trainable=False,
        name=self._name + "_column_indices",
        dtype=tf.uint32)

  @property
  def name(self):
    return self._name

  @property
  def shape(self):
    return self._shape

  @property
  def size(self):
    return np.prod(self._shape)

  @property
  def dtype(self):
    return self._dtype

  @property
  def sparsity(self):
    return self._sparsity

  @property
  def row_indices(self):
    return self._row_indices

  @property
  def row_offsets(self):
    return self._row_offsets

  @property
  def column_indices(self):
    return self._column_indices


class SparseMatrix(object):
  """Compressed sparse row matrix type."""

  def __init__(self,
               name,
               shape=None,
               matrix=None,
               initializer=initializers.Uniform(),
               connector=connectors.Uniform(0.8),
               trainable=True,
               dtype=tf.float32):
    if matrix is None:
      assert shape is not None and len(shape) == 2
      matrix = connector(initializer(shape))
      self._shape = shape
    else:
      assert shape is None
      assert len(matrix.shape) == 2
      self._shape = matrix.shape
    self._name = name
    self._trainable = trainable
    self._dtype = dtype
    self._sparsity = 1.0 - np.count_nonzero(matrix) / matrix.size

    # Create a numpy version of the sparse matrix.
    values_, row_indices_, row_offsets_, column_indices_ = _dense_to_sparse(
        matrix)

    # Create tensors for the matrix shape on the host. These are for internal
    # use and should generally not be used by end-user. Use the normal python
    # 'shape' property instead.
    with tf.device("cpu"):
      self._rows = tf.get_variable(
          initializer=self._shape[0],
          trainable=False,
          name=self._name + "_rows",
          dtype=tf.int32)
      self._columns = tf.get_variable(
          initializer=self._shape[1],
          trainable=False,
          name=self._name + "_columns",
          dtype=tf.int32)

    # Convert the sparse matrix to TensorFlow variables.
    self._values = tf.get_variable(
        initializer=values_,
        trainable=self.trainable,
        name=self._name + "_values",
        dtype=self._dtype)
    self._row_indices = tf.get_variable(
        initializer=row_indices_,
        trainable=False,
        name=self._name + "_row_indices",
        dtype=tf.uint32)
    self._row_offsets = tf.get_variable(
        initializer=row_offsets_,
        trainable=False,
        name=self._name + "_row_offsets",
        dtype=tf.uint32)
    self._column_indices = tf.get_variable(
        initializer=column_indices_,
        trainable=False,
        name=self._name + "_column_indices",
        dtype=tf.uint32)

    # Add this matrix to the collection of trainable matrices.
    track_trainable_sparse_matrix(self)

  @classmethod
  def _wrap_existing(cls, shape, rows, columns, values, row_indices,
                     row_offsets, column_indices):
    """Helper to wrap existing tensors in a SparseMatrix object."""
    matrix = cls.__new__(cls)

    # Set the members appropriately.
    #
    # pylint: disable=protected-access
    matrix._shape = shape
    matrix._rows = rows
    matrix._columns = columns
    matrix._values = values
    matrix._row_indices = row_indices
    matrix._row_offsets = row_offsets
    matrix._column_indices = column_indices
    matrix._name = values.name
    matrix._trainable = False
    matrix._dtype = values.dtype
    # pylint: enable=protected-access
    return matrix

  @property
  def name(self):
    return self._name

  @property
  def shape(self):
    return self._shape

  @property
  def size(self):
    return np.prod(self._shape)

  @property
  def trainable(self):
    return self._trainable

  @property
  def dtype(self):
    return self._dtype

  @property
  def sparsity(self):
    return self._sparsity

  @property
  def values(self):
    return self._values

  @property
  def row_indices(self):
    return self._row_indices

  @property
  def row_offsets(self):
    return self._row_offsets

  @property
  def column_indices(self):
    return self._column_indices

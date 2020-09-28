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

"""Defines TensorFlow API for sparse matrix kernels."""
import tensorflow.compat.v1 as tf

from sgk.sparse.ops.backend import kernels
from sgk.sparse.sparse_matrix import SparseMatrix

# NOTE: We access a lot of private members of the SparseMatrix class to
# make the API cleaner for users. Disable pylint warnings for this file.
#
# pylint: disable=protected-access


def spmm(sparse_matrix, dense_matrix, transpose_lhs=False, transpose_rhs=False):
  """Sparse matrix matrix multiplication.

  Computes the product of a sparse matrix and a dense matrix.

  Args:
    sparse_matrix: SparseMatrix, the left-hand sparse operand to the matrix
      product.
    dense_matrix: Tensor, the right-hand, dense operand to the matrix product.
    transpose_lhs: bool, whether to transpose the lhs operand.
    transpose_rhs: bool, whether to transpose the rhs operand.

  Returns:
    Tensor, the dense matrix result of the product.
  """
  return kernels.spmm(sparse_matrix._rows, sparse_matrix._columns,
                      sparse_matrix.values, sparse_matrix.row_indices,
                      sparse_matrix.row_offsets, sparse_matrix.column_indices,
                      dense_matrix, transpose_lhs, transpose_rhs)


def replicated_spmm(values,
                    topology,
                    dense_matrix,
                    transpose_lhs=False,
                    transpose_rhs=False):
  """Convenience API for replicated spmm.

  TODO(tgale): Add a better matrix type instead of having this.

  Args:
    values: Tensor, the replicated sparse matrix values.
    topology: SparseTopology, the sparse matrix topology.
    dense_matrix: Tensor, the right-hand, dense operand to the matrix product.
    transpose_lhs: bool, whether to transpose the lhs operand.
    transpose_rhs: bool, whether to transpose the rhs operand.

  Returns:
    Tensor, the dense matrix result of the product.
  """
  return kernels.spmm(topology._rows, topology._columns, values,
                      topology.row_indices, topology.row_offsets,
                      topology.column_indices, dense_matrix, transpose_lhs,
                      transpose_rhs)


@tf.RegisterGradient("Spmm")
def _spmm_grad(op, grad):
  """Gradient operation for sparse matrix matrix multiplication."""
  # Collect the inputs.
  m = op.inputs[0]
  k = op.inputs[1]
  values = op.inputs[2]
  row_indices = op.inputs[3]
  row_offsets = op.inputs[4]
  column_indices = op.inputs[5]
  dense_matrix = op.inputs[6]

  # Sparse matrix gradient: multiply the gradient by the transposed
  # dense matrix.
  sparse_matrix_grad = kernels.sddmm(
      m,
      k,
      row_indices,
      row_offsets,
      column_indices,
      grad,
      dense_matrix,
      transpose_rhs=True)

  # Dense matrix gradient: transpose the sparse weights, calculate the
  # new row indices, and multiply sparse matrix with dense gradient.
  values_t, row_offsets_t, column_indices_t = kernels.csr_transpose(
      m, k, values, row_offsets, column_indices)
  row_indices_t = diffsort(row_offsets_t)
  dense_matrix_grad = kernels.spmm(k, m, values_t, row_indices_t, row_offsets_t,
                                   column_indices_t, grad)

  # NOTE: Because we exposed the sparse matrix meta-data as arguments to
  # the underlying op, we need to return 'None' as gradients for these
  # tensors.
  #
  # TODO(tgale): Make sure there are no performance implications for this.
  return [None, None, sparse_matrix_grad, None, None, None, dense_matrix_grad]


def fused_spmm(sparse_matrix,
               dense_matrix,
               bias,
               transpose_lhs=False,
               transpose_rhs=False):
  """Sparse matrix matrix multiplication with fused bias and relu.

  Computes the product of a sparse matrix and a dense matrix.

  Args:
    sparse_matrix: SparseMatrix, the left-hand sparse operand to the matrix
      product.
    dense_matrix: Tensor, the right-hand, dense operand to the matrix product.
    bias: Tesnor, the bias to add to the result.
    transpose_lhs: bool, whether to transpose the lhs operand.
    transpose_rhs: bool, whether to transpose the rhs operand.

  Returns:
    Tensor, the dense matrix result of the product.
  """
  return kernels.fused_spmm(sparse_matrix._rows, sparse_matrix._columns,
                            sparse_matrix.values, sparse_matrix.row_indices,
                            sparse_matrix.row_offsets,
                            sparse_matrix.column_indices, dense_matrix, bias,
                            transpose_lhs, transpose_rhs)


def sddmm(lhs_matrix,
          rhs_matrix,
          sparse_topology,
          transpose_lhs=False,
          transpose_rhs=False):
  """Sampled dense dense matrix multiplication.

  Computes selected outputs from the product of two dense matrices.

  Args:
    lhs_matrix: Tensor, the left-hand, dense matrix for the product.
    rhs_matrix: Tensor, the right-hand, dense matrix for the product.
    sparse_topology: SparseMatrix, specifying which outputs are to be computed.
    transpose_lhs: bool, whether to transpose the lhs operand.
    transpose_rhs: bool, whether to trasponse the rhs operand.

  Returns:
    A SparseMatrix holding the selected output values.
  """
  output_values = kernels.sddmm(sparse_topology._rows, sparse_topology._columns,
                                sparse_topology.row_indices,
                                sparse_topology.row_offsets,
                                sparse_topology.column_indices, lhs_matrix,
                                rhs_matrix, transpose_lhs, transpose_rhs)
  return SparseMatrix._wrap_existing(sparse_topology.shape,
                                     sparse_topology._columns,
                                     sparse_topology._rows, output_values,
                                     sparse_topology.row_indices,
                                     sparse_topology.row_offsets,
                                     sparse_topology.column_indices)


def replicated_sddmm(lhs_matrix,
                     rhs_matrix,
                     sparse_topology,
                     transpose_lhs=False,
                     transpose_rhs=False):
  """Convenience API for replicated sddmm."""
  return kernels.sddmm(sparse_topology._rows, sparse_topology._columns,
                       sparse_topology.row_indices, sparse_topology.row_offsets,
                       sparse_topology.column_indices, lhs_matrix, rhs_matrix,
                       transpose_lhs, transpose_rhs)


@tf.RegisterGradient("Sddmm")
def _sddmm_grad(op, grad):
  """Gradient operation for sampled dense dense matrix multiplication."""
  # Collect the inputs.
  m = op.inputs[0]
  n = op.inputs[1]
  row_indices = op.inputs[2]
  row_offsets = op.inputs[3]
  column_indices = op.inputs[4]
  lhs_matrix = op.inputs[5]
  rhs_matrix = op.inputs[6]

  # lhs matrix gradient: multiply the sparse gradient by the rhs matrix.
  lhs_matrix_grad = kernels.spmm(m, n, grad, row_indices, row_offsets,
                                 column_indices, rhs_matrix)

  # rhs matrix gradient: transpose the sparse gradient, calculate the new
  # row indices, and multiply the sparse gradient with the lhs matrix.
  grad_t, row_offsets_t, column_indices_t = kernels.csr_transpose(
      m, n, grad, row_offsets, column_indices)
  row_indices_t = diffsort(row_offsets_t)
  rhs_matrix_grad = kernels.spmm(n, m, grad_t, row_indices_t, row_offsets_t,
                                 column_indices_t, lhs_matrix)

  # NOTE: Because we exposed the sparse matrix meta-data as arguments to
  # the underlying op, we need to return 'None' as gradients for these
  # tensors.
  #
  # TODO(tgale): Make sure there are no performance implications for this.
  return [None] * 5 + [lhs_matrix_grad, rhs_matrix_grad]


def diffsort(offsets):
  """Calculate the argsort of the difference between the input offsets.

  Useful for sorting row indices in sparse matrices.

  Args:
    offsets: Tensor, array of offsets for the sparse of each row, where
      `offset[i+1] - offsets[i]` is the length of row i. Length `m+1`,
      where 'm' is the number of rows.

  Returns:
    Tensor, array of row indices sorted by row length, from largest to
      smallest.
  """
  diffs = (offsets - tf.roll(offsets, shift=-1, axis=0))[:-1]
  return tf.cast(tf.argsort(diffs, direction="DESCENDING"), tf.uint32)


def transpose(sparse_matrix):
  """Transpose a sparse matrix.

  Args:
    sparse_matrix: SparseMatrix, the sparse matrix to be transposed.

  Returns:
    SparseMatrix, a sparse matrix that is the transpose of the input
      sparse matrix.
  """
  values, row_offsets, column_indices = kernels.csr_transpose(
      sparse_matrix._rows, sparse_matrix._columns, sparse_matrix.values,
      sparse_matrix.row_offsets, sparse_matrix.column_indices)

  # Sort the row indices.
  row_indices = diffsort(row_offsets)

  # Wrap the individual tensors in a SparseMatrix and return.
  return SparseMatrix._wrap_existing(
      list(reversed(sparse_matrix.shape)), sparse_matrix._columns,
      sparse_matrix._rows, values, row_indices, row_offsets, column_indices)


def csr2idx(sparse_matrix):
  """Convert compressed sparse row meta-data to index format.

  Args:
    sparse_matrix: SparseMatrix, the sparse matrix to be converted.

  Returns:
   Tensor, the linear indices for the sparse matrix.
  """
  return kernels.csr2idx(sparse_matrix._rows, sparse_matrix._columns,
                         sparse_matrix.row_offsets,
                         sparse_matrix.column_indices)


def idx2csr(indices, rows, columns):
  """Convert index format meta-data to compressed sparse row format.

  Args:
    indices: Tensor, the linear indices for the sparse matrix.
    rows: Tensor, the number of rows in the sparse matrix.
    columns: Tensor, the number of columns in the sparse matrix.

  Returns:
    row_indices: Tensor, the row indices sorted by size.
    row_offset: Tensor, the offsets for each row in the sparse matrix.
    column_indices: Tensor, the column indices for each nonzero in the
      matrix.
  """
  # Calculate the length of each row by histogramming the indices.
  row_lengths = tf.histogram_fixed_width(
      indices, [0, rows * columns], nbins=rows)

  row_offsets = tf.concat([[0], tf.cumsum(row_lengths)], axis=0)
  row_indices = tf.argsort(row_lengths, direction="DESCENDING")
  column_indices = tf.mod(indices, columns)
  return row_indices, row_offsets, column_indices


def depthwise_conv2d(inputs, filters, strides, padding):
  return kernels.depthwise_conv(inputs, filters, strides, padding)


def fused_depthwise_conv2d(inputs, filters, bias, strides, padding):
  return kernels.fused_depthwise_conv(inputs, filters, bias, strides, padding)


bias_relu = kernels.bias_relu


def sparse_softmax(x):
  output_values = kernels.csr_softmax(x.values, x.row_indices, x.row_offsets,
                                      x.column_indices)
  return SparseMatrix._wrap_existing(x.shape, x._columns, x._rows,
                                     output_values, x.row_indices,
                                     x.row_offsets, x.column_indices)


def replicated_sparse_softmax(values, topology):
  return kernels.csr_softmax(values, topology.row_indices, topology.row_offsets,
                             topology.column_indices)


def fused_softmax(x):
  """Fused softmax over the last dimension."""
  input_shape = tf.shape(x)
  x = tf.reshape(x, [-1, input_shape[-1]])
  out = kernels.fused_softmax(x)
  return tf.reshape(out, input_shape)

# pylint: enable=protected-access

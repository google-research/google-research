# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""JAX Ops for symmetric matrices used by the Shampoo optimizer."""

import functools
from typing import Any, List, Optional, Sequence, Union

from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class SlicedSymmetricMatrix:
  """A symmetric matrix represented by lower-triangular block row slices.

  For example, the symmetric matrix M = [[a, b^T], [b, c]] would be represented
  by the block rows a and [b, c].

  The matrix may be batched, in which case each entry of block_rows may have
  dimension greater than 2. The last two dimensions represent the rows and cols.
  """
  block_rows: List[jnp.ndarray]


def product_with_transpose(
    mat1,
    mat2,
    axes,
    precision = lax.Precision.DEFAULT,
):
  """Returns mat1 * mat2^T for two matrices (possibly batched).

  The rows and columns are the last two dimensions for each matrix.

  Args:
    mat1: First matrix.
    mat2: Second matrix.
    axes: The axes over which to apply the product.
    precision: JAX precision to use for the multiplication.
  """
  return jnp.tensordot(a=mat1, b=mat2, axes=axes, precision=precision)


@functools.partial(jax.jit, static_argnames=("block_size", "axes", "precision"))
def sliced_transposed_product(
    mat,
    block_size,
    axes = (-1,),
    precision = lax.Precision.DEFAULT,
):
  """Returns the blocked slices representing a symmetric contraction.

  Specifically, the output is a contraction of the input mat with itself, in the
  specified axes.

  Args:
    mat: The matrix for which we will compute a contraction with itself.
    block_size: The size of row blocks to compute.
    axes: Axes to use for the contraction.
    precision: The precision to use in each computation.

  Raises:
    ValueError: Raised when the specified block size does not evenly divide
      the number of rows of the input mat.
  """
  rank = len(mat.shape)

  def _make_axis_positive(ax):
    assert -rank <= ax < rank
    return ax + rank if ax < 0 else ax

  positive_axes = [_make_axis_positive(ax) for ax in axes]
  assert len(positive_axes) == len(axes)
  remaining_axes = set(range(rank)) - set(positive_axes)
  assert len(remaining_axes) == 1
  remaining_ax = remaining_axes.pop()

  num_rows = mat.shape[remaining_ax]
  if num_rows % block_size != 0:
    raise ValueError(
        "The row dimension must be divisible by block_size. "
        f"Instead got row dimension={num_rows} and block_size={block_size}.")

  block_rows = []
  for i in range(num_rows // block_size):
    start_indices = [0]*rank
    start_indices[remaining_ax] = i * block_size

    slice_sizes = list(mat.shape)
    slice_sizes[remaining_ax] = block_size

    slice_sizes_full = list(mat.shape)
    slice_sizes_full[remaining_ax] = (i + 1) * block_size

    block_rows.append(
        product_with_transpose(
            lax.dynamic_slice(
                mat, start_indices=start_indices, slice_sizes=slice_sizes),
            lax.dynamic_slice(
                mat, start_indices=[0] * rank, slice_sizes=slice_sizes_full),
            axes=(axes, axes),
            precision=precision))

  return SlicedSymmetricMatrix(block_rows=block_rows)


@functools.partial(jax.jit, static_argnames=("block_size", "axes", "precision"))
def sliced_transposed_product_concat(
    mat,
    block_size,
    axes = (-1,),
    precision = lax.Precision.DEFAULT,
):
  """Returns the concatenated slices representing mat*mat^T.

  Args:
    mat: The matrix for which we will compute mat*mat^T. It does not need to be
      square, and may be batched.
    block_size: The size of row blocks to compute.
    axes: Axes to use for the contraction.
    precision: The precision to use in each computation.

  Raises:
    ValueError: Raised when the specified block size does not evenly divide
      the number of rows of the input mat.
  """
  sliced_symmetric_matrix = sliced_transposed_product(
      mat=mat, block_size=block_size, axes=axes, precision=precision)
  return jnp.concatenate(sliced_symmetric_matrix.block_rows, axis=-1)


@jax.jit
def materialize_matrix(symmetric_matrix):
  """Returns a materialized symmetric matrix.

  Args:
    symmetric_matrix: the matrix represented by lower-triangular block slices.
  """
  block_rows = symmetric_matrix.block_rows
  block_size = block_rows[0].shape[-2]
  num_blocks = len(block_rows)

  # Slice the lower-triangular and diagonal blocks into blocks.
  blocks = [[
      block_row[Ellipsis, i * block_size:(i + 1) * block_size] for i in range(k + 1)
  ] for k, block_row in enumerate(block_rows)]

  # Generate the (off-diagonal) upper-triangular blocks.
  off_diags = [[] for _ in range(num_blocks - 1)]
  for k, block_row in enumerate(block_rows[1:]):
    for i in range(k + 1):
      off_diags[i].append(
          jnp.swapaxes(
              a=block_row[Ellipsis, i * block_size:(i + 1) * block_size],
              axis1=-1,
              axis2=-2))

  return jnp.block([row + row_t for row, row_t in zip(blocks[:-1], off_diags)] +
                   [blocks[-1]])


@functools.partial(jax.jit, static_argnames=("num_blocks"))
def materialize_matrix_from_concat(
    block_rows_concat,
    num_blocks = None,
):
  """Returns a materialized symmetric matrix from concatenated slices.

  Args:
    block_rows_concat: The matrix represented as the concatenated
      lower-triangular blocks.
    num_blocks: The number of block-rows used to represent the symmetric matrix.
      If not specified, it is inferred from the shape of block_rows_concat.
  """
  if num_blocks is None:
    num_blocks = find_num_blocks(block_rows_concat)

  block_size = block_rows_concat.shape[-2]

  block_rows = [
      block_rows_concat[Ellipsis, (k * (k + 1)) // 2 *
                        block_size:(((k + 1) * (k + 2)) // 2 + 1) * block_size]
      for k in range(num_blocks)
  ]

  return materialize_matrix(SlicedSymmetricMatrix(block_rows=block_rows))


@functools.partial(jax.jit, static_argnames=("alpha", "beta", "axes"))
def update_sliced_rows(
    symmetric_matrix,
    mat,
    alpha,
    beta,
    axes = (-1,),
):
  """Implements the blocked equivalent of SYRK.

  Specifically, the symmetric matrix (represented using lower-triangular block
  rows) is updated using the sliced product of mat.

  Args:
    symmetric_matrix: The symmetric matrix to update.
    mat: The matrix to use for the update = mat * mat^T. The number of rows
      should match that of symmetric_matrix.
    alpha: The weight for the update.
    beta: The weight for the original symmetric matrix.
    axes: Axes to use for the contraction of the update.

  Returns:
    The updated rows of alpha * mat * mat^T + beta * symmetric_matrix.
  """
  block_size = symmetric_matrix.block_rows[0].shape[-2]
  sym_prod = sliced_transposed_product(
      mat=mat, block_size=block_size, axes=axes)
  return SlicedSymmetricMatrix(block_rows=[
      update * alpha + row * beta
      for update, row in zip(sym_prod.block_rows, symmetric_matrix.block_rows)
  ])


def num_blocks_from_total_blocks(total_blocks):
  """Returns the number of blocks (i.e.

  block rows) from the total blocks.

  This is the inverse of the function x -> x*(x+1)/2.

  For example, the matrix M = [[A, B^T], [B, C]] may be represented using a
  total of 3 blocks ([A, B, C]). The number of corresponding block rows is 2.

  Args:
    total_blocks: The total blocks used to represent the matrix.
  """
  num_blocks = np.round(
      (np.sqrt(8 * total_blocks + 1) - 1) / 2).astype(np.int32)
  if (num_blocks * (num_blocks + 1)) / 2 != total_blocks:
    raise ValueError(
        f"total_blocks={total_blocks} does not correspond to "
        "a symmetric matrix. It must have the form total_blocks = x*(x+1)/2.")
  return num_blocks


def find_num_blocks(block_rows_concat):
  """Returns the number of (row) blocks representing the concatenated matrix.

  For example, an input with dimensions [256, 2560] represents 10 square blocks,
  which matches 4 lower-triangular block rows (1+2+3+4). So this function will
  return 4.

  Use ordinary numpy functions here so that the returned value is static.

  Args:
    block_rows_concat: The concatenated block array.

  Raises:
    ValueError: When the dimensions of the matrix do not correspond to a lower
    triangular block representation.
  """
  # Compute the number of square blocks used to represent the matrix.
  total_blocks = block_rows_concat.shape[-1] / block_rows_concat.shape[-2]
  # Determine the number of block rows by inverting y = x*(x+1)/2.
  return num_blocks_from_total_blocks(total_blocks)


@functools.partial(jax.jit, static_argnames=("block_size"))
def slice_symmetric_matrix(
    mat,
    block_size,
):
  """Returns sliced row blocks.

  Args:
    mat: A symmetric matrix.
    block_size: The size of the row slices.
  """
  num_rows = mat.shape[-2]
  num_cols = mat.shape[-1]
  if num_rows != num_cols:
    raise ValueError("mat is not square.")
  if num_rows % block_size != 0:
    raise ValueError("block size does not evenly divide rows. "
                     f"num_rows={num_rows}, block_size={block_size}")
  return SlicedSymmetricMatrix(block_rows=[
      mat[Ellipsis, i * block_size:(i + 1) * block_size, 0:(i + 1) * block_size]
      for i in range(num_rows // block_size)
  ])


@functools.partial(jax.jit, static_argnames=("block_size"))
def slice_symmetric_matrix_concat(
    mat,
    block_size,
):
  """Returns the concatenated sliced row blocks.

  Args:
    mat: A symmetric matrix.
    block_size: The size of the row slices.
  """
  sliced_symmetric_matrix = slice_symmetric_matrix(
      mat=mat, block_size=block_size)
  return jnp.concatenate(sliced_symmetric_matrix.block_rows, axis=-1)


def sliced_matrix_diag(mat):
  """Returns the diagonal of the symmetric matrix.

  Args:
    mat: The symmetric matrix represented in concatenated block form.
  """
  rows, cols = mat.shape
  total_blocks = cols // rows
  num_blocks = num_blocks_from_total_blocks(total_blocks)
  diags = []
  for i in range(num_blocks):
    last_index = rows * ((i+2) * (i+1)) // 2
    first_index = last_index - rows
    diags.append(jnp.diag(mat[Ellipsis, first_index:last_index]))
  return jnp.concatenate(diags, axis=-1)


def diag_as_concat(diag, block_size):
  """Returns the representation of a diagonal matrix in symmetric block form.

  Args:
    diag: The 1D array for the diagonals.
    block_size: The size of blocks to use. Must divide the length of diag.
  """
  assert len(diag.shape) == 1  # diag must be 1D.
  assert len(diag) % block_size == 0
  num_diag_blocks = len(diag) // block_size
  blocks = []
  for i in range(num_diag_blocks):
    blocks.append(
        jnp.zeros(shape=(block_size, block_size * i), dtype=diag.dtype))
    blocks.append(jnp.diag(diag[i * block_size:(i + 1) * block_size]))
  return jnp.concatenate(blocks, axis=-1)


def row_abs_maxes(mat):
  """Returns the max of the absolute values of the rows of the full matrix.

  For example the symmetric matrix M = [[1, 6], [6, 2]] is represented using
  mat = [1, 6, 2] with block_size = 1. In this case the function returns the
  aboslute row maxes of the original symmetric matrix, [6, 6].

  Args:
    mat: The symmetric matrix represented as the concatenated blocks.
  """
  rows, cols = mat.shape

  # Find col and row max for each block.
  col_maxes = []
  row_maxes = []
  for i in range(cols // rows):
    block = jnp.abs(mat[Ellipsis, i * rows:(i + 1) * rows])
    col_maxes.append(jnp.max(block, axis=1))
    row_maxes.append(jnp.max(block, axis=0))

  # global row max from block maxes.
  num_blocks = num_blocks_from_total_blocks(cols // rows)
  maxes = []
  for i in range(num_blocks):
    maxes.append(
        jnp.concatenate(
            row_maxes[(i * (i + 1) // 2):((i + 2) * (i + 1) // 2)] + [
                col_maxes[((j + 1) * (j + 2)) // 2 - (j - i + 1)]
                for j in range(i + 1, num_blocks)
            ],
            axis=-1))

  return jnp.max(jnp.stack(maxes), axis=0)


def times_vector(mat, vec):
  """Returns the symmetric block-concatenated matrix multiplied by a vector.

  Specifically, each value in the vector is multiplied by a row of the full
  matrix. That is, the vector is broadcast and multiplied element-wise. Note
  this would be the transpose of full_mat * vec if full_mat represented the full
  symmetric matrix.

  Args:
    mat: The symmetric matrix represented as the concatenated blocks.
    vec: The vector, having the same dimension as the materialized matrix.
  """
  rows, cols = mat.shape
  num_blocks = num_blocks_from_total_blocks(cols // rows)
  multiplied = []
  for i in range(num_blocks):
    mat_block = mat[Ellipsis,
                    rows * ((i + 1) * i) // 2:rows * ((i + 1) * (i + 2)) // 2]
    vec_block = vec[Ellipsis, rows * i:rows * (i + 1)]
    multiplied.append(jnp.einsum("...ij,...i->ij", mat_block, vec_block))
  return jnp.concatenate(multiplied, axis=-1)

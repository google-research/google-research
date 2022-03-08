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
from typing import List, Union

from flax import struct
import jax
from jax import lax
import jax.numpy as jnp


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
    precision = lax.Precision.DEFAULT,
):
  """Returns mat1 * mat2^T for two matrices (possibly batched).

  The rows and columns are the last two dimensions for each matrix.

  Args:
    mat1: First matrix.
    mat2: Second matrix.
    precision: JAX precision to use for the multiplication.
  """
  return jnp.einsum("...ij,...kj->...ik", mat1, mat2, precision=precision)


@functools.partial(jax.jit, static_argnames=("block_size", "precision"))
def sliced_transposed_product(
    mat,
    block_size,
    precision = lax.Precision.DEFAULT,
):
  """Returns the blocked slices representing a symmetric matrix mat*mat^T.

  Args:
    mat: The matrix for which we will compute mat*mat^T. It does not need to be
      square, and may be batched.
    block_size: The size of row blocks to compute.
    precision: The precision to use in each computation.

  Raises:
    ValueError: Raised when the specified block size does not evenly divide
      the number of rows of the input mat.
  """
  num_rows = mat.shape[-2]
  if num_rows % block_size != 0:
    raise ValueError(
        "The row dimension must be divisible by block_size. "
        f"Instead got row dimension={num_rows} and block_size={block_size}.")
  block_rows = [
      product_with_transpose(mat[Ellipsis, i * block_size:(i + 1) * block_size, :],
                             mat[Ellipsis, 0:(i + 1) * block_size, :], precision)
      for i in range(num_rows // block_size)
  ]
  return SlicedSymmetricMatrix(block_rows=block_rows)


@functools.partial(jax.jit, static_argnames=("block_size", "precision"))
def sliced_transposed_product_concat(
    mat,
    block_size,
    precision = lax.Precision.DEFAULT,
):
  """Returns the concatenated slices representing mat*mat^T.

  Args:
    mat: The matrix for which we will compute mat*mat^T. It does not need to be
      square, and may be batched.
    block_size: The size of row blocks to compute.
    precision: The precision to use in each computation.

  Raises:
    ValueError: Raised when the specified block size does not evenly divide
      the number of rows of the input mat.
  """
  sliced_symmetric_matrix = sliced_transposed_product(
      mat=mat, block_size=block_size, precision=precision)
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
    num_blocks,
):
  """Returns a materialized symmetric matrix from concatenated slices.

  Args:
    block_rows_concat: The matrix represented as the concatenated
      lower-triangular blocks.
    num_blocks: The number of block-rows used to represent the symmetric matrix.
  """
  block_size = block_rows_concat.shape[-2]

  block_rows = [
      block_rows_concat[Ellipsis, (k * (k + 1)) // 2 *
                        block_size:(((k + 1) * (k + 2)) // 2 + 1) * block_size]
      for k in range(num_blocks)
  ]

  return materialize_matrix(SlicedSymmetricMatrix(block_rows=block_rows))


@functools.partial(jax.jit, static_argnames=("alpha", "beta"))
def update_sliced_rows(
    symmetric_matrix,
    mat,
    alpha,
    beta,
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

  Returns:
    The updated rows of alpha * mat * mat^T + beta * symmetric_matrix.
  """
  block_size = symmetric_matrix.block_rows[0].shape[-2]
  sym_prod = sliced_transposed_product(mat=mat, block_size=block_size)
  return SlicedSymmetricMatrix(block_rows=[
      update * alpha + row * beta
      for update, row in zip(sym_prod.block_rows, symmetric_matrix.block_rows)
  ])

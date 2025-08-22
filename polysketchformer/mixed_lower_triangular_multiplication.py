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

"""Implements mixed lower triangular multiplication."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def mixed_lt_multiply(
    a,
    b,
    c,
    a_prime,
    b_prime,
    grain_size,
    power,
    precision = jax.lax.Precision.DEFAULT,
):
  """Lower triangular multiplication with diagonal blocks being different.

  Given a, b and c, the usual lower triangular product is defined as the matrix
  lt(a @ b.T) @ c. The mixed lower triangular product involves two additional
  matrices a_prime, b_prime and a parameter grain_size. The output is defined as
  follows:

  Consider the matrix m_1 = lt(a @ b.T). Split the matrix m_1 into blocks of
  size grain_size x grain_size. When a and b have the same number of rows,
  we obtain that the diagonal blocks of m_1 are lower triangular, all the blocks
  above the diagonal blocks of m_1 are zeros. We now construct m_2 as follows:
  1. The blocks under main diagonal of m_2 are the same as the blocks under
      main diagonal of m_1.
  2. The diagonal blocks of m_2 are the same as the diagonal blocks of the
     matrix lt ((a_prime @ b_prime.T) ** power)

  Return the product of m_2 @ c

  ** Important **: The matrix m_2 and the output are the functions of grain_size

                  Inputs:
                    ┌────┐     ┌────┐      ┌────┐      ┌────┐      ┌────┐
                    │    │     │    │      │    │      │    │      │    │
    grain_size ◄────┤A_1 │     │B_1 │      │C_1 │      │A'_1│      │B'_1│
                    │    │     │    │      │    │      │    │      │    │
                    ├────┤     ├────┤      ├────┤      ├────┤      ├────┤
                    │    │     │    │      │    │      │    │      │    │
                    │A_2 │     │B_2 │      │C_2 │      │A'_2│      │B'_2│
                    │    │     │    │      │    │      │    │      │    │
                    ├────┤     ├────┤      ├────┤      ├────┤      ├────┤
                    │    │     │    │      │    │      │    │      │    │
                    │    │     │    │      │    │      │    │      │    │
                    │    │     │    │      │    │      │    │      │    │
                    ├────┤     ├────┤      ├────┤      ├────┤      ├────┤
                    │    │     │    │      │    │      │    │      │    │
                    │A_t │     │B_t │      │C_t │      │A'_t│      │B'_t│
                    │    │     │    │      │    │      │    │      │    │
                    └────┘     └────┘      └────┘      └────┘      └────┘
                  Result:
                     ┌──────────┬──────────┬──────────┬──────────┐     ┌───────┐
                     │          │          │          │          │     │       │
                     │LT(A'_1 @ │          │          │          │     │       │
                     │   B'_1.T)│    0     │    0     │    0     │     │  C_1  │
                     │  **power │          │          │          │     │       │
                     │          │          │          │          │     │       │
                     ├──────────┼──────────┼──────────┼──────────┤     ├───────┤
                     │          │          │          │          │     │       │
                     │          │LT(A'_2 @ │          │          │     │       │
                     │A_2@B_1.T │   B'_2.T)│    0     │    0     │     │  C_2  │
                     │          │  **power │          │          │     │       │
                     │          │          │          │          │     │       │
                     ├──────────┼──────────┼──────────┼──────────┤  @  ├───────┤
                     │          │          │          │          │     │       │
                     │          │          │          │          │     │       │
                     │ ........ │ .......  │ .......  │    0     │     │       │
                     │          │          │          │          │     │       │
                     │          │          │          │          │     │       │
                     ├──────────┼──────────┼──────────┼──────────┤     ├───────┤
                     │          │          │          │          │     │       │
                     │          │          │          │LT(A'_t @ │     │       │
                     │A_t@B_1.T │A_t@B_2.T │ .......  │   B'_t.T)│     │  C_t  │
                     │          │          │          │ **power  |     │       │
                     │          │          │          │          │     │       │
                     └──────────┴──────────┴──────────┴──────────┘     └───────┘

  Args:
    a: An array of shape [batch, ..., n, r]
    b: An array of shape [batch, ..., n, r]
    c: An array of shape [batch, ..., n, d]
    a_prime: An array of shape [batch, ..., n, r_prime]
    b_prime: An array of shape [batch, ..., n, r_prime]
    grain_size: A parameter denoting the ``local'' block size
    power: An integer denoting the power to which the entries of the diagonal
      blocks are to be raised.
    precision: Precision to be used in the einsums

  Returns:
    An array of shape [batch, ..., n, d] where over the batch dimensions, the
    operation described above is performed.
    cache: Unimplemented
  """
  assert a.shape == b.shape and a.shape[:-1] == c.shape[:-1]

  batch_dims = a.shape[:-2]
  n, r = a.shape[-2:]
  _, d = c.shape[-2:]
  r_prime = a_prime.shape[-1]

  assert n % grain_size == 0

  a_view = a.reshape(batch_dims + (-1, grain_size, r))
  b_view = b.reshape(batch_dims + (-1, grain_size, r))
  a_prime_view = a_prime.reshape(batch_dims + (-1, grain_size, r_prime))
  b_prime_view = b_prime.reshape(batch_dims + (-1, grain_size, r_prime))
  c_view = c.reshape(batch_dims + (-1, grain_size, d))

  a_prime_b_prime_products = jnp.einsum(
      '...ti, ...si ->...ts', a_prime_view, b_prime_view, precision=precision
  )

  bc_products = jnp.einsum(
      '...si, ...sk -> ...ik',
      b_view[:, :-1, :, :],
      c_view[:, :-1, :, :],
      precision=precision,
  )

  lt_a_prime_b_prime_products = jnp.tril(a_prime_b_prime_products) ** power
  result = lt_a_prime_b_prime_products @ c_view

  bc_products_cumsum = jnp.cumsum(bc_products, axis=-3)

  correction = a_view[Ellipsis, 1:, :, :] @ bc_products_cumsum

  pad_list = [(0, 0)] * (len(a.shape) + 1)
  pad_list[-3] = (1, 0)
  correction = jnp.pad(correction, pad_list)

  result = result + correction
  result = result.reshape(c.shape)
  return result, None


def mixed_tensor_lt_multiply(
    a,
    b,
    c,
    a_prime,
    b_prime,
    grain_size,
    power,
    precision = jax.lax.Precision.DEFAULT,
):
  """Lower triangular multiplication with diagonal blocks being different.

  Given a, b and c, the usual tesor lower triangular product is defined as the
  matrix lt(a @ b.T) @ c. The mixed tensor lower triangular product involves two
  additional matrices a_prime, b_prime and a parameter grain_size. The output is
  defined as follows:

  Consider the matrix m_1 = lt(a @ b.T) ** 2. Split the matrix m_1 into blocks
  of size grain_size x grain_size. When a and b have the same number of rows,
  we obtain that the diagonal blocks of m_1 are lower triangular, all the blocks
  above the diagonal blocks of m_1 are zeros. We now construct m_2 as follows:
  1. The blocks under main diagonal of m_2 are the same as the blocks under
      main diagonal of m_1.
  2. The diagonal blocks of m_2 are the same as the diagonal blocks of the
     matrix lt ((a_prime @ b_prime.T) ** power)

  Return the product of m_2 @ c

  ** Important **: The matrix m_2 and the output are the functions of grain_size

                  Inputs:
                    ┌────┐     ┌────┐      ┌────┐      ┌────┐      ┌────┐
                    │    │     │    │      │    │      │    │      │    │
    grain_size ◄────┤A_1 │     │B_1 │      │C_1 │      │A'_1│      │B'_1│
                    │    │     │    │      │    │      │    │      │    │
                    ├────┤     ├────┤      ├────┤      ├────┤      ├────┤
                    │    │     │    │      │    │      │    │      │    │
                    │A_2 │     │B_2 │      │C_2 │      │A'_2│      │B'_2│
                    │    │     │    │      │    │      │    │      │    │
                    ├────┤     ├────┤      ├────┤      ├────┤      ├────┤
                    │    │     │    │      │    │      │    │      │    │
                    │    │     │    │      │    │      │    │      │    │
                    │    │     │    │      │    │      │    │      │    │
                    ├────┤     ├────┤      ├────┤      ├────┤      ├────┤
                    │    │     │    │      │    │      │    │      │    │
                    │A_t │     │B_t │      │C_t │      │A'_t│      │B'_t│
                    │    │     │    │      │    │      │    │      │    │
                    └────┘     └────┘      └────┘      └────┘      └────┘
                  Result:
                     ┌──────────┬──────────┬──────────┬──────────┐     ┌───────┐
                     │          │          │          │          │     │       │
                     │LT(A'_1 @ │          │          │          │     │       │
                     │   B'_1.T)│    0     │    0     │    0     │     │  C_1  │
                     │  **power │          │          │          │     │       │
                     │          │          │          │          │     │       │
                     ├──────────┼──────────┼──────────┼──────────┤     ├───────┤
                     │          │          │          │          │     │       │
                     │          │LT(A'_2 @ │          │          │     │       │
                     │(A_2@     │   B'_2.T)│    0     │    0     │     │  C_2  │
                     │B_1.T)**2 │  **power │          │          │     │       │
                     │          │          │          │          │     │       │
                     ├──────────┼──────────┼──────────┼──────────┤  @  ├───────┤
                     │          │          │          │          │     │       │
                     │          │          │          │          │     │       │
                     │ ........ │ .......  │ .......  │    0     │     │       │
                     │          │          │          │          │     │       │
                     │          │          │          │          │     │       │
                     ├──────────┼──────────┼──────────┼──────────┤     ├───────┤
                     │          │          │          │          │     │       │
                     │          │          │          │LT(A'_t @ │     │       │
                     │(A_t@     │(A_t@     │ .......  │   B'_t.T)│     │  C_t  │
                     │ B_1.T)**2│B_2.T)**2 │          │ **power  │     │       │
                     │          │          │          │          │     │       │
                     └──────────┴──────────┴──────────┴──────────┘     └───────┘

  Args:
    a: An array of shape [batch, ..., n, r]
    b: An array of shape [batch, ..., n, r]
    c: An array of shape [batch, ..., n, d]
    a_prime: An array of shape [batch, ..., n, r_prime]
    b_prime: An array of shape [batch, ..., n, r_prime]
    grain_size: A parameter denoting the ``local'' block size
    power: An integer denoting the power to which the entries of the diagonal
      blocks are to be raised.
    precision: Precision to be used in einsums in the implementation

  Returns:
    An array of shape [batch, ..., n, d] where over the batch dimensions, the
    operation described above is performed.
    cache: Unimplemented
  """

  assert a.shape == b.shape and a.shape[:-1] == c.shape[:-1]

  batch_dims = a.shape[:-2]
  n, r = a.shape[-2:]
  _, d = c.shape[-2:]
  r_prime = a_prime.shape[-1]

  assert n % grain_size == 0

  a_view = a.reshape(batch_dims + (-1, grain_size, r))  # [a1, ..., at]
  b_view = b.reshape(batch_dims + (-1, grain_size, r))  # [b1, ..., bt]
  a_prime_view = a_prime.reshape(
      batch_dims + (-1, grain_size, r_prime)
  )  # [a1, ..., at]
  b_prime_view = b_prime.reshape(
      batch_dims + (-1, grain_size, r_prime)
  )  # [b1, ..., bt]
  c_view = c.reshape(batch_dims + (-1, grain_size, d))  # [c1, ..., ct]

  # self tensoring rows of a <=> self tensoring rows of blocks in a_view
  # (t,ij)-th entry of (a1 tensor a1) is a1[t, i] * a1[t, j]
  # (t, s)-th entry of (a1 tensor a1) @ (b1 tensor b1).T is
  # \sum_{ij} (a1 tensor a1)_{t, ij} * (b1 tensor b1)_{s, ij}
  # \sum_{i,j} (a1)_{t, i} * (a1)_{t, j} * (b1)_{s, i} * (b1)_{s, j}

  lt_a_prime_b_prime_products = (
      jnp.tril(
          jnp.einsum(
              '...ti, ...si -> ...ts',
              a_prime_view,
              b_prime_view,
              precision=precision,
          )
      )
      ** power
  )

  # analog of b_transpose_c_products in the above function
  b_tensor_transpose_c_products = jnp.einsum(
      '...ti, ...tj, ...td -> ...ijd',
      b_view[Ellipsis, :-1, :, :],
      b_view[Ellipsis, :-1, :, :],
      c_view[Ellipsis, :-1, :, :],
      precision=precision,
  )

  result = jnp.matmul(lt_a_prime_b_prime_products, c_view)

  b_tensor_transpose_c_products_cum_sum = jnp.cumsum(
      b_tensor_transpose_c_products, axis=-4
  )

  correction = jnp.einsum(
      '...ti, ...tj, ...ijd -> ...td',
      a_view[Ellipsis, 1:, :, :],
      a_view[Ellipsis, 1:, :, :],
      b_tensor_transpose_c_products_cum_sum,
      precision=precision,
  )

  pad_list = [(0, 0)] * (len(a.shape) + 1)
  pad_list[-3] = (1, 0)
  correction = jnp.pad(correction, pad_list)
  result = result + correction
  result = result.reshape(c.shape)

  return result, None

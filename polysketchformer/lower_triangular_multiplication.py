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

"""Implementation of lower triangular multiplication algorithm.

This file provides a function lt_multiply to compute lt(a @ b.T) @ c given three
matrices a, b and c of appropriate dimensions.

This file also provides another function lt_tensor_multiply that computes
lt( (a tensor a) @ (b tensor b).T) @ c given three matrices a, b and c of
appropriate dimensions.

The functions implement a block-based algorithm described in
https://arxiv.org/abs/2310.01655
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp


def lt_multiply(
    a,
    b,
    c,
    grain_size,
    precision = jax.lax.Precision.DEFAULT,
    build_cache = False,
):
  """Computes the 'lower triangular product'.

  Given a, b and c, the lower triangular product is defined to be the matrix
  lt(a @ b.T) @ c. When a batch dimension is present, the operation is defined
  with respect last to dimensions.

  Args:
    a: An array of shape [batch, ..., n, r]
    b: An array of shape [batch, ..., n, r]
    c: An array of shape [batch, ..., n, d]
    grain_size: an integer parameter that divides n
    precision: a precision parameter that defines the precision at which the
      intermediate multiplications are to be performed.
    build_cache: If set to True, then builds a cache and returns it useful for
      inference.

  Returns:
    A tuple. If build_cache is True, the first item in the tuple is an array
    equal to lt(a @ b.T) @ c of shape [batch, ..., n, d] is returned and the
    second item is an array called "cache" of shape [batch, ..., r, d] equal to
    b.T @ c. If build_cache is False, the first item of the tuple is the same
    as before but the second item is set to None.
  """
  assert a.shape == b.shape and a.shape[:-1] == c.shape[:-1]

  batch_dims = a.shape[:-2]
  n, r = a.shape[-2:]
  _, d = c.shape[-2:]

  assert n % grain_size == 0  # Grain size must divide the number of rows

  if n == grain_size:
    result = (
        jnp.tril(
            jnp.einsum(
                '...ti, ...si -> ...ts',
                a,
                b,
                precision=precision,
            )
        )
        @ c
    )
    if build_cache:
      cache = jnp.einsum(
          '...ti, ...tj -> ...ij',
          b,
          c,
          precision=precision,
      )
      return result, cache

  # We list the meaning of each array in a comment on the side/above.
  # The operations are described for a single example in the batch.

  a_view = a.reshape(
      batch_dims + (-1, grain_size, r)
  )  # [a_1, ...] where a_i is a grain_size x r matrix
  b_view = b.reshape(
      batch_dims + (-1, grain_size, r)
  )  # [b_1, ...] where b_i is a grain_size x r matrix
  c_view = c.reshape(
      batch_dims + (-1, grain_size, d)
  )  # [c_1, ...] where c_i is a grain_size x d matrix

  # Computes [a_1 @ b_1.T, ..., ]
  ab_products = jnp.einsum(
      '...ti, ...si -> ...ts',
      a_view,
      b_view,
      precision=precision,
  )

  # Computes [b_1.T @ c_1, ..., ] excluding last matrix
  bc_products = jnp.einsum(
      '...si, ...sk -> ...ik',
      b_view[Ellipsis, :-1, :, :],  # Excludes last matrix
      c_view[Ellipsis, :-1, :, :],  # Excludes last matrix
      precision=precision,
  )

  lt_ab_products = jnp.tril(ab_products)  # [lt(a_1 @ b_1.T), ...]

  # Computes [lt(a_1 @ b_1.T) @ c_1, ...]
  result = jnp.matmul(lt_ab_products, c_view, precision=precision)

  # Computes [b_1.T @ c_1, b_1.T @ c_1 + b_2.T @ c_2, ...]
  bc_products_cum_sum = jnp.cumsum(bc_products, axis=-3)

  # Computes [a_2 @ (b_1.T @ c_1), a_3 @ (b_1.T @ c_1 + b_2.T @ c_2), ...]
  correction = jnp.matmul(
      a_view[Ellipsis, 1:, :, :], bc_products_cum_sum, precision=precision
  )

  pad_list = [(0, 0)] * (len(a.shape) + 1)
  pad_list[-3] = (1, 0)
  correction = jnp.pad(correction, pad_list)  # Appends a 0 matrix.

  # [lt(a_1 @ b_1.T) @ c_1, lt(a_2 @ b_2.T) @ c_2 + a_2 @ (b_1.T @ c_1), ...]
  result = result + correction

  result = result.reshape(c.shape)

  cache = None
  if build_cache:
    cache = bc_products_cum_sum[Ellipsis, -1, :, :] + jnp.einsum(
        '...si, ...sd -> ...id',
        b_view[Ellipsis, -1, :, :],
        c_view[Ellipsis, -1, :, :],
        precision=precision,
    )
  return result, cache


def tensor_lt_multiply(
    a,
    b,
    c,
    grain_size,
    precision = jax.lax.Precision.DEFAULT,
    build_cache = False,
):
  """Computes the lower triangular product after tensoring a and b.

  Given a matrix a, the matrix (a tensor a) is defined by tensoring each
  row of a with itself which "squares" the number of columns in a.

  This function takes matrices a, b and c of appropriate input sizes and
  computes lt ( (a tensor a) @ (b tensor b).T ) @ c using a block-based
  algorithm with the given grain_size parameter. When a batch dimension is
  present, the operation is defined with respect last to dimensions.

  Instead of tensoring a and b and passing it to lt_multiply, we directly
  implement this algorithm using einsums. This is more efficient in practice
  when using TPUs/GPUs.

  Args:
    a: An array of shape [batch, ..., n, r]
    b: Input array of size [batch, ..., n, r]
    c: Input array of size [batch, ..., n, d]
    grain_size: number of rows in a block
    precision: precision of the einsum and matmul operations
    build_cache: If set to True, then builds a cache and returns it useful for
      inference.

  Returns:
    A tuple. If build_cache is True, then the first item is an array of shape
    [batch, ..., n, d] equal to lt ((a tensor a) @ (b tensor b).T) @ c. The
    second item is an array called "cache" which holds the representation of
    (b tensor b).T @ c and has the shape [batch, ..., r, r, d]. If build_cache
    is False, the first item in the tuple is the same as before but the second
    item in the tuple is set to None.
  """

  assert a.shape == b.shape and a.shape[:-1] == c.shape[:-1]

  batch_dims = a.shape[:-2]
  n, r = a.shape[-2:]
  _, d = c.shape[-2:]

  assert n % grain_size == 0

  if n == grain_size:
    result = (
        jnp.tril(
            jnp.einsum('...ti, ...si->...ts', a, b, precision=precision) ** 2
        )
        @ c
    )
    cache = None
    if build_cache:
      cache = jnp.einsum(
          '...ti, ...tj, ...td->...ijd', b, b, c, precision=precision
      )
    return result, cache

  a_view = a.reshape(batch_dims + (-1, grain_size, r))  # [a1, ..., at]
  b_view = b.reshape(batch_dims + (-1, grain_size, r))  # [b1, ..., bt]
  c_view = c.reshape(batch_dims + (-1, grain_size, d))  # [c1, ..., ct]

  # Analog of ab_products in the above
  a_tensor_b_tensor_products = jnp.einsum(
      '...ti, ...si -> ...ts',
      a_view,
      b_view,
      precision=precision,
  ) ** 2

  b_tensor_transpose_c_products = jnp.einsum(
      '...ti, ...tj, ...td -> ...ijd',
      b_view[Ellipsis, :-1, :, :],
      b_view[Ellipsis, :-1, :, :],
      c_view[Ellipsis, :-1, :, :],
      precision=precision,
  )

  lt_a_tensor_b_tensor_products = jnp.tril(a_tensor_b_tensor_products)

  result = jnp.matmul(
      lt_a_tensor_b_tensor_products, c_view, precision=precision
  )

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

  cache = None
  if build_cache:
    cache = b_tensor_transpose_c_products_cum_sum[
        Ellipsis, -1, :, :, :  # Take the last matrix
    ] + jnp.einsum(
        '...ti, ...tj, ...td -> ...ijd',
        b_view[Ellipsis, -1, :, :],
        b_view[Ellipsis, -1, :, :],
        c_view[Ellipsis, -1, :, :],
    )  # Add the remaining contribution
  return result, cache

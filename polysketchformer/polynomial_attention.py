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

"""Implements polynomial attention."""

import jax
import jax.numpy as jnp


def polynomial_attention(
    query,
    key,
    value,
    degree = 4,
    is_causal = False,
    normalization_bias = 1.0,
    precision = jax.lax.Precision.DEFAULT,
):
  """Implementation of polynomial attention.

  Args:
    query: An array of shape [batch, ..., q_length, qk_dim]
    key: An array of shape [batch, ..., kv_length, qk_dim]
    value: An array of shape [batch, ..., kv_length, v_dim]
    degree: (even) degree of the polynomial to be used.
    is_causal: True if we need to apply causal masking and false other wise. If
               is_causal = True, we require q_length == kv_length.
    normalization_bias: The constant we add in the denominator to make the
                        computation stable.
    precision: The precision to be used when using matmuls and einsums.

  Returns:
    Outputs the result of applying polynomial attention on the matrices
    query, key and value.
  """
  assert degree % 2 == 0  # 'degree' must be even
  assert query.shape[:-2] == key.shape[:-2] == value.shape[:-2]  # Batches match
  assert query.shape[-1] == key.shape[-1]  # Check if it supports dot products
  assert key.shape[-2] == value.shape[-2]  # Check if num_keys match num_values
  if is_causal:
    # Checks if num_queries = num_keys to support causal masking
    assert query.shape[-2] == key.shape[-2]

  dot_products = jnp.einsum(
      '...ti, ...si -> ...ts', query, key, precision=precision
  )

  if is_causal:
    dot_products = jnp.tril(dot_products)

  attention_weights_unnormalized = dot_products**degree
  scalings = jnp.einsum('...ts -> ...t', attention_weights_unnormalized)
  scalings = normalization_bias + scalings
  scalings = jnp.reciprocal(scalings)
  attention_weights_matrix = jnp.einsum(
      '...ts, ...t -> ...ts', attention_weights_unnormalized, scalings
  )
  result = jnp.matmul(attention_weights_matrix, value, precision=precision)
  return result

# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Implements linear time squared attention."""

from typing import Callable

import jax
import jax.numpy as jnp

from polysketchformer import lower_triangular_multiplication


def make_squared_attn_fn(
    is_causal,
    grain_size,
):
  """Returns a function that applies the squared attention in linear time.

  Given Q, K, V matrices, in the causal setting we define the output of the
  squared attention as
            o_i = sum_{j=1}^i <q_i, k_j>^2 * v_j / normalization_i
  where normalization_i is defined as 1.0 + sum_{j = 1}^i <q_i, k_j>^2. 1.0 is
  added to increase the numerical stability of reciprocal computation.

  In the non-causal setting, all the sum terms go sum_{j=1}^n instead of
  stopping at i.

  Args:
    is_causal: True or False to denote causal / non-causal setting
    grain_size: Block size to be used in LT multiplication algorithm
  """
  if is_causal:

    def attention_function(
        query, key, value
    ):
      """The function applies the squared attention w/ causal mask.

      Args:
        query: An array of shape [batch, ..., q_length, qk_dim]
        key: An array of shape [batch, ..., kv_length, qk_dim]
        value: An array of shape [batch, ..., kv_length, v_dim]

      Returns:
        An array of shape [batch, ..., q_length, v_dim]
      """
      pad_list = [(0, 0)] * len(query.shape)
      pad_list[-1] = (0, 1)
      value_padded = jnp.pad(
          value, pad_list, mode='constant', constant_values=1.0
      )  # Concatenates a column of 1.0s to all the value arrays.

      result, _ = lower_triangular_multiplication.tensor_lt_multiply(
          query, key, value_padded, grain_size
      )
      inverse_scalings = result[Ellipsis, -1]
      inverse_scalings = 1.0 + inverse_scalings
      # The normalization factor is added to make the avoid numerical issues
      # during division and make the training stable.

      scalings = jnp.reciprocal(inverse_scalings)
      result = result[Ellipsis, :-1] * scalings[Ellipsis, None]
      return result

  else:

    def attention_function(
        query, key, value
    ):
      """The function applies the squared attention w/o causal mask.

      Args:
        query: An array of shape [batch, ..., q_length, qk_dim]
        key: An array of shape [batch, ..., kv_length, qk_dim]
        value: An array of shape [batch, ..., kv_length, v_dim]

      Returns:
        An array of shape [batch, ..., q_length, v_dim]
      """

      # Append column of 1s to the value matrices before running multiply.
      pad_list = [(0, 0)] * len(query.shape)
      pad_list[-1] = (0, 1)
      value_padded = jnp.pad(
          value, pad_list, mode='constant', constant_values=1.0
      )

      first_product = jnp.einsum(
          '...ki, ...kj, ...kv-> ...ijv',
          key,
          key,
          value_padded,
      )  # Computes (key tensor key).T @ value_padded.
      result = jnp.einsum(
          '...qi, ...qj, ...ijv->...qv',
          query,
          query,
          first_product,
      )  # Computes (query tensor query) @ first_product.

      inverse_scalings = result[Ellipsis, -1]
      inverse_scalings = 1.0 + inverse_scalings
      # The normalization factor is added to make the avoid numerical issues
      # during division and make the training stable.

      scalings = jnp.reciprocal(inverse_scalings)
      result = result[Ellipsis, :-1] * scalings[Ellipsis, None]
      return result

  return attention_function

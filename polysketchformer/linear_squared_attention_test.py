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

"""Tests for linear_squared_attention."""

from absl.testing import absltest
import jax.numpy as jnp

from polysketchformer import linear_squared_attention


class LinearSquaredAttentionTest(absltest.TestCase):

  def test_causal(self):
    context_length = 32
    dimension = 8
    query = (
        jnp.arange(context_length * dimension)
        .astype(jnp.float32)
        .reshape((1, context_length, dimension))
    )
    key = (
        jnp.arange(context_length * dimension, 2 * context_length * dimension)
        .astype(jnp.float32)
        .reshape((1, context_length, dimension))
    )
    value = query

    causal_squared_attention = linear_squared_attention.make_squared_attn_fn(
        is_causal=True, grain_size=16
    )
    impl_output = causal_squared_attention(query, key, value)

    # Direct calculation
    squared_attn_matrix = jnp.tril(query @ key.transpose(0, 2, 1)) ** 2
    inverse_scalings = jnp.einsum("...qk -> ...q", squared_attn_matrix)
    inverse_scalings = 1.0 + inverse_scalings
    scalings = jnp.reciprocal(inverse_scalings)
    attention_weights = squared_attn_matrix * scalings[Ellipsis, None]
    direct_result = attention_weights @ value

    # Check allclose
    self.assertTrue(jnp.allclose(direct_result, impl_output))

  def test_non_causal(self):
    context_length = 32
    dimension = 8
    query = (
        jnp.arange(context_length * dimension)
        .astype(jnp.float32)
        .reshape((1, context_length, dimension))
    )
    key = (
        jnp.arange(context_length * dimension, 2 * context_length * dimension)
        .astype(jnp.float32)
        .reshape((1, context_length, dimension))
    )
    value = query

    non_causal_squared_attention = (
        linear_squared_attention.make_squared_attn_fn(
            is_causal=False, grain_size=16
        )
    )
    impl_output = non_causal_squared_attention(query, key, value)

    # Direct calculation
    squared_attn_matrix = (query @ key.transpose(0, 2, 1)) ** 2
    # Lack of jnp.tril is the only difference from causal

    inverse_scalings = jnp.einsum("...qk -> ...q", squared_attn_matrix)
    inverse_scalings = 1.0 + inverse_scalings
    scalings = jnp.reciprocal(inverse_scalings)
    attention_weights = squared_attn_matrix * scalings[Ellipsis, None]
    direct_result = attention_weights @ value

    # Check allclose
    self.assertTrue(jnp.allclose(direct_result, impl_output))


if __name__ == "__main__":
  absltest.main()

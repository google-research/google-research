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

"""Tests for linear_squared_attention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax.scipy import linalg

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


class MixedLinearSquaredAttentionTest(parameterized.TestCase):

  @parameterized.product(grain_size=[1, 2, 4, 16], power=[2, 4])
  def test_causal(self, grain_size, power):
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
    query_prime = (
        jnp.arange(
            2 * context_length * dimension, 3 * context_length * dimension
        )
        .astype(jnp.float32)
        .reshape((1, context_length, dimension))
    )
    key_prime = (
        jnp.arange(
            3 * context_length * dimension, 4 * context_length * dimension
        )
        .astype(jnp.float32)
        .reshape((1, context_length, dimension))
    )
    value = query

    causal_mixed_squared_attention = (
        linear_squared_attention.make_mixed_squared_attn_fn(
            is_causal=True, grain_size=grain_size, power=power
        )
    )
    impl_output = causal_mixed_squared_attention(
        query, key, value, query_prime, key_prime
    )

    # Computing the reference output
    original_weights_matrix = jnp.tril((query @ key.transpose(0, 2, 1)) ** 2)
    within_blocks_weight_matrix = jnp.tril(
        (query_prime @ key_prime.transpose(0, 2, 1)) ** power
    )
    blocks_grain_size = jnp.ones((grain_size, grain_size), dtype=jnp.int32)
    block_diagonal_matrix = linalg.block_diag(
        *([blocks_grain_size] * (context_length // grain_size))
    )
    block_diagonal_mask = block_diagonal_matrix.astype(jnp.bool)
    mixed_attention_weights = jnp.where(
        block_diagonal_mask,
        within_blocks_weight_matrix,
        original_weights_matrix,
    )

    scalings = jnp.einsum("...qk -> ...q", mixed_attention_weights) + 1.0
    scalings = jnp.reciprocal(scalings)

    reference_output = (mixed_attention_weights @ value) * scalings[Ellipsis, None]
    self.assertTrue(jnp.allclose(reference_output, impl_output))


if __name__ == "__main__":
  absltest.main()

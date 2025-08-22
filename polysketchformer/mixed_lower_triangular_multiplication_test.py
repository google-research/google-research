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

"""Tests the implementation of Mixed Lower Triangular Multiplication Algorithm."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax.scipy import linalg

from polysketchformer import mixed_lower_triangular_multiplication


class MixedLowerTriangularMultiplicationTest(parameterized.TestCase):

  @parameterized.product(
      grain_size=[1, 2, 4], power=[1, 2, 4]
  )
  def test_mixed_lt_multiply(self, grain_size, power):
    """Tests the Mixed LT Multiplication Algorithm."""
    n = 16
    r = 8
    r_prime = 6
    batches = 4

    a = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    b = (
        jnp.arange(batches * n * r, 2 * batches * n * r)
        .astype(jnp.float32)
        .reshape((batches, n, r))
    )
    c = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    a_prime = (
        jnp.arange(batches * n * r_prime)
        .astype(jnp.float32)
        .reshape((batches, n, r_prime))
    )
    b_prime = (
        jnp.arange(batches * n * r_prime, 2 * batches * n * r_prime)
        .astype(jnp.float32)
        .reshape((batches, n, r_prime))
    )

    ab_transpose = jnp.einsum('...ti, ...si->...ts', a, b)
    a_prime_b_prime_transpose = jnp.einsum(
        '...ti, ...si->...ts', a_prime, b_prime
    ) ** power
    blocks_grain_size = jnp.ones((grain_size, grain_size), dtype=jnp.int32)
    block_diagonal_matrix = linalg.block_diag(
        *([blocks_grain_size] * (n // grain_size))
    )
    block_diagonal_mask = block_diagonal_matrix.astype(jnp.bool)
    mixed_matrix = jnp.where(
        block_diagonal_mask, a_prime_b_prime_transpose, ab_transpose
    )
    mixed_matrix_lt = jnp.tril(mixed_matrix)
    reference_result = mixed_matrix_lt @ c
    impl_result, _ = mixed_lower_triangular_multiplication.mixed_lt_multiply(
        a, b, c, a_prime, b_prime, grain_size, power=power
    )
    self.assertTrue(jnp.allclose(reference_result, impl_result))

  @parameterized.product(
      grain_size=[1, 2, 4], power=[1, 2, 4]
  )
  def test_mixed_tensor_lt_multiply(self, grain_size, power):
    """Tests the Mixed Tensor LT Multiplication Algorithm."""
    n = 16
    r = 8
    r_prime = 6
    batches = 4

    a = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    b = (
        jnp.arange(batches * n * r, 2 * batches * n * r)
        .astype(jnp.float32)
        .reshape((batches, n, r))
    )
    c = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    a_prime = (
        jnp.arange(batches * n * r_prime)
        .astype(jnp.float32)
        .reshape((batches, n, r_prime))
    )
    b_prime = (
        jnp.arange(batches * n * r_prime, 2 * batches * n * r_prime)
        .astype(jnp.float32)
        .reshape((batches, n, r_prime))
    )

    ab_transpose = jnp.einsum('...ti, ...si->...ts', a, b)
    ab_transpose_squared = ab_transpose**2
    a_prime_b_prime_transpose = jnp.einsum(
        '...ti, ...si->...ts', a_prime, b_prime
    ) ** power
    blocks_grain_size = jnp.ones((grain_size, grain_size), dtype=jnp.int32)
    block_diagonal_matrix = linalg.block_diag(
        *([blocks_grain_size] * (n // grain_size))
    )
    block_diagonal_mask = block_diagonal_matrix.astype(jnp.bool)
    mixed_matrix = jnp.where(
        block_diagonal_mask, a_prime_b_prime_transpose, ab_transpose_squared
    )
    mixed_matrix_lt = jnp.tril(mixed_matrix)
    reference_result = mixed_matrix_lt @ c
    impl_result, _ = (
        mixed_lower_triangular_multiplication.mixed_tensor_lt_multiply(
            a, b, c, a_prime, b_prime, grain_size, power=power
        )
    )
    self.assertTrue(jnp.allclose(reference_result, impl_result))


if __name__ == '__main__':
  absltest.main()

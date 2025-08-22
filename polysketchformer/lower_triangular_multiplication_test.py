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

"""Test Lower Triangular Multiplication Algorithm."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from polysketchformer import lower_triangular_multiplication


class LowerTriangularMultiplicationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'grain_size_1', 'grain_size': 1},
      {'testcase_name': 'grain_size_2', 'grain_size': 2},
      {'testcase_name': 'grain_size_4', 'grain_size': 4},
  )
  def test_lt_multiply_build_cache(self, grain_size):
    """Test lt_multiply with different grain sizes when build_cache is True."""
    n = 4
    r = 3
    d = 8
    batches = 4

    # Instantiate inputs for the test.
    a = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    b = (
        jnp.arange(batches * n * r, 2 * batches * n * r)
        .astype(jnp.float32)
        .reshape((batches, n, r))
    )
    c = jnp.arange(batches * n * d).astype(jnp.float32).reshape((batches, n, d))

    # Expected outputs from the implementation.
    direct_result = jnp.tril(a @ b.transpose(0, 2, 1)) @ c
    direct_cache = b.transpose(0, 2, 1) @ c

    # Compute outputs from the implementation.
    lt_multiply_result, cache = (
        lower_triangular_multiplication.lt_multiply(
            a,
            b,
            c,
            grain_size=grain_size,
            build_cache=True,
        )
    )

    self.assertTrue(
        jnp.allclose(
            lt_multiply_result, direct_result, rtol=1e-3, atol=1e-5
        )
    )
    self.assertTrue(jnp.allclose(cache, direct_cache, rtol=1e-3, atol=1e-5))

  @parameterized.named_parameters(
      {'testcase_name': 'grain_size_1', 'grain_size': 1},
      {'testcase_name': 'grain_size_2', 'grain_size': 2},
      {'testcase_name': 'grain_size_4', 'grain_size': 4},
  )
  def test_lt_multiply_no_build_cache(self, grain_size):
    """Test lt_multiply with different grain sizes when build_cache is False."""
    n = 4
    r = 3
    d = 8
    batches = 4

    # Instantiate inputs for the test.
    a = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    b = (
        jnp.arange(batches * n * r, 2 * batches * n * r)
        .astype(jnp.float32)
        .reshape((batches, n, r))
    )
    c = jnp.arange(batches * n * d).astype(jnp.float32).reshape((batches, n, d))

    # Expected output from the implementation.
    direct_result = jnp.tril(a @ b.transpose(0, 2, 1)) @ c

    # Compute outputs from the implementation.
    lt_multiply_result, cache = (
        lower_triangular_multiplication.lt_multiply(
            a,
            b,
            c,
            grain_size=grain_size,
            build_cache=False,
        )
    )

    # Check that the expected outputs are close to the actual outputs.
    self.assertTrue(
        jnp.allclose(
            lt_multiply_result, direct_result, rtol=1e-3, atol=1e-5
        )
    )
    self.assertIsNone(cache)

  @parameterized.named_parameters(
      {'testcase_name': 'grain_size_1', 'grain_size': 1},
      {'testcase_name': 'grain_size_2', 'grain_size': 2},
      {'testcase_name': 'grain_size_4', 'grain_size': 4},
  )
  def test_tensor_lt_multiply_build_cache(self, grain_size):
    """Test tensor_lt_multiply with different grain sizes."""
    n = 4
    r = 3
    d = 8
    batches = 4

    # Instantiating inputs for the test.
    a = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    b = (
        jnp.arange(batches * n * r, 2 * batches * n * r)
        .astype(jnp.float32)
        .reshape((batches, n, r))
    )
    c = jnp.arange(batches * n * d).astype(jnp.float32).reshape((batches, n, d))

    # Expected outputs.
    direct_result = jnp.tril((a @ b.transpose(0, 2, 1)) ** 2) @ c
    direct_cache = jnp.einsum('...ti, ...tj, ...td -> ...ijd', b, b, c)

    tensor_lt_multiply_result, cache = (
        lower_triangular_multiplication.tensor_lt_multiply(
            a, b, c, grain_size=grain_size, build_cache=True
        )
    )

    # Checking closeness of the outputs from implementation with expected
    # outputs.

    self.assertTrue(
        jnp.allclose(
            tensor_lt_multiply_result,
            direct_result,
            rtol=1e-3,
            atol=1e-5,
        )
    )
    self.assertTrue(
        jnp.allclose(
            cache,
            direct_cache,
            rtol=1e-3,
            atol=1e-5,
        )
    )

  @parameterized.named_parameters(
      {'testcase_name': 'grain_size_1', 'grain_size': 1},
      {'testcase_name': 'grain_size_2', 'grain_size': 2},
      {'testcase_name': 'grain_size_4', 'grain_size': 4},
  )
  def test_tensor_lt_multiply_no_build_cache(self, grain_size):
    """Test tensor_lt_multiply when build_cache=False."""
    n = 32
    r = 8
    d = 8
    batches = 3

    # Instantiating inputs for the test.
    a = jnp.arange(batches * n * r).astype(jnp.float32).reshape((batches, n, r))
    b = (
        jnp.arange(batches * n * r, 2 * batches * n * r)
        .astype(jnp.float32)
        .reshape((batches, n, r))
    )
    c = jnp.arange(batches * n * d).astype(jnp.float32).reshape((batches, n, d))

    # Expected output.
    direct_result = jnp.tril((a @ b.transpose(0, 2, 1)) ** 2) @ c

    # Outputs from the implementation with different grain_size params.

    tensor_lt_multiply_result, cache = (
        lower_triangular_multiplication.tensor_lt_multiply(
            a, b, c, grain_size=grain_size, build_cache=False
        )
    )

    # Checking closeness of the outputs from implementation with expected
    # outputs.
    self.assertTrue(
        jnp.allclose(
            tensor_lt_multiply_result,
            direct_result,
            rtol=1e-3,
            atol=1e-5,
        )
    )
    self.assertIsNone(cache)


if __name__ == '__main__':
  absltest.main()

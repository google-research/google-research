# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Tests for gfsa.sparse_operator."""

import re
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from gfsa import sparse_operator


class SparseOperatorTest(parameterized.TestCase):

  def setUp(self):
    super(SparseOperatorTest, self).setUp()
    self.operator = sparse_operator.SparseCoordOperator(
        input_indices=jnp.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 0],
            [1, 1],
        ]),
        output_indices=jnp.array([[0], [1], [2], [3], [2]]),
        values=jnp.array([1., 10., 100., 1000., 10000.]),
    )

  def test_without_batch(self):
    """Tests a sparse operator without batching.

    Also has a repeated input-output pair, which should be summed.
    """
    v = jnp.array([
        [1., 2.],
        [3., 4.],
    ])
    result = self.operator.apply_add(v, jnp.zeros([4]))
    expected = np.array([1., 30., 40400., 3000.])
    np.testing.assert_allclose(result, expected)

  def test_batched(self):
    """Tests a sparse operator with a batched-over dimension."""
    v = jnp.array([
        [[1., 2.], [3., 4.], [5., 6.]],
        [[7., 8.], [9., 0.], [1., 2.]],
    ])
    result = self.operator.apply_add(
        v, jnp.zeros([3, 4]), in_dims=(2, 0), out_dims=(1,))
    expected = np.array([
        [1., 20., 80800., 2000.],
        [3., 40., 0., 4000.],
        [5., 60., 20200., 6000.],
    ])
    np.testing.assert_allclose(result, expected)

  def test_batched_transposed(self):
    """Tests transposing the linear operator and then batching."""
    v = np.array([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 0., 1., 2.],
    ])
    result = self.operator.transpose().apply_add(
        v, jnp.zeros([2, 3, 2]), in_dims=(1,), out_dims=(2, 0))
    expected = jnp.array([
        [[1., 4020.], [5., 8060.], [9., 2000.]],
        [[0., 30300.], [0., 70700], [0., 10100.]],
    ])
    np.testing.assert_allclose(result, expected)

  def test_sparse_coord_operator_is_a_pytree(self):
    """Tests that jax tree operations work on SparseCoordOperators."""
    op_with_zeros = jax.tree_map(jnp.zeros_like, self.operator)
    expected = sparse_operator.SparseCoordOperator(
        input_indices=jnp.zeros([5, 2], dtype=int),
        output_indices=jnp.zeros([5, 1], dtype=int),
        values=jnp.zeros([5], dtype=jnp.float32),
    )
    jax.tree_multimap(np.testing.assert_allclose, op_with_zeros, expected)

  @parameterized.named_parameters(
      {
          "testcase_name": "too_many_default_dims",
          "in_shape": [2, 2, 2],
          "out_shape": [2, 2, 2],
          "in_dims": None,
          "out_dims": None,
          "expected_error": "Expected 2 input dimensions, got 3"
      }, {
          "testcase_name": "too_many_in_dims",
          "in_shape": [2, 2, 2],
          "out_shape": [2, 2, 2],
          "in_dims": (0, 1, 2, 3),
          "out_dims": (0, 1),
          "expected_error": "Expected 2 input dimensions, got 4"
      }, {
          "testcase_name": "duplicate_in_dims",
          "in_shape": [2, 2, 2],
          "out_shape": [2, 2, 2],
          "in_dims": (1, 1),
          "out_dims": (0, 1),
          "expected_error": "Duplicate entries in in_dims: (1, 1)"
      }, {
          "testcase_name": "too_many_out_dims",
          "in_shape": [2, 2, 2],
          "out_shape": [2, 2, 2],
          "in_dims": (0, 1),
          "out_dims": (0, 1, 2),
          "expected_error": "Expected 2 output dimensions, got 3"
      }, {
          "testcase_name": "duplicate_out_dims",
          "in_shape": [2, 2, 2],
          "out_shape": [2, 2, 2],
          "in_dims": (0, 2),
          "out_dims": (1, 1),
          "expected_error": "Duplicate entries in out_dims: (1, 1)"
      }, {
          "testcase_name":
              "mismatched_batch",
          "in_shape": [2, 2, 3],
          "out_shape": [2, 2, 3],
          "in_dims": (0, 1),
          "out_dims": (1, 2),
          "expected_error":
              "Input and output must have the same batch sizes: got [3] and [2]"
      })
  def test_bad_args(self, in_shape, out_shape, in_dims, out_dims,
                    expected_error):
    operator = sparse_operator.SparseCoordOperator(
        input_indices=jnp.zeros([5, 2], dtype=int),
        output_indices=jnp.zeros([5, 2], dtype=int),
        values=jnp.zeros([5], dtype=jnp.float32),
    )
    with self.assertRaisesRegex(ValueError, re.escape(expected_error)):
      operator.apply_add(
          jnp.zeros(in_shape), jnp.zeros(out_shape), in_dims, out_dims)

  def test_pad_nonzeros(self):
    operator = sparse_operator.SparseCoordOperator(
        input_indices=jnp.arange(10).reshape([5, 2]),
        output_indices=jnp.arange(5).reshape([5, 1]),
        values=jnp.arange(5, dtype=jnp.float32),
    )
    padded_operator = operator.pad_nonzeros(7)
    apply_orig = operator.apply_add(
        jnp.arange(100).reshape([10, 10]), jnp.zeros([5]))
    apply_padded = padded_operator.apply_add(
        jnp.arange(100).reshape([10, 10]), jnp.zeros([5]))
    np.testing.assert_allclose(apply_orig, apply_padded)


if __name__ == "__main__":
  absltest.main()

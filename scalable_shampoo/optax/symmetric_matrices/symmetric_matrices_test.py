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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from scalable_shampoo.optax.symmetric_matrices import symmetric_matrices

SlicedSymmetricMatrix = symmetric_matrices.SlicedSymmetricMatrix
_PARAMS = [{
    "shape": (8, 8),
    "block_size": 2,
    "axes": (-1,)
}, {
    "shape": (16, 8),
    "block_size": 4,
    "axes": (-1,)
}, {
    "shape": (2, 8, 8),
    "block_size": 2,
    "axes": (0, 1),
}, {
    "shape": (4, 16, 8),
    "block_size": 4,
    "axes": (0, 2),
}]


# Used for testing.
@functools.partial(jax.jit, static_argnames=("block_size"))
def slice_symmetric_matrix(
    mat,
    block_size,
):
  """Returns sliced row blocks.

  Args:
    mat: A symmetric matrix.
    block_size: The size of the row slices.
  """
  num_rows = mat.shape[-2]
  num_cols = mat.shape[-1]
  if num_rows != num_cols:
    raise ValueError("mat is not square.")
  if num_rows % block_size != 0:
    raise ValueError("block size does not evenly divide rows.")
  return SlicedSymmetricMatrix(block_rows=[
      mat[Ellipsis, i * block_size:(i + 1) * block_size, 0:(i + 1) * block_size]
      for i in range(num_rows // block_size)
  ])


def _make_random_mat(shape, seed=0):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(key=key, shape=shape, dtype=jnp.bfloat16)


def _make_random_spd_mat(shape, axes, seed=0):
  """Make an SPD matrix by contracting a random matrix along specified axes."""
  mat = _make_random_mat(shape=shape, seed=seed)
  return symmetric_matrices.product_with_transpose(
      mat1=mat, mat2=mat, axes=(axes, axes))


class SymmetricMatricesTest(parameterized.TestCase):

  def assertAllClose(self, x, y, atol=1e-5, rtol=1e-5):

    def assert_close(x, y):
      x = x.astype("float32") if x.dtype == jnp.bfloat16 else x
      y = y.astype("float32") if y.dtype == jnp.bfloat16 else y
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

    jax.tree_util.tree_map(assert_close, x, y)

  def test_raises_dimension_error(self):
    with self.assertRaisesRegex(ValueError, "must be divisible by block_size"):
      symmetric_matrices.sliced_transposed_product(
          mat=_make_random_mat(shape=[16, 16]), block_size=5)

  @parameterized.parameters(_PARAMS)
  def test_symmetric_matrix_block_rows(self, shape, block_size, axes):
    """To test sliced multiplication, compare to slicing after full matmul."""
    mat = _make_random_mat(shape=shape, seed=0)
    expected_slices = slice_symmetric_matrix(
        mat=_make_random_spd_mat(shape, axes, seed=0), block_size=block_size)
    generated_slices = symmetric_matrices.sliced_transposed_product(
        mat=mat, axes=axes, block_size=block_size)
    self.assertAllClose(expected_slices.block_rows, generated_slices.block_rows)

  @parameterized.parameters(_PARAMS)
  def test_symmetric_matrix_block_rows_concat(self, shape, block_size, axes):
    """Test the sliced multiplication with concatenated slices."""
    mat = _make_random_mat(shape=shape, seed=0)
    expected_slices_concat = jnp.concatenate(
        slice_symmetric_matrix(
            mat=_make_random_spd_mat(shape, axes, seed=0),
            block_size=block_size).block_rows,
        axis=-1)
    generated_slices_concat = (
        symmetric_matrices.sliced_transposed_product_concat(
            mat=mat, axes=axes, block_size=block_size))
    self.assertAllClose(expected_slices_concat, generated_slices_concat)

  @parameterized.parameters(_PARAMS)
  def test_materialize_matrix(self, shape, block_size, axes):
    """To test materialization, slice and reconstruct a symmetric matrix."""
    sym_mat = _make_random_spd_mat(shape, axes, seed=0)
    sliced_mat = slice_symmetric_matrix(mat=sym_mat, block_size=block_size)
    reconstructed_sym_mat = symmetric_matrices.materialize_matrix(sliced_mat)
    self.assertAllClose(sym_mat, reconstructed_sym_mat)

  @parameterized.parameters(_PARAMS)
  def test_materialize_matrix_concat(self, shape, block_size, axes):
    """Test the materialization from concatenated slices."""
    sym_mat = _make_random_spd_mat(shape, axes, seed=0)
    sliced_mat_concat = jnp.concatenate(
        slice_symmetric_matrix(mat=sym_mat, block_size=block_size).block_rows,
        axis=-1)
    reconstructed_sym_mat = symmetric_matrices.materialize_matrix_from_concat(
        sliced_mat_concat, num_blocks=shape[-2] // block_size)
    self.assertAllClose(sym_mat, reconstructed_sym_mat)

  @parameterized.parameters(_PARAMS)
  def test_update_sliced_rows(self, shape, block_size, axes):
    alpha = 0.4
    beta = 3
    mat = _make_random_mat(shape=shape, seed=0)
    sym_mat = _make_random_spd_mat(shape, axes, seed=1)
    sym_mat_sliced = slice_symmetric_matrix(mat=sym_mat, block_size=block_size)
    expected_slices = slice_symmetric_matrix(
        mat=(alpha * symmetric_matrices.product_with_transpose(
            mat1=mat, mat2=mat, axes=(axes, axes)) + beta * sym_mat),
        block_size=block_size)
    generated_slices = symmetric_matrices.update_sliced_rows(
        symmetric_matrix=sym_mat_sliced,
        mat=mat,
        alpha=alpha,
        beta=beta,
        axes=axes)
    self.assertAllClose(expected_slices.block_rows, generated_slices.block_rows)

  @parameterized.parameters(_PARAMS)
  def test_find_num_blocks(self, shape, block_size, axes):
    mat_concat = jnp.concatenate(
        slice_symmetric_matrix(
            mat=_make_random_spd_mat(shape, axes, seed=0),
            block_size=block_size).block_rows,
        axis=-1)
    found_num_blocks = symmetric_matrices.find_num_blocks(mat_concat)
    expected_num_blocks = shape[-2] / block_size
    self.assertEqual(found_num_blocks, expected_num_blocks)

if __name__ == "__main__":
  absltest.main()

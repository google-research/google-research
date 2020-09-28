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

"""Test for sparse matrix class and utilities."""
from absl.testing import parameterized
import numpy as np
import scipy
import tensorflow.compat.v1 as tf

from sgk.sparse import connectors
from sgk.sparse import initializers
from sgk.sparse import sparse_matrix


@parameterized.parameters((4, 4, 0.0), (64, 128, 0.8), (512, 512, 0.64),
                          (273, 519, 0.71))
class SparseMatrixTest(tf.test.TestCase, parameterized.TestCase):

  def testCreateMatrix(self, m, n, sparsity):
    matrix = sparse_matrix.SparseMatrix(
        "matrix", [m, n], connector=connectors.Uniform(sparsity))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values, row_indices, row_offsets, column_indices = sess.run([
          matrix.values, matrix.row_indices, matrix.row_offsets,
          matrix.column_indices
      ])

      # Check the shape of the matrix.
      self.assertLen(values.shape, 1)
      self.assertLen(row_indices.shape, 1)
      self.assertLen(row_offsets.shape, 1)
      self.assertLen(column_indices.shape, 1)

      # Check the sparsity matches the target.
      target_nonzeros = m * n - int(round(sparsity * m * n))
      self.assertEqual(values.shape[0], target_nonzeros)

  def testDenseToSparse(self, m, n, sparsity):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Create a dense matrix in numpy with the specified sparsity.
    matrix = connector(initializer([m, n]))

    # Convert to a sparse numpy matrix.
    values, row_indices, row_offsets, column_indices = sparse_matrix._dense_to_sparse(
        matrix)

    # Create a scipy version of the matrix.
    expected_output = scipy.sparse.csr_matrix(
        (values, column_indices, row_offsets), [m, n])

    # Create the expected row indices.
    expected_row_indices = np.argsort(-1 * np.diff(expected_output.indptr))

    # Compare the matrices.
    self.assertAllEqual(expected_output.data, values)
    self.assertAllEqual(expected_output.indptr, row_offsets)
    self.assertAllEqual(expected_output.indices, column_indices)
    self.assertAllEqual(expected_row_indices, row_indices)


if __name__ == "__main__":
  tf.test.main()

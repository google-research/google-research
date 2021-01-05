# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

from absl.testing import parameterized
import scipy
import tensorflow.compat.v1 as tf

from sgk.sparse import sparse_matrix


class TestCase(tf.test.TestCase, parameterized.TestCase):

  def sparse_to_scipy(self, values, row_offsets, column_indices, shape):
    """Convert sparse numpy matrix into scipy sparse csr_matrix."""
    return scipy.sparse.csr_matrix((values, column_indices, row_offsets), shape)

  def dense_to_scipy(self, matrix):
    """Convert dense numpy matrix into scipy sparse csr_matrix."""
    values, _, row_offsets, column_indices = sparse_matrix._dense_to_sparse(
        matrix)
    return self.sparse_to_scipy(values, row_offsets, column_indices,
                                matrix.shape)

  def assert_sparse_matrix_equal(self, m1, m2, rtol=1e-6, atol=1e-6):
    """Verify that two sparse matrices are equal."""
    # Verify the shapes of the matrices are equal.
    self.assertAllEqual(m1.shape, m2.shape)

    # Verify the row offsets and column indices are equal.
    self.assertAllEqual(m1.indptr, m2.indptr)
    self.assertAllEqual(m1.indices, m2.indices)

    # Verify that the matrix values are (almost) equal.
    self.assertAllClose(m1.data, m2.data, rtol=rtol, atol=atol)

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


import numpy as np
import scipy.sparse.linalg
import tensorflow as tf


import sparse_deferred as sd
from sparse_deferred.algorithms import svd
import sparse_deferred.tf as sdtf


class SvdTest(tf.test.TestCase):

  def test_svd_works_like_numpy(self):
    n = 100
    row_ids = np.random.randint(0, n, size=2000)
    col_ids = np.random.randint(0, n, size=2000)
    row_ids = np.concatenate([row_ids, np.arange(n)])
    col_ids = np.concatenate([col_ids, np.arange(n)])
    row_ids, col_ids = zip(*set(zip(row_ids, col_ids)))  # Uniqify
    shape = (n, n)

    mat = sd.SparseMatrix(
        sdtf.engine, indices=(tf.constant(row_ids), tf.constant(col_ids)),
        dense_shape=shape)

    dense_mat = np.zeros(shape=shape)
    dense_mat[row_ids, col_ids] = 1.0
    u, s, vt = scipy.sparse.linalg.svds(dense_mat, k=3)
    s = s[::-1]  # Arrange from large to small (more standard)
    u = u[:, ::-1]
    v = vt[::-1].T

    our_u, our_s, our_v = svd.truncated_svd(sdtf.engine, mat, k=3,
                                            n_redundancy=10)
    our_v *= np.sign(our_u[0])
    our_u *= np.sign(our_u[0])
    v *= np.sign(u[0])
    u *= np.sign(u[0])

    self.assertAllClose(our_s, s, atol=1e-2)
    self.assertAllClose(our_v, v, atol=1e-2)
    self.assertAllClose(our_u, u, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()

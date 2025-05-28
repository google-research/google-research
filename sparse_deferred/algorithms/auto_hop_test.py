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

from absl.testing import parameterized
import tensorflow as tf
import sparse_deferred as sd
from sparse_deferred.algorithms import auto_hop
import sparse_deferred.tf as sdtf

# neighborhood <- logical switch
n_x_ls = sd.SparseMatrix(
    sdtf.engine,
    indices=([0, 0, 1, 1], [0, 1, 2, 3]),
    dense_shape=(2, 4),
)

# logical switch <- port
ls_x_p = sd.SparseMatrix(
    sdtf.engine,
    indices=([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7]),
    dense_shape=(4, 8),
)

# Neighborhood features
n_x = tf.constant([[7], [5]])

# Logical switch features
ls_x = tf.constant([[2, 4], [2, 1], [9, 5], [4, 3]])

# Port features
p_x = tf.constant([
    [7, 2, 3],
    [1, 8, 9],
    [3, 4, 6],
    [8, 6, 6],
    [1, 4, 8],
    [5, 8, 1],
    [1, 4, 5],
    [3, 5, 3],
])

# (target, source): [list of matrices with shape |target| x |source|]
adj_dictionaries = {
    ("n", "ls"): [n_x_ls],
    ("ls", "p"): [ls_x_p],
}

feat_dictionaries = {"n": n_x, "ls": ls_x, "p": p_x}


class TopologicalSortTest(tf.test.TestCase, parameterized.TestCase):

  def test_auto_hop_directional(self):

    result = auto_hop.recursive_propagate(
        feat_dictionaries, adj_dictionaries, "n", 2
    )

    self.assertAllEqual(len(result), 3)
    self.assertAllEqual(result[2], n_x)
    self.assertAllEqual(result[1], n_x_ls @ ls_x)
    self.assertAllEqual(result[0], n_x_ls @ (ls_x_p @ p_x))

  def test_auto_hop_bidirectional(self):

    result = auto_hop.recursive_propagate(
        feat_dictionaries, adj_dictionaries, "n", 2, bidirectional=True
    )

    self.assertAllEqual(len(result), 4)
    self.assertAllEqual(result[3], n_x)
    self.assertAllEqual(result[2], n_x_ls @ ls_x)
    self.assertAllEqual(
        result[1], n_x_ls @ (n_x_ls.T @ n_x)  # Go back to start point
    )
    self.assertAllEqual(
        result[0],
        n_x_ls @ (ls_x_p @ p_x),
    )

  def test_hops_larger_than_graph(self):

    result = auto_hop.recursive_propagate(
        feat_dictionaries, adj_dictionaries, "n", 10
    )

    self.assertAllEqual(len(result), 3)


if __name__ == "__main__":
  tf.test.main()

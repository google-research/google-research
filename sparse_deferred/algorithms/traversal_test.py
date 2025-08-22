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

import tensorflow as tf

from sparse_deferred.algorithms import traversal
import sparse_deferred.tf as sdtf


# 1 ---> 0----
#  \      \   \
#   \---> 2 --> 3
idx1 = [
    [0, 1],  # 0 <-- 1
    [2, 1],  # 2 <-- 1
    [2, 0],  # 2 <-- 1
    [3, 0],  # 2 <-- 1
    [3, 2],  # 2 <-- 1
]
shape1 = [4, 4]


# 3 ---> 0 --\
#  \          \
#   \---> 2 --> 1
#    \        /
#     \---> 4
idx2 = [
    [0, 3],
    [2, 3],
    [4, 3],
    [1, 0],
    [1, 2],
    [1, 4],
]
shape2 = [5, 5]


class ReachableTest(tf.test.TestCase):

  def test_one_hop_reachable(self):
    mat1 = sdtf.GatherScatterSparseMatrix(
        tf.sparse.SparseTensor(idx1, tf.ones(len(idx1)), shape1))
    start_points = tf.transpose(  # 5 traversals, each marking the start nodes.
        tf.constant([[True, False, False, False],
                     [False, True, False, False],
                     [False, False, True, False],
                     [False, False, False, True],
                     [True, True, True, False]]))
    reachables = traversal.one_hop_reachable(mat1, start_points)
    # 4 nodes, 5 traversals.
    self.assertAllEqual(reachables.shape, [4, 5])

    self.assertAllEqual(  # 1-hop from {0} reaches {2, 3}
        reachables[:, 0],
        [False, False, True, True])
    self.assertAllEqual(  # 1-hop from {1} reaches {0, 2}
        reachables[:, 1],
        [True, False, True, False])
    self.assertAllEqual(  # 1-hop from {2} reaches {3}
        reachables[:, 2],
        [False, False, False, True])
    self.assertAllEqual(  # 1-hop from {3} reaches nothing
        reachables[:, 3],
        [False, False, False, False])
    self.assertAllEqual(  # 1-hop from {0, 1, 2} reaches {0, 2, 3}
        reachables[:, 4],
        [True, False, True, True])

    mat2 = sdtf.GatherScatterSparseMatrix(
        tf.sparse.SparseTensor(idx2, tf.ones(len(idx2)), shape2))
    start_points = tf.transpose(
        tf.constant([[True, False, False, False, False],
                     [False, False, True, False, False],
                     [True, False, False, False, True]]))
    reachables = traversal.one_hop_reachable(mat2, start_points)
    # 5 nodes, 3 traversals.
    self.assertAllEqual(reachables.shape, [5, 3])
    self.assertAllEqual(  # 1-hop from {0} reaches {1}
        reachables[:, 0],
        [False, True, False, False, False])
    self.assertAllEqual(  # 1-hop from {2} reaches {1}
        reachables[:, 1],
        [False, True, False, False, False])
    self.assertAllEqual(  # 1-hop from {0, 4} reaches {1}
        reachables[:, 2],
        [False, True, False, False, False])

  def test_multi_hop_reachable(self):
    mat2 = sdtf.GatherScatterSparseMatrix(
        tf.sparse.SparseTensor(idx2, tf.ones(len(idx2)), shape2))
    start_points = tf.transpose(
        tf.constant([[True, False, False, False, False],
                     [False, False, True, False, False],
                     [True, False, False, False, True]]))
    reachables_1 = traversal.multi_hop_reachable(
        mat2, start_points, include_transpose=True)
    self.assertAllEqual(  # inclusive bidir 1-hop from {0} reaches {0, 1, 3}
        reachables_1[:, 0],
        [True, True, False, True, False])
    self.assertAllEqual(  # inclusive bidir 1-hop from {2} reaches {1, 2, 3}
        reachables_1[:, 1],
        [False, True, True, True, False])
    self.assertAllEqual(  # inclusive bidir 1-hop from {0, 4} reaches all but 2.
        reachables_1[:, 2],
        [True, True, False, True, True])
    reachables_2 = traversal.multi_hop_reachable(
        mat2, start_points, include_transpose=True, hops=2)
    # Everything is reachable within 2 hops (radius of graph).
    self.assertAllEqual(reachables_2, tf.ones_like(reachables_2))


if __name__ == '__main__':
  tf.test.main()

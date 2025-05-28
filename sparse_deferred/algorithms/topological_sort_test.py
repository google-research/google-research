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

from sparse_deferred.algorithms import topological_sort
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

order1 = [1, 0, 2, 3]
argsort_order1 = [1, 0, 2, 3]
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

order2 = [1, 2, 1, 0, 1]
argsort_order2 = [3, 0, 2, 4, 1]
reversed_argsort_order2 = [1, 0, 2, 4, 3]
shape2 = [5, 5]


class TopologicalSortTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single ordering',
          edges=idx1,
          shape=shape1,
          expected_order=order1,
      ),
      dict(
          testcase_name='ties in ordering',
          edges=idx2,
          shape=shape2,
          expected_order=order2,
      ),
  )
  def test_topological_sort(self, edges, shape, expected_order):
    self.assertAllEqual(
        expected_order,
        topological_sort.topological_order(
            sdtf.SparseMatrix(
                tf.sparse.SparseTensor(edges, tf.ones(len(edges)), shape)
            )
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='single ordering: ascending',
          edges=idx1,
          shape=shape1,
          direction='ASCENDING',
          expected_order=argsort_order1,
      ),
      dict(
          testcase_name='single ordering: descending',
          edges=idx1,
          shape=shape1,
          direction='DESCENDING',
          expected_order=argsort_order1[::-1],
      ),
      dict(
          testcase_name='ties in ordering: ascending',
          edges=idx2,
          shape=shape2,
          direction='ASCENDING',
          expected_order=argsort_order2,
      ),
      dict(
          testcase_name='ties in ordering: descending',
          edges=idx2,
          shape=shape2,
          direction='DESCENDING',
          expected_order=reversed_argsort_order2,
      ),
  )
  def test_argsort_topological_sort(
      self, edges, shape, direction, expected_order
  ):
    self.assertAllEqual(
        expected_order,
        topological_sort.argsort_topological_order(
            sdtf.SparseMatrix(
                tf.sparse.SparseTensor(edges, tf.ones(len(edges)), shape)
            ),
            direction=direction,
        ),
    )


if __name__ == '__main__':
  tf.test.main()

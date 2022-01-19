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

"""Tests for simulation functions library."""
from absl.testing import absltest
import numpy as np

from graph_embedding.simulations import heterogeneous_sbm_utils as hsu


class GetClusterTypeComponentsTest(absltest.TestCase):

  def test_equal_sizes(self):
    (cluster_index_lists, type_components) = hsu.GetClusterTypeComponents(
        [2, 2])
    self.assertEqual(cluster_index_lists, [[0, 1], [2, 3]])
    self.assertEqual(type_components, [[0, 2], [1, 3]])

  def test_first_greater_than_second(self):
    (cluster_index_lists, type_components) = hsu.GetClusterTypeComponents(
        [3, 2])
    self.assertEqual(cluster_index_lists, [[0, 1, 2], [3, 4]])
    self.assertEqual(type_components, [[0, 2, 4], [1, 3]])

  def test_second_greater_than_first(self):
    (cluster_index_lists, type_components) = hsu.GetClusterTypeComponents(
        [2, 3])
    self.assertEqual(cluster_index_lists, [[0, 1], [2, 3, 4]])
    self.assertEqual(type_components, [[0, 2, 4], [1, 3]])

  def test_three_type_mixed(self):
    (cluster_index_lists, type_components) = hsu.GetClusterTypeComponents(
        [3, 4, 2])
    self.assertEqual(cluster_index_lists, [[0, 1, 2], [3, 4, 5, 6], [7, 8]])
    self.assertEqual(type_components, [[0, 2, 4, 6, 8], [1, 3, 5, 7]])


class GetCrossLinksTest(absltest.TestCase):

  def test_equal_sizes(self):
    self.assertEqual(hsu.GetCrossLinks([2, 2], 0, 1),
                     [(0, 2), (1, 3)])

  def test_first_greater_than_second(self):
    self.assertEqual(hsu.GetCrossLinks([3, 2], 0, 1),
                     [(0, 4), (2, 4), (1, 3)])

  def test_second_greater_than_first(self):
    self.assertEqual(hsu.GetCrossLinks([2, 3], 0, 1),
                     [(0, 2), (0, 4), (1, 3)])

  def test_three_type_mixed12(self):
    self.assertEqual(hsu.GetCrossLinks([3, 4, 2], 0, 1),
                     [(0, 4), (0, 6), (2, 4), (2, 6), (1, 3), (1, 5)])


class GetPropMatTest(absltest.TestCase):

  def test_homogeneous_inptus(self):
    np.testing.assert_array_almost_equal(
        hsu.GetPropMat(3, 4.0),
        np.array([[4.0, 1.0, 1.0],
                  [1.0, 4.0, 1.0],
                  [1.0, 1.0, 4.0]]))

  def test_heterogeneous_inputs(self):
    np.testing.assert_array_almost_equal(
        hsu.GetPropMat(3, 3.0, 2, 2.0, 4.0),
        np.array([[3.0, 1.0, 1.0, 1.0, 4.0],
                  [1.0, 3.0, 1.0, 4.0, 1.0],
                  [1.0, 1.0, 3.0, 1.0, 4.0],
                  [1.0, 4.0, 1.0, 2.0, 1.0],
                  [4.0, 1.0, 4.0, 1.0, 2.0]]))


if __name__ == '__main__':
  absltest.main()

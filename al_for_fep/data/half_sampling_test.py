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

"""Tests for half_sampling data splitter."""
import unittest

from absl.testing import parameterized
import numpy as np

from al_for_fep.data import half_sampling


class HalfSamplingTest(parameterized.TestCase):

  def test_orthogonal_array_construction(self):
    np.testing.assert_array_almost_equal(
        half_sampling.orthogonal_array(2),
        np.array([[0, 0, 0, 0], [1, 1, 0, 1], [2, 0, 1, 1], [3, 1, 1, 0]]))

  @parameterized.parameters(2, 3, 4, 5)
  def test_large_orthogonal_array_balanced(self, log2_shards):
    result = half_sampling.orthogonal_array(log2_shards).transpose()

    self.assertEqual(len(result), len(np.unique(result)))
    np.testing.assert_array_equal(
        np.unique(result[1:].sum(axis=1)), np.array(2**(log2_shards - 1)))

  def test_split_success(self):
    half_sampler = half_sampling.HalfSamplingSplit(2)

    test_examples = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    test_targets = np.array([11, 12, 13, 14, 15, 16, 17, 18])

    datasets = np.array(list(half_sampler.split(test_examples, test_targets)),
                        dtype=object)

    expected_splits = [[[1, 2, 3, 4, 5, 6, 7, 8],
                        [11, 12, 13, 14, 15, 16, 17, 18]],
                       [[2, 4, 6, 8], [12, 14, 16, 18]],
                       [[1, 3, 5, 7], [11, 13, 15, 17]],
                       [[3, 4, 7, 8], [13, 14, 17, 18]],
                       [[1, 2, 5, 6], [11, 12, 15, 16]],
                       [[2, 3, 6, 7], [12, 13, 16, 17]],
                       [[1, 4, 5, 8], [11, 14, 15, 18]]]

    for result, expected_split in zip(datasets, expected_splits):
      self.assertLen(result, 2)
      self.assertLen(expected_split, 2)
      np.testing.assert_array_equal(result[0], expected_split[0])
      np.testing.assert_array_equal(result[1], expected_split[1])

  def test_split_mismatch_input_throws(self):
    with self.assertRaisesRegex(
        ValueError, 'Example list and target list should have the same length'):
      half_sampler = half_sampling.HalfSamplingSplit(2)
      _ = list(half_sampler.split(np.array([1, 2, 3, 4]), np.array([1])))


if __name__ == '__main__':
  unittest.main()

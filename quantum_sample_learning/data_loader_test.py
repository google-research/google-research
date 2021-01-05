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

# Lint as: python3
"""Tests for data_loader."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from quantum_sample_learning import data_loader


class DataLoaderTest(parameterized.TestCase):

  def test_convert_binary_digits_array_to_bitstrings(self):
    self.assertListEqual(
        list(data_loader.convert_binary_digits_array_to_bitstrings(
            np.array([[0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 0, 1]]))),
        [3, 4, 1])

  @parameterized.parameters(
      # 2 -> 010
      (2, 0, 0),
      (2, 1, 0),
      (2, 2, 1),
      (2, 3, 1),
      (2, 4, 1),
      # 3 -> 011
      (3, 0, 0),
      (3, 1, 1),
      (3, 2, 2),
      (3, 3, 2),
      # 4 -> 100
      (4, 0, 0),
      (4, 1, 0),
      (4, 2, 0),
      (4, 3, 1),
      )
  def test_count_set_bits(self, n, k, expected_count):
    self.assertEqual(data_loader.count_set_bits(n, k), expected_count)

  def test_reorder_subset_parity_subset_parity_size_1(self):
    # For a probability distribution of 8 bit strings: 0, 1, 2, 3
    # 0 -> 000
    # 1 -> 001
    # 2 -> 010
    # 3 -> 011
    # When subset_parity_size=1, the indice order are [0, 2], [1, 3].
    reordered = data_loader.reorder_subset_parity(
        probabilities=np.array([0., 0.2, 0.3, 0.5]), subset_parity_size=1)
    self.assertIsInstance(reordered, np.ndarray)
    self.assertIn(0., reordered[:2])
    self.assertIn(0.3, reordered[:2])
    self.assertIn(0.2, reordered[2:])
    self.assertIn(0.5, reordered[2:])

  def test_reorder_subset_parity_subset_parity_size_2(self):
    # For a probability distribution of 8 bit strings: 0, 1, 2, 3
    # 0 -> 000
    # 1 -> 001
    # 2 -> 010
    # 3 -> 011
    # When subset_parity_size=2, the indice order are 0, [1, 2], 3.
    reordered = data_loader.reorder_subset_parity(
        probabilities=np.array([0., 0.2, 0.3, 0.5]), subset_parity_size=1)
    self.assertIsInstance(reordered, np.ndarray)
    self.assertAlmostEqual(0., reordered[0])
    self.assertIn(0.2, reordered[1:3])
    self.assertIn(0.3, reordered[1:3])
    self.assertAlmostEqual(0.5, reordered[3])


if __name__ == '__main__':
  absltest.main()

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

"""Tests for top_k.

These are taken from
https://github.com/google-research/google-research/blob/master/dp_topk/baseline_mechanisms_test.py.
"""

from absl.testing import absltest
import numpy as np

from private_kendall import top_k


class TopKTest(absltest.TestCase):

  def test_sorted_top_k_returns_correct_input_ascending(self):
    counts = np.arange(10)
    k = 3
    expected_items = np.array([9, 8, 7])
    items = top_k.sorted_top_k(counts, k)
    np.testing.assert_array_equal(items, expected_items)

  def test_sorted_top_k_returns_correct_input_descending(self):
    counts = np.arange(10)[::-1]
    k = 3
    expected_items = np.arange(k)
    items = top_k.sorted_top_k(counts, k)
    np.testing.assert_array_equal(items, expected_items)

  def test_basic_peeling_mechanism_nondp_returns_correct_input_ascending(self):
    counts = np.arange(10)
    expected_items = np.array([9, 8, 7])
    k = 3
    epsilon = 1000
    l_inf_sensitivity = 0.1
    monotonic = False
    items = top_k.basic_peeling_mechanism(counts, k, epsilon, l_inf_sensitivity,
                                          monotonic)
    np.testing.assert_array_equal(items, expected_items)

  def test_basic_peeling_mechanism_nondp_returns_correct_input_descending(self):
    counts = np.arange(10)[::-1]
    k = 3
    expected_items = np.arange(k)
    epsilon = 1000
    l_inf_sensitivity = 0.1
    monotonic = False
    items = top_k.basic_peeling_mechanism(counts, k, epsilon, l_inf_sensitivity,
                                          monotonic)
    np.testing.assert_array_equal(items, expected_items)

  if __name__ == '__main__':
    absltest.main()

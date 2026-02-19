# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for utils.

This code is partially taken from
https://github.com/google-research/google-research/blob/master/dp_l2/utils_test.py
and
https://github.com/google-research/google-research/blob/master/dp_topk/baseline_mechanisms_test.py
"""

from absl.testing import absltest

from dp_mm_domain import utils


class UtilsTest(absltest.TestCase):

  # binary_search tests are taken from
  # https://github.com/google-research/google-research/blob/master/dp_l2/utils_test.py
  def test_linear_function(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -0.5, decreasing=True),
        0.501,
        delta=2e-3,
    )

  def test_quadratic_function(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -(x**2), -2.0, decreasing=True),
        1.415,
        delta=2e-3,
    )

  def test_exponential_function(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -(2**x), -5.0, decreasing=True),
        2.322,
        delta=2e-3,
    )

  def test_zero_threshold(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, 0.0, decreasing=True),
        0.001,
        delta=2e-3,
    )

  def test_negative_threshold(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -1.0, decreasing=True),
        1.001,
        delta=2e-3,
    )

  def test_high_threshold(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: -x, -100.0, decreasing=True),
        100.001,
        delta=2e-3,
    )

  def test_small_tolerance(self):
    self.assertAlmostEqual(
        utils.binary_search(
            lambda x: -x, -0.5, tolerance=1e-6, decreasing=True
        ),
        0.5,
        delta=2e-6,
    )

  def test_large_tolerance(self):
    self.assertAlmostEqual(
        utils.binary_search(
            lambda x: -x, -0.5, tolerance=1e-1, decreasing=True
        ),
        0.5,
        delta=2e-1,
    )

  def test_linear_func_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x, 0.5, decreasing=False),
        0.499,
        delta=2e-3,
    )

  def test_quadratic_func_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x**2, 2.0, decreasing=False),
        1.414,
        delta=2e-3,
    )

  def test_exponential_func_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: 2**x, 5.0, decreasing=False),
        2.322,
        delta=2e-3,
    )

  def test_zero_threshold_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x, 0.0, decreasing=False),
        -0.001,
        delta=2e-3,
    )

  def test_negative_threshold_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x, 1.0, decreasing=False),
        0.999,
        delta=2e-3,
    )

  def test_high_threshold_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x, 100.0, decreasing=False),
        99.999,
        delta=2e-3,
    )

  def test_small_tolerance_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x, 0.5, tolerance=1e-6, decreasing=False),
        0.5,
        delta=2e-6,
    )

  def test_large_tolerance_increasing(self):
    self.assertAlmostEqual(
        utils.binary_search(lambda x: x, 0.5, tolerance=1e-1, decreasing=False),
        0.5,
        delta=2e-1,
    )

  def test_l0_bound_users_correct_number_of_users(self):
    input_data = [
        [1, 2, 3, 4, 5],
        [1, 2, 4],
        [1, 2, 3],
        [5, 6],
        [1, 5, 6, 9, 10, 11],
        [7],
        [8, 9, 10, 11],
    ]
    l0_bound = 3
    output = utils.l0_bound_users(input_data, l0_bound)
    self.assertLen(output, len(input_data))

  def test_l0_bound_users_l0_bound_is_respected(self):
    input_data = [
        [1, 2, 3, 4, 5],
        [1, 2, 4],
        [1, 2, 3],
        [5, 6],
        [1, 5, 6, 9, 10, 11],
        [7],
        [8, 9, 10, 11],
    ]
    l0_bound = 3
    output = utils.l0_bound_users(input_data, l0_bound)
    for _, user in enumerate(output):
      self.assertLessEqual(len(user), l0_bound)

  def test_l0_bound_users_subsample_is_subset_of_input(self):
    input_data = [
        [1, 2, 3, 4, 5],
        [1, 2, 4],
        [1, 2, 3],
        [5, 6],
        [1, 5, 6, 9, 10, 11],
        [7],
        [8, 9, 10, 11],
    ]
    l0_bound = 3
    output = utils.l0_bound_users(input_data, l0_bound)
    self.assertLen(output, len(input_data))
    for idx, user in enumerate(output):
      self.assertContainsSubset(user, set(input_data[idx]))

  def test_l0_bound_users_sets_with_items_less_than_l0_bound_unchanged(self):
    input_data = [
        [1],
        [1, 2, 4],
        [1, 2],
        [5, 6],
        [1, 5, 6, 9, 10, 11],
        [7],
        [8, 9, 10, 11],
    ]
    l0_bound = 5
    output = utils.l0_bound_users(input_data, l0_bound)
    for idx, user in enumerate(output):
      if len(input_data[idx]) <= l0_bound:
        self.assertEqual(len(user), len(input_data[idx]))

  def test_get_hist(self):
    input_data = [
        [1, 2, 9, 3, 4, 5],
        [1, 2, 4],
        [1, 2, 3],
        [5, 6],
        [5, 6, 9, 10, 1, 11],
        [7],
        [8, 9, 10, 11],
    ]
    hist = utils.get_hist(input_data)
    self.assertEqual(hist[1], 4)
    self.assertEqual(hist[2], 3)
    self.assertEqual(hist[3], 2)
    self.assertEqual(hist[4], 2)
    self.assertEqual(hist[5], 3)
    self.assertEqual(hist[6], 2)
    self.assertEqual(hist[7], 1)
    self.assertEqual(hist[8], 1)
    self.assertEqual(hist[9], 3)
    self.assertEqual(hist[10], 2)

  def test_remove_elements_from_list_of_lists(self):
    list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    elements_to_remove = set([2, 5, 8])
    expected_list_of_lists = [[1, 3], [4, 6], [7, 9]]
    new_list_of_lists = utils.remove_elements_from_list_of_lists(
        list_of_lists, elements_to_remove
    )
    self.assertEqual(new_list_of_lists, expected_list_of_lists)

  # em_epsilon_cdp tests are taken from
  # https://github.com/google-research/google-research/blob/master/dp_topk/baseline_mechanisms.py
  def test_em_epsilon_cdp_delta_zero(self):
    k = 10
    epsilon = 1.1
    delta = 0
    local_epsilon = utils.em_epsilon_cdp(epsilon, delta, k)

    self.assertAlmostEqual(epsilon / k, local_epsilon, places=4)

  def test_em_epsilon_cdp_k_one(self):
    k = 1
    epsilon = 1.1
    delta = 0.1
    local_epsilon = utils.em_epsilon_cdp(epsilon, delta, k)

    self.assertAlmostEqual(epsilon, local_epsilon, places=4)

  def test_em_epsilon_cdp_k_ten(self):
    k = 10
    epsilon = 1.1
    delta = 0.1
    local_epsilon = utils.em_epsilon_cdp(epsilon, delta, k)

    self.assertAlmostEqual(0.29264, local_epsilon, places=4)

  def test_items_to_users_map(self):
    input_data = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 5],
        [6, 7],
    ]
    items_to_users = utils.get_items_to_users(input_data)
    self.assertDictEqual(
        items_to_users,
        {
            1: {0, 1, 2},
            2: {0, 1},
            3: {0},
            4: {1},
            5: {2},
            6: {3},
            7: {3},
        },
    )

  def test_remove_users_with_item_items_to_users(self):
    input_data = [[1, 2, 4], [1, 3, 4], [1, 3], [2], [2]]
    freq_hist = utils.get_hist(input_data)
    items_to_users = utils.get_items_to_users(input_data)

    items_to_users, _ = utils.remove_users_with_item(
        items_to_users, freq_hist, input_data, 3
    )
    self.assertDictEqual(
        items_to_users,
        {
            1: {0},
            2: {0, 3, 4},
            3: set(),
            4: {0},
        },
    )

  def test_remove_users_with_item_freq_hist(self):
    input_data = [[1, 2, 4], [1, 3, 4], [1, 3], [2], [2]]
    freq_hist = utils.get_hist(input_data)
    items_to_users = utils.get_items_to_users(input_data)

    _, freq_hist = utils.remove_users_with_item(
        items_to_users, freq_hist, input_data, 1
    )
    self.assertDictEqual(
        freq_hist,
        {
            1: 0,
            2: 2,
            3: 0,
            4: 0,
        },
    )


if __name__ == '__main__':
  absltest.main()

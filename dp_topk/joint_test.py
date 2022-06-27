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

"""Tests for joint."""

import operator

from absl.testing import absltest
import numpy as np

from dp_topk import joint
from dp_topk.differential_privacy import NeighborType


def assert_array_less_equal(x, y, err_msg='', verbose=True):
  return np.testing.assert_array_compare(
      operator.__le__,
      x,
      y,
      err_msg=err_msg,
      verbose=verbose,
      header='x is not less than or equal to y.',
      equal_inf=False)


class JointTest(absltest.TestCase):

  def test_make_diff_matrix_distinct_outputs(self):
    item_counts = np.repeat(np.arange(5), 5)[::-1]
    k = 5
    diff_matrix = joint.make_diff_matrix(item_counts, k)
    distinct_count = len(np.unique(diff_matrix))
    expected_distinct_count = len(item_counts) * k
    self.assertEqual(distinct_count, expected_distinct_count)

  def test_make_diff_matrix_increasing_rows(self):
    item_counts = np.repeat(np.arange(5), 5)[::-1]
    k = 5
    diff_matrix = joint.make_diff_matrix(item_counts, k)
    rows_increasing = [
        all(diff_matrix[row_idx][i] < diff_matrix[row_idx][i + 1]
            for i in range(len(item_counts) - 1))
        for row_idx in range(k)
    ]
    expected_rows_increasing = [1] * k
    np.testing.assert_array_equal(rows_increasing, expected_rows_increasing)

  def test_make_diff_matrix_decreasing_columns(self):
    item_counts = np.repeat(np.arange(5), 5)[::-1]
    k = 5
    diff_matrix = joint.make_diff_matrix(item_counts, k)
    columns_decreasing = [
        all(diff_matrix[i][col_idx] > diff_matrix[i + 1][col_idx]
            for i in range(k - 1))
        for col_idx in range(len(item_counts))
    ]
    expected_columns_decreasing = [1] * len(item_counts)
    np.testing.assert_array_equal(columns_decreasing,
                                  expected_columns_decreasing)

  def test_make_diff_matrix(self):
    item_counts = np.array([5, 5, 3])
    k = 2
    diff_matrix = joint.make_diff_matrix(item_counts, k)
    expected_diff_matrix = np.array([[1. / 3, 5. / 12, 2.5],
                                     [1. / 12, 1. / 6, 2.25]])
    np.testing.assert_array_almost_equal(
        diff_matrix, expected_diff_matrix, decimal=6)

  def test_get_diffs_to_positions(self):
    diff_matrix = np.array([[21, 4, 3, 12, 9], [7, 6, 22, 13, 0],
                            [17, 10, 5, 15, 2], [8, 16, 20, 14, 24],
                            [19, 11, 1, 18, 23]])
    diffs_to_positions = joint.get_diffs_to_positions(diff_matrix)
    expected_diffs_to_positions = np.array([[
        1, 4, 2, 0, 0, 2, 1, 1, 3, 0, 2, 4, 0, 1, 3, 2, 3, 2, 4, 4, 3, 0, 1, 4,
        3
    ],
                                            [
                                                4, 2, 4, 2, 1, 2, 1, 0, 0, 4, 1,
                                                1, 3, 3, 3, 3, 1, 0, 3, 0, 2, 0,
                                                2, 4, 4
                                            ]])
    np.testing.assert_array_equal(diffs_to_positions,
                                  expected_diffs_to_positions)

  def test_brute_compute_log_diff_counts(self):
    diff_matrix = np.array([[0.3125, 2.375, 2.4375, 6.5],
                            [-1.9375, 0.125, 0.1875, 4.25]])
    with np.errstate(divide='ignore'):
      expected_log_diff_counts = np.log([0, 0, 0, 2, 2, 2, 3, 3])
    brute_log_diff_counts = joint.brute_compute_log_diff_counts(
        diff_matrix, np.sort(np.ndarray.flatten(diff_matrix)))
    np.testing.assert_array_equal(brute_log_diff_counts,
                                  expected_log_diff_counts)

  def test_compute_log_diff_counts(self):
    for d in [5, 6, 7]:
      for k in [2, 3, 4]:
        for _ in range(100):
          uniform_item_counts = np.sort(np.random.choice(10 * d, size=d))[::-1]
          diff_matrix = joint.make_diff_matrix(uniform_item_counts, k)
          diffs_to_positions = joint.get_diffs_to_positions(diff_matrix)
          sorted_diffs = np.sort(diff_matrix, axis=None)
          log_diff_counts = joint.compute_log_diff_counts(
              diff_matrix, diffs_to_positions)
          expected_log_diff_counts = joint.brute_compute_log_diff_counts(
              diff_matrix, sorted_diffs)
          np.testing.assert_array_almost_equal(
              log_diff_counts, expected_log_diff_counts, decimal=6)

  def test_racing_sample_distribution(self):
    log_terms = np.array([0, 0, 1, 1.5, 2, 3])
    sampled_counts = np.zeros(len(log_terms))
    num_trials = 10000
    for _ in range(num_trials):
      sampled_counts[joint.racing_sample(log_terms)] += 1
    expected_sample_probs = np.array(
        [0.0273, 0.0273, 0.0741, 0.122, 0.201, 0.548])
    expected_sample_widths = 4 * np.sqrt(
        expected_sample_probs * (1 - expected_sample_probs) / num_trials)
    np.testing.assert_array_less(
        sampled_counts,
        num_trials * (expected_sample_probs + expected_sample_widths))
    np.testing.assert_array_less(
        num_trials * (expected_sample_probs - expected_sample_widths),
        sampled_counts)

  def test_sample_diff_idx_distribution_add_remove(self):
    sorted_diffs = np.array(
        [-1.9375, 0.125, 0.1875, 0.3125, 2.375, 2.4375, 4.25, 6.5])
    with np.errstate(divide='ignore'):
      log_diff_counts = np.log([0, 0, 1, 3, 4, 2, 15, 19])
    sampled_counts = np.zeros(len(log_diff_counts))
    eps = 1.
    num_trials = 100000
    for _ in range(num_trials):
      sampled_counts[joint.sample_diff_idx(log_diff_counts, sorted_diffs, eps,
                                           NeighborType.ADD_REMOVE)] += 1
    expected_sample_probs = np.array(
        [0, 0, 0.109, 0.327, 0.160, 0.0801, 0.221, 0.103])
    expected_sample_widths = 4 * np.sqrt(
        expected_sample_probs * (1 - expected_sample_probs) / num_trials)
    assert_array_less_equal(
        sampled_counts,
        num_trials * (expected_sample_probs + expected_sample_widths))
    assert_array_less_equal(
        num_trials * (expected_sample_probs - expected_sample_widths),
        sampled_counts)

  def test_sample_diff_idx_distribution_swap(self):
    sorted_diffs = np.array(
        [-1.9375, 0.125, 0.1875, 0.3125, 2.375, 2.4375, 4.25, 6.5])
    with np.errstate(divide='ignore'):
      log_diff_counts = np.log([0, 0, 1, 3, 4, 2, 15, 19])
    sampled_counts = np.zeros(len(log_diff_counts))
    eps = 1.
    num_trials = 10000
    for _ in range(num_trials):
      sampled_counts[joint.sample_diff_idx(log_diff_counts, sorted_diffs, eps,
                                           NeighborType.SWAP)] += 1
    expected_sample_probs = np.array(
        [0, 0, 0.0575, 0.172, 0.139, 0.0697, 0.317, 0.244])
    expected_sample_widths = 4 * np.sqrt(
        expected_sample_probs * (1 - expected_sample_probs) / num_trials)
    assert_array_less_equal(
        sampled_counts,
        num_trials * (expected_sample_probs + expected_sample_widths))
    assert_array_less_equal(
        num_trials * (expected_sample_probs - expected_sample_widths),
        sampled_counts)

  def test_sequence_from_diff_pick_first(self):
    diff_matrix = np.array([[
        0.3611111111, 5.3888888889, 10.4166666667, 10.4444444444, 13.4722222222,
        13.5
    ],
                            [
                                -4.8055555556, 0.2222222222, 5.25, 5.2777777778,
                                8.3055555556, 8.3333333333
                            ],
                            [
                                -9.9722222222, -4.9444444444, 0.0833333333,
                                0.1111111111, 3.1388888889, 3.1666666667
                            ]])
    diff = 5.25
    expected_sequence = np.array([0, 2, 1])
    sequence = joint.sequence_from_diff(diff, 1, 2, diff_matrix, lambda x: x[0])
    np.testing.assert_array_equal(sequence, expected_sequence)

  def test_sequence_from_diff_pick_last(self):
    diff_matrix = np.array([[
        0.3611111111, 5.3888888889, 10.4166666667, 10.4444444444, 13.4722222222,
        13.5
    ],
                            [
                                -4.8055555556, 0.2222222222, 5.25, 5.2777777778,
                                8.3055555556, 8.3333333333
                            ],
                            [
                                -9.9722222222, -4.9444444444, 0.0833333333,
                                0.1111111111, 3.1388888889, 3.1666666667
                            ]])
    diff = 5.25
    expected_sequence = np.array([0, 2, 5])
    sequence = joint.sequence_from_diff(diff, 1, 2, diff_matrix,
                                        lambda x: x[-1])
    np.testing.assert_array_equal(sequence, expected_sequence)

  def test_sequence_from_diff_distribution(self):
    diff_matrix = np.array([[0.3125, 2.375, 2.4375, 6.5],
                            [-1.9375, 0.125, 0.1875, 4.25]])
    diff = 4.25
    sequence_counts = np.zeros(4)
    num_trials = 10000
    for _ in range(num_trials):
      sequence = joint.sequence_from_diff(diff, 1, 3, diff_matrix)
      if sequence[0] not in [0, 1, 2] or sequence[1] != 3:
        sequence_counts[3] += 1
      else:
        sequence_counts[sequence[0]] += 1
    expected_sequence_probs = np.array([1. / 3, 1. / 3, 1. / 3, 0])
    expected_sequence_widths = 4 * np.sqrt(
        expected_sequence_probs * (1 - expected_sequence_probs) / num_trials)
    assert_array_less_equal(
        sequence_counts,
        num_trials * (expected_sequence_probs + expected_sequence_widths))
    assert_array_less_equal(
        num_trials * (expected_sequence_probs - expected_sequence_widths),
        sequence_counts)

  def test_joint_distribution_add_remove(self):
    item_counts = np.array([10, 10, 5])
    k = 2
    eps = 1
    neighbor_type = NeighborType.ADD_REMOVE
    sensitivity = 1
    diff_matrix = np.array([[1. / 3, 5. / 12, 5.5], [1. / 12, 1. / 6, 5.25]])
    sequence_counts = np.zeros(7)
    num_trials = 10000
    for _ in range(num_trials):
      sequence = joint.joint(item_counts, k, eps, neighbor_type)
      sequence_diff = max(diff_matrix[0, sequence[0]], diff_matrix[1,
                                                                   sequence[1]])
      if sequence_diff == 1. / 12:
        sequence_counts[0] += 1
      elif sequence_diff == 1. / 6:
        sequence_counts[1] += 1
      elif sequence_diff == 1. / 3:
        sequence_counts[2] += 1
      elif sequence_diff == 5. / 12:
        sequence_counts[3] += 1
      elif sequence_diff == 5.25:
        sequence_counts[4] += 1
      elif sequence_diff == 5.5:
        sequence_counts[5] += 1
      else:
        sequence_counts[6] += 1
    with np.errstate(divide='ignore'):
      log_diff_counts = np.log(np.array([0, 0, 1, 1, 2, 2]))
    sorted_diffs = np.array([1. / 12, 1. / 6, 1. / 3, 5. / 12, 5.25, 5.5])
    unnormalized_probabilities = np.exp(log_diff_counts - (eps * sorted_diffs /
                                                           (2 * sensitivity)))
    expected_sequence_probs = np.zeros(7)
    expected_sequence_probs[:-1] = unnormalized_probabilities / np.sum(
        unnormalized_probabilities)
    expected_sequence_widths = 4 * np.sqrt(
        expected_sequence_probs * (1 - expected_sequence_probs) / num_trials)
    assert_array_less_equal(
        sequence_counts,
        num_trials * (expected_sequence_probs + expected_sequence_widths))
    assert_array_less_equal(
        num_trials * (expected_sequence_probs - expected_sequence_widths),
        sequence_counts)

  def test_joint_distribution_swap(self):
    item_counts = np.array([10, 10, 5])
    k = 2
    eps = 1
    neighbor_type = NeighborType.SWAP
    sensitivity = 2
    diff_matrix = np.array([[1. / 3, 5. / 12, 5.5], [1. / 12, 1. / 6, 5.25]])
    sequence_counts = np.zeros(7)
    num_trials = 10000
    for _ in range(num_trials):
      sequence = joint.joint(item_counts, k, eps, neighbor_type)
      sequence_diff = max(diff_matrix[0, sequence[0]], diff_matrix[1,
                                                                   sequence[1]])
      if sequence_diff == 1. / 12:
        sequence_counts[0] += 1
      elif sequence_diff == 1. / 6:
        sequence_counts[1] += 1
      elif sequence_diff == 1. / 3:
        sequence_counts[2] += 1
      elif sequence_diff == 5. / 12:
        sequence_counts[3] += 1
      elif sequence_diff == 5.25:
        sequence_counts[4] += 1
      elif sequence_diff == 5.5:
        sequence_counts[5] += 1
      else:
        sequence_counts[6] += 1
    with np.errstate(divide='ignore'):
      log_diff_counts = np.log(np.array([0, 0, 1, 1, 2, 2]))
    sorted_diffs = np.array([1. / 12, 1. / 6, 1. / 3, 5. / 12, 5.25, 5.5])
    unnormalized_probabilities = np.exp(log_diff_counts - (eps * sorted_diffs /
                                                           (2 * sensitivity)))
    expected_sequence_probs = np.zeros(7)
    expected_sequence_probs[:-1] = unnormalized_probabilities / np.sum(
        unnormalized_probabilities)
    expected_sequence_widths = 4 * np.sqrt(
        expected_sequence_probs * (1 - expected_sequence_probs) / num_trials)
    assert_array_less_equal(
        sequence_counts,
        num_trials * (expected_sequence_probs + expected_sequence_widths))
    assert_array_less_equal(
        num_trials * (expected_sequence_probs - expected_sequence_widths),
        sequence_counts)

if __name__ == '__main__':
  absltest.main()

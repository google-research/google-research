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

"""Tests for tukey."""

from absl.testing import absltest
import numpy as np

from dp_regression import tukey


def assert_array_less_equal(x, y, err_msg='', verbose=True):
  return np.testing.assert_array_compare(
      operator.__le__,
      x,
      y,
      err_msg=err_msg,
      verbose=verbose,
      header='x is not less than or equal to y.',
      equal_inf=False)


class TukeyTest(absltest.TestCase):

  def test_perturb_and_sort_matrix(self):
    projections = np.asarray([[-0.3, 22, 47, -11, 5], [99, 1, 2, -11, 8.4]])
    sorted_projection_1 = np.asarray([-11, -0.3, 5, 22, 47])
    sorted_projection_2 = np.asarray([-11, 1, 2, 8.4, 99])
    perturbed_and_sorted_matrix = tukey.perturb_and_sort_matrix(projections)
    np.testing.assert_array_less(perturbed_and_sorted_matrix[0],
                                 sorted_projection_1 + 1e-10)
    np.testing.assert_array_less(sorted_projection_1 - 1e-10,
                                 perturbed_and_sorted_matrix[0])
    np.testing.assert_array_less(perturbed_and_sorted_matrix[1],
                                 sorted_projection_2 + 1e-10)
    np.testing.assert_array_less(sorted_projection_2 - 1e-10,
                                 perturbed_and_sorted_matrix[1])

  def test_log_measure_geq_all_depths(self):
    projections = np.asarray([[-0.5, -0.4, -0.2, 0.1, 0.5],
                              [-0.1, 0.4, 1.0, 1.7, 2.5]])
    log_measure_geq_all = tukey.log_measure_geq_all_depths(projections)
    # 1 * 2.6 = 2.6, 0.5 + 1.3 = 0.65, depth 3 has volume 0
    np.testing.assert_array_almost_equal(
        np.exp(log_measure_geq_all), np.asarray([2.6, 0.65, 0]))

  def test_racing_sample_distribution(self):
    log_terms = np.array([0, 0, 1, 1.5, 2, 3])
    sampled_counts = np.zeros(len(log_terms))
    num_trials = 10000
    for _ in range(num_trials):
      sampled_counts[tukey.racing_sample(log_terms)] += 1
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

  def test_restricted_racing_sample_depth_distribution(self):
    projections = np.asarray([[1, 2, 2.5, 3, 3.5, 4.4], [-8, 3, 4, 7, 11, 13],
                              [-2, 0, 0.1, 0.2, 0.3, 6]])
    epsilon = 1.1
    restricted_depth = 2
    sampled_counts = np.zeros(2)
    num_trials = 10000
    for _ in range(num_trials):
      sampled_counts[tukey.restricted_racing_sample_depth(
          projections, epsilon, restricted_depth) - 1] += 1
    # volume of depth >= 2 = 1.5 * 8 * 0.3 = 3.6
    # volume of depth >= 3 = 0.5 * 3 * 0.1 = 0.15
    # volume of depth exactly 2 = 3.6 - 0.15 = 3.45
    unnormalized_mass_depth_2 = 3.45 * np.exp(epsilon * 2)
    unnormalized_mass_depth_3 = 0.15 * np.exp(epsilon * 3)
    total_mass = unnormalized_mass_depth_2 + unnormalized_mass_depth_3
    expected_sample_probs = np.array(
        [unnormalized_mass_depth_2, unnormalized_mass_depth_3])
    expected_sample_probs = expected_sample_probs / total_mass
    expected_sample_widths = 4 * np.sqrt(
        expected_sample_probs * (1 - expected_sample_probs) / num_trials)
    np.testing.assert_array_less(
        sampled_counts,
        num_trials * (expected_sample_probs + expected_sample_widths))
    np.testing.assert_array_less(
        num_trials * (expected_sample_probs - expected_sample_widths),
        sampled_counts)

  def test_sample_geq_1d(self):
    projection = np.asarray([-0.5, -0.4, -0.33, -0.2, 0.03, 0.1])
    depths = [1, 2, 3]
    num_trials = 1000
    num_wrong_depth = 0
    for depth in depths:
      for _ in range(num_trials):
        sample = tukey.sample_exact_1d(depth, projection)
        insert_index = np.searchsorted(projection, sample)
        sample_depth = min(insert_index, len(projection) - insert_index)
        if sample_depth < depth:
          num_wrong_depth += 1
    self.assertEqual(num_wrong_depth, 0)

  def test_sample_exact_1d(self):
    projection = np.asarray([-0.5, -0.4, -0.33, -0.2, 0.03, 0.1])
    depths = [1, 2, 3]
    num_trials = 1000
    num_wrong_depth = 0
    for depth in depths:
      for _ in range(num_trials):
        sample = tukey.sample_exact_1d(depth, projection)
        insert_index = np.searchsorted(projection, sample)
        sample_depth = min(insert_index, len(projection) - insert_index)
        if sample_depth != depth:
          num_wrong_depth += 1
    self.assertEqual(num_wrong_depth, 0)

  def test_sample_exact(self):
    projections = np.asarray([[-0.5, -0.4, -0.33, -0.2, 0.03, 0.1],
                              [-0.1, 0.4, 0.5, 1.0, 1.5, 1.7]])
    depths = [1, 2, 3]
    num_trials = 1000
    num_wrong_depth = 0
    for depth in depths:
      for _ in range(num_trials):
        sample = tukey.sample_exact(depth, projections)
        insert_indices = [
            np.searchsorted(projections[i], sample[i])
            for i in range(len(projections))
        ]
        sample_depths = [
            min(insert_index,
                len(projections[0]) - insert_index)
            for insert_index in insert_indices
        ]
        if min(sample_depths) != depth:
          num_wrong_depth += 1
    self.assertEqual(num_wrong_depth, 0)

  def test_distance_to_unsafety_dense_points(self):
    volumes = np.array([
        128, 64, 1, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993, 0.992,
        0.991, 0.990, 0.989, 0.988, 0.987
    ])
    log_volumes = np.log(volumes)
    epsilon = 1.1
    delta = 0.5
    t = 6
    k_low = -1
    k_high = t-1
    expected_distance = 1
    result = tukey.distance_to_unsafety(log_volumes, epsilon, delta, t, k_low,
                                        k_high)
    self.assertEqual(result, expected_distance)

  def test_distance_to_unsafety_sparse_points(self):
    volumes = np.array([128, 64, 32,16,8,4,2,1])
    log_volumes = np.log(volumes)
    epsilon = 1.1
    delta = 0.5
    t = 4
    k_low = -1
    k_high = t-1
    expected_distance = -1
    result = tukey.distance_to_unsafety(log_volumes, epsilon, delta, t, k_low,
                                        k_high)
    self.assertEqual(result, expected_distance)

if __name__ == '__main__':
  absltest.main()

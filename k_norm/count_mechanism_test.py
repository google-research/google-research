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

"""Tests for count_mechanism."""

from absl.testing import absltest
import numpy as np

from k_norm import count_mechanism
from k_norm import sum_mechanism


class CountMechanismTest(absltest.TestCase):

  def test_sample_orthant_num_positive_multinomial_distribution(self):
    d = 4
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    k = 2
    num_trials = 1000
    # Possible returned values range over j in 0, 1, 2, 3, 4, and by the
    # proof of Theorem 8.6 in the paper, each class j has total unnormalized
    # probability mass
    # z_j' = (sum_i=1^k A_{j,i-1}) * (sum_i=1^k A_{d-j,i-1}) / (j! (d-j)!)
    # for matrix of Eulerian numbers
    # A = [[1], [1], [1, 1], [1, 4, 1], [1, 11, 11, 1]]
    # where trailing 0s are dropped in each row of A. Thus
    # z_0' = 1 * (1 + 11) / (0! * 4!) = 1/2
    # z_1' = 1 * (1 + 4) / (1! * 3!) = 5/6
    # z_2' = 2 * 2 / (2! * 2!) = 1
    # z_3' = z_1' = 5/6
    # z_4' = z_0' = 1/2
    # and the normalized weights are
    # z_0 = 1/2 / (22/6) = 3/22
    # z_1 = 5/6 / (22/6) = 5/22
    # z_2 = 1 / (22/6) = 6/22
    # z_3 = z_1 = 5/22
    # z_4 = z_0 = 3/22.
    orthant_js = np.zeros((num_trials))
    for i in range(num_trials):
      orthant_js[i] = count_mechanism.sample_orthant_num_positive(
          eulerian_numbers, k
      )
    _, counts = np.unique(orthant_js, axis=0, return_counts=True)
    # The five possible values for ascent_indices follow a multinomial
    # distribution, so we apply the same statistical testing logic used in
    # test_compute_add_ascent_indices_multinomial_distribution in
    # sum_mechanism_test.
    expected_counts = num_trials * np.asarray([3/22, 5/22, 6/22, 5/22, 3/22])
    ci_radii = np.sqrt(expected_counts)
    for idx in range(d+1):
      self.assertLess(expected_counts[idx] - 3 * ci_radii[idx], counts[idx])
      self.assertLess(counts[idx], expected_counts[idx] + 3 * ci_radii[idx])

  def test_sample_from_orthant_num_positive(self):
    d = 4
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    k = 2
    num_trials = 1000
    samples = np.zeros((num_trials, d))
    nums_positive_coordinates = np.random.choice(d + 1, size=num_trials)
    for i in range(num_trials):
      samples[i] = count_mechanism.sample_from_orthant(
          eulerian_numbers, nums_positive_coordinates[i], k
      )
    nums_actual_positive_coordinates = np.sum(samples >= 0, axis=1)
    for idx in range(num_trials):
      self.assertEqual(
          nums_actual_positive_coordinates[idx], nums_positive_coordinates[idx]
      )

  def test_sample_count_ball_norms(self):
    d = 4
    k = 2
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    num_samples = 1000
    samples = np.asarray([
        count_mechanism.sample_count_ball(eulerian_numbers, k)
        for _ in range(num_samples)
    ])
    np.testing.assert_array_less(np.linalg.norm(samples, axis=1, ord=1),
                                 np.ones(num_samples) * k)
    np.testing.assert_array_less(np.linalg.norm(samples, axis=1, ord=np.inf),
                                 np.ones(num_samples))


if __name__ == '__main__':
  absltest.main()

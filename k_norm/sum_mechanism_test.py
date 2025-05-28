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

"""Tests for sum_mechanism."""

from absl.testing import absltest
import numpy as np

from k_norm import sum_mechanism


class SumMechanismTest(absltest.TestCase):

  def test_compute_eulerian_numbers(self):
    d = 4
    # See, for example, the table at
    # https://en.wikipedia.org/wiki/Eulerian_number#Basic_properties.
    eulerian_numbers = [[1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 4, 1, 0, 0],
                        [1, 11, 11, 1, 0]]
    output = sum_mechanism.compute_eulerian_numbers(d)
    for row in range(d+1):
      np.testing.assert_array_equal(output[row], eulerian_numbers[row])

  def test_compute_add_ascent_indices_multinomial_distribution(self):
    d = 4
    num_ascents = 2
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    num_trials = 1000
    # Possible outputs for compute_add_ascent_indices(4, 2) are [0, 1, 1, 0],
    # [0, 1, 0, 1], and [0, 0, 1, 1]. These respectively correspond to
    # permutations (4123), (1423), and (1243); (1324), (3124), (1342), and
    # (3412); and (2314), (2134), (2341), and (2413).
    counts = [0, 0, 0]
    i = 0
    while i < num_trials:
      ascent_indices = list(
          sum_mechanism.compute_add_ascent_indices(eulerian_numbers,
                                                   num_ascents)
          )
      if ascent_indices == [0, 1, 1, 0]:
        counts[0] += 1
      elif ascent_indices == [0, 1, 0, 1]:
        counts[1] += 1
      elif ascent_indices == [0, 0, 1, 1]:
        counts[2] += 1
      i += 1
    # The three possible values for ascent_indices follow a multinomial
    # distribution. n samples from a multinomial with probabilities p_1 = 3/11,
    # p_2 = 4/11, and p_3 = 4/11 have respect expected counts 3n/11, 4n/11, and
    # 4n/11 and standard deviations sqrt(n(3/11)(8/11)), sqrt(n(4/11)(7/11)),
    # and sqrt(n(4/11)(7/11)), so we take three standard deviations as a
    # conservative confidence interval radius.
    expected_count_1 = 3 * num_trials / 11
    expected_count_2 = 4 * num_trials / 11
    ci_radius_1 = np.sqrt(expected_count_1 * 8 / 11)
    ci_radius_2 = np.sqrt(expected_count_2 * 7 / 11)
    self.assertLess(expected_count_1 - 3 * ci_radius_1, counts[0])
    self.assertLess(counts[0], expected_count_1 + 3 * ci_radius_1)
    self.assertLess(expected_count_2 - 3 * ci_radius_2, counts[1])
    self.assertLess(counts[1], expected_count_2 + 3 * ci_radius_2)
    self.assertLess(expected_count_2 - 3 * ci_radius_2, counts[2])
    self.assertLess(counts[2], expected_count_2 + 3 * ci_radius_2)

  def test_sample_permutation_with_ascents_uniform_distribution(self):
    d = 4
    num_ascents = 2
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    num_trials = 1000
    # The possible permutations in S_{4, 2} are (1243), (1324), (1342), (1423),
    # (2134), (2314), (2341), (2413), (3124), (3412), (4123), and each has a
    # 1 / 11 probability in a uniform distribution. The confidence intervals
    # are generated using the same logic as test_compute_add_ascent_indices.
    permutations = np.zeros((num_trials, d))
    i = 0
    while i < num_trials:
      permutations[i] = sum_mechanism.sample_permutation_with_ascents(
          eulerian_numbers, num_ascents
      )
      i += 1
    _, counts = np.unique(permutations, axis=0, return_counts=True)
    expected_count = num_trials / 11
    ci_radius = np.sqrt(expected_count * 10 / 11)
    for count in counts:
      self.assertLess(expected_count - 3 * ci_radius, count)
      self.assertLess(count, expected_count + 3 * ci_radius)

  def test_phi(self):
    # phi(x) = y where y_j = x_{j-1} - x_j + (x_{j-1} < x_j) and x_0 = 0, so
    # phi((0.1, 0.2, 0.1)) = (0 - 0.1 + 1, 0.1 - 0.2 + 1, 0.2 - 0.1)
    x = [0.1, 0.2, 0.1]
    expected_phi = [0.9, 0.9, 0.1]
    np.testing.assert_array_equal(sum_mechanism.phi(x), expected_phi)

  def test_sample_slice_index_binomial_distribution(self):
    d = 3
    k = 2
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    # |R_1| = 1 / 3! = 1/6 and |R_2| = 4 / 3! = 2/3, so the final probability
    # masses are 1/5 for slice index 1 and 4/5 for slice index 2.
    # confidence intervals are generated using the same logic as
    # test_compute_add_ascent_indices, though we only need to test one slice
    # index count.
    num_trials = 1000
    index_1_count = 0
    i = 0
    while i < num_trials:
      if sum_mechanism.sample_slice_index(eulerian_numbers, k) == 1:
        index_1_count += 1
      i += 1
    expected_count_1 = 4 * num_trials / 5
    ci_radius_1 = np.sqrt(expected_count_1 * 1 / 5)
    self.assertLess(expected_count_1 - 3 * ci_radius_1, index_1_count)
    self.assertLess(index_1_count, expected_count_1 + 3 * ci_radius_1)

  def test_sample_fundamental_simplex_increasing_coordinates(self):
    d = 10
    num_samples = 1000
    samples = [
        sum_mechanism.sample_fundamental_simplex(d)
        for _ in range(num_samples)
    ]
    for sample in samples:
      # The fundamental simplex consists of points in (0,1)^d with increasing
      # coordinate values.
      self.assertTrue(np.all(sample[:-1] < sample[1:]))

  def test_sample_sum_ball_norms(self):
    d = 4
    k = 2
    eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
    num_samples = 1000
    samples = np.asarray([
        sum_mechanism.sample_sum_ball(eulerian_numbers, k)
        for _ in range(num_samples)
    ])
    np.testing.assert_array_less(np.linalg.norm(samples, axis=1, ord=1),
                                 np.ones(num_samples) * k)
    np.testing.assert_array_less(np.linalg.norm(samples, axis=1, ord=np.inf),
                                 np.ones(num_samples))

if __name__ == '__main__':
  absltest.main()

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

from absl.testing import absltest
import numpy as np

from dp_ripple import ripple_vote

# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name


class RippleVoteTest(absltest.TestCase):

  def test_compute_labeled_forest_table(self):
    F = ripple_vote._compute_labeled_forest_table(5)
    np.testing.assert_array_equal(
        F,
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 3.0, 1.0, 0.0, 0.0],
            [0.0, 16.0, 15.0, 6.0, 1.0, 0.0],
            [0.0, 125.0, 110.0, 45.0, 10.0, 1.0],
        ]),
    )

  def test_prufer_to_tree(self):
    prufer_sequence = [1, 3, 2, 4, 3]
    tree_edges = ripple_vote._prufer_to_tree(prufer_sequence)
    assert tree_edges == [(0, 1), (1, 3), (2, 5), (2, 4), (3, 4), (3, 6)]

  def test_compute_P_n_leq_for_vote(self):
    d = 2
    n = 4
    F = ripple_vote._compute_labeled_forest_table(d)
    assert ripple_vote._compute_P_n_leq_size(d, n, F) == 25

    d = 2
    n = 3
    assert ripple_vote._compute_P_n_leq_size(d, n, F) == 16

    d = 2
    n = 2
    assert ripple_vote._compute_P_n_leq_size(d, n, F) == 9

    d = 2
    n = 1
    assert ripple_vote._compute_P_n_leq_size(d, n, F) == 4

    d = 2
    n = 0
    assert ripple_vote._compute_P_n_leq_size(d, n, F) == 1

  def test_compute_P_n(self):
    d = 3
    n = 0
    F = ripple_vote._compute_labeled_forest_table(d)
    assert ripple_vote._compute_P_n_size(d, n, F) == 1

    d = 3
    n = 1
    assert ripple_vote._compute_P_n_size(d, n, F) == 14

    d = 2
    n = 4
    assert ripple_vote._compute_P_n_size(d, n, F) == 16

    d = 2
    n = 3
    assert ripple_vote._compute_P_n_size(d, n, F) == 12

    d = 2
    n = 2
    assert ripple_vote._compute_P_n_size(d, n, F) == 8

    d = 2
    n = 1
    assert ripple_vote._compute_P_n_size(d, n, F) == 4

    d = 2
    n = 0
    assert ripple_vote._compute_P_n_size(d, n, F) == 1

  def test_sample_point_from_P_n_is_uniform(self):
    d = 2
    n = 3
    F = ripple_vote._compute_labeled_forest_table(d)
    freq = {}
    num_points = 12000
    for _ in range(num_points):
      point = ripple_vote._sample_point_from_P_n(d, n, F)
      if str(point) not in freq:
        freq[str(point)] = 0
      freq[str(point)] += 1
    size_P_n = ripple_vote._compute_P_n_size(d, n, F)
    assert len(freq.keys()) == size_P_n
    prob_of_a_point = 1 / size_P_n
    # We approximate the expected number of points using a binomial distribution
    expected_std = (num_points * prob_of_a_point * (1 - prob_of_a_point)) ** 0.5
    for key in freq:
      np.testing.assert_allclose(freq[key], 1000, atol=4 * expected_std)

  def test_w_n_appoximately_sum_to_1(self):
    d = 5
    max_n = 100
    eps = 1
    eulerian_numbers = ripple_vote._compute_eulerian_numbers(d)
    F = ripple_vote._compute_labeled_forest_table(d)
    normalizing_constant = ripple_vote._compute_normalizing_constant(
        d, eps, F, eulerian_numbers
    )
    first_max_n_ws = [
        ripple_vote._compute_w_n(d, n, eps, normalizing_constant, F)
        for n in range(max_n)
    ]
    np.testing.assert_allclose(sum(first_max_n_ws), 1, atol=1e-4)
    return


if __name__ == "__main__":
  absltest.main()

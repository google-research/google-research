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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dp_posets import random_poset


class RandomPosetTest(parameterized.TestCase):

  @parameterized.parameters(
      ([{1}, {2}, {}], np.asarray([[1, 1, 0], [0, 1, 1], [0, 0, 1]])),
      ([{1, 2}, {}, {}], np.asarray([[1, 1, 1], [0, 1, 0], [0, 0, 1]])),
  )
  def test_get_matrix_from_adj(self, adj, expected_result):
    result = random_poset.get_matrix_from_adj(adj)
    for coordinate, value in np.ndenumerate(result):
      self.assertEqual(value, expected_result[coordinate])

  @parameterized.parameters(
      (
          [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
          np.asarray([[1, 1, 1], [0, 1, 1], [0, 0, 1]]),
      ),
      (
          [[1, 1, 1], [0, 1, 0], [0, 0, 1]],
          np.asarray([[1, 1, 1], [0, 1, 0], [0, 0, 1]]),
      ),
  )
  def test_get_transitive_closure_from_adj_matrix(
      self, adj_matrix, expected_result
  ):
    result = random_poset.get_transitive_closure_from_adj_matrix(adj_matrix)
    for coordinate, value in np.ndenumerate(result):
      self.assertEqual(value, expected_result[coordinate])

  @parameterized.parameters(
      ([{1}, {2}, {}], np.asarray([[1, 0, 0], [1, 1, 0], [1, 1, 1]])),
      ([{1, 2}, {}, {}], np.asarray([[1, 0, 0], [1, 1, 0], [1, 0, 1]])),
  )
  def test_get_order_from_adj(self, adj, expected_result):
    result = random_poset.get_order_from_adj(adj)
    for coordinate, value in np.ndenumerate(result):
      self.assertEqual(value, expected_result[coordinate])

  @parameterized.parameters(([{1}, {2}, {}], 3), ([{1, 2}, {}, {}], 2))
  def test_find_depth_of_graph(self, adj, expected_result):
    self.assertEqual(
        random_poset.find_depth_of_graph(adj), expected_result
    )

  @parameterized.parameters(([{1}, {2}, {}], 2), ([{}, {}, {}], 0))
  def test_find_number_of_edges_of_graph(self, adj, expected_result):
    self.assertEqual(
        random_poset.find_number_of_edges_of_graph(adj),
        expected_result,
    )

  @parameterized.parameters(
      ([{1}, {2}, {0}], True), ([{1, 2}, {}, {}], False), ([{0}, {}, {}], True)
  )
  def test_is_cyclic(self, adj, expected_result):
    self.assertEqual(random_poset.is_cyclic(adj), expected_result)

  def test_uniformity(self):
    """Test if generate_random_dag is truly a uniform sampler."""
    adj_to_count = {}
    num_samples = 100000
    num_size_three_dags = 25
    for _ in range(num_samples):
      adj = random_poset.generate_random_dag(3)
      sorted_adj = [sorted(neighbor_list) for neighbor_list in adj]
      if str(sorted_adj) not in adj_to_count:
        adj_to_count[str(sorted_adj)] = 1
      else:
        adj_to_count[str(sorted_adj)] += 1

    # Use 0.99 confidence interval with normal approximation
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval.
    z_score = 2.576
    expected_count = num_samples / num_size_three_dags
    probability = 1 / num_size_three_dags
    # Standard error of the bernoulli trial parameter is sqrt(p(1-p)/n), we
    # multiply by n to get error for the expected count.
    standard_error = np.sqrt(expected_count * (1 - probability))
    margin_of_error = z_score * standard_error

    with self.subTest("all outcomes are present"):
      self.assertLen(set(adj_to_count.keys()), num_size_three_dags)

    for adj, count in adj_to_count.items():
      with self.subTest(f"Count {adj} is within margin of error"):
        self.assertGreater(expected_count, count - margin_of_error)
        self.assertLess(expected_count, count + margin_of_error)


if __name__ == "__main__":
  absltest.main()

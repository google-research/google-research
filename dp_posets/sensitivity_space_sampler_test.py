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

import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import scipy

from dp_posets import sensitivity_space_sampler


class SensitivitySpaceSamplerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Corresponds to graph 2->1, 5->3, 5->4
    self.ground_set_depth_one = [1, 2, 3, 4, 5]
    self.order_depth_one = np.array([
        [1, 1, 0, 0, 0],
        [-1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, -1, -1, 1],
    ])

    # Corresponds to graph 2->1->0, 5->3, 5->4
    self.ground_set_depth_two = [0, 1, 2, 3, 4, 5]
    self.order_depth_two = np.array([
        [1, 1, 1, 0, 0, 0],
        [-1, 1, 1, 0, 0, 0],
        [-1, -1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, -1, -1, 1],
    ])

  @parameterized.named_parameters(
      ("no_maximal_element", np.array([[1, 0], [0, 1]]), 0),
      ("one_maximal_element", np.array([[1, 1], [-1, 1]]), 1),
      (
          "two_maximal_elements",
          np.array([[1, 1, 1], [-1, 1, 0], [-1, 0, 1]]),
          1,
      ),
  )
  def test_get_maximal_element_index(self, order, result):
    self.assertEqual(
        sensitivity_space_sampler.get_maximal_element_index(order), result
    )

  @parameterized.parameters((5, [4], 0), (5, [2], -1), (5, [4, 2], 0))
  def test_get_max_inf_bound_on_poset_depth_one(
      self, maximal, extended_subposet, expected_result
  ):
    result = sensitivity_space_sampler.get_max_inf_bound(
        extended_subposet=extended_subposet,
        target_element=maximal,
        order=self.order_depth_one,
        ground_set=self.ground_set_depth_one,
    )
    self.assertEqual(result, expected_result)

  @parameterized.parameters((2, [0, 1], 1), (2, [0, 4], 0), (2, [4, 5, 1], 2))
  def test_get_max_inf_bound_on_poset_depth_two(
      self, maximal, extended_subposet, expected_result
  ):

    result = sensitivity_space_sampler.get_max_inf_bound(
        extended_subposet=extended_subposet,
        target_element=maximal,
        order=self.order_depth_two,
        ground_set=self.ground_set_depth_two,
    )
    self.assertEqual(result, expected_result)

  def test_get_bipartition_with_size_one(self):
    ground_set = [1]
    order = np.array([[1]])
    partition_a, partition_b = sensitivity_space_sampler.get_bipartition(
        ground_set, order
    )

    self.assertTrue(not partition_a or not partition_b)
    self.assertTrue(partition_a == [1] or partition_b == [1])

  def test_get_bipartition_with_size_larger_than_one(self):

    partition_a, partition_b = sensitivity_space_sampler.get_bipartition(
        self.ground_set_depth_one, self.order_depth_one
    )

    with self.subTest("Union of partitions is the poset"):
      self.assertEqual(
          set(partition_a + partition_b), set(self.ground_set_depth_one)
      )

    with self.subTest("Partitions are disjoint"):
      self.assertEqual(set(partition_a).intersection(set(partition_b)), set())

  def test_get_bipartition_is_approximately_uniform(self):
    np.random.seed(0)

    ground_set = [0, 1]
    order = np.array([[1, 0], [0, 1]])

    partitions = [
        ((), (0, 1)),
        ((), (1, 0)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((0, 1), ()),
        ((1, 0), ()),
    ]

    # Use 0.99 confidence interval with normal approximation
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval.
    z_score = 2.576
    num_samples = 10000
    expected_count = num_samples / len(partitions)
    probability = 1 / len(partitions)
    # Standard error of the bernoulli trial parameter is sqrt(p(1-p)/n), we
    # multiply by n to get error for the expected count.
    standard_error = np.sqrt(expected_count * (1 - probability))
    margin_of_error = z_score * standard_error

    results = [
        sensitivity_space_sampler.get_bipartition(ground_set, order)
        for _ in range(num_samples)
    ]

    # Convert to tuples to use Counter
    results = [
        (tuple(partition_a), tuple(partition_b))
        for partition_a, partition_b in results
    ]
    outcome_counts = collections.Counter(results)

    with self.subTest("all outcomes are present"):
      self.assertEqual(set(outcome_counts.keys()), set(partitions))

    for outcome, count in outcome_counts.items():
      with self.subTest(f"Count {outcome} is within margin of error"):
        self.assertGreater(expected_count, count - margin_of_error)
        self.assertLess(expected_count, count + margin_of_error)

  def test_get_filters_chain_from_extended_subposet(self):
    extended_subposet = [1, 3, 2]
    # Order given by 2->1, 5->3, 5->4
    expected_chain = [[], [2], [2, 3, 5], [2, 3, 5, 1]]
    chain = sensitivity_space_sampler.get_filters_chain_from_extended_subposet(
        extended_subposet,
        ground_set=self.ground_set_depth_one,
        order=self.order_depth_one,
    )
    with self.subTest("Chain has correct length"):
      self.assertLen(chain, len(expected_chain))

    for f, expected_f in zip(chain, expected_chain):
      self.assertEqual(set(f), set(expected_f))

  def test_get_vertices_from_filters_chain(self):
    ground_set = [1, 2, 3, 4, 5]
    # Assumes order 2->1, 5->3, 5->4, and bipartition [3, 2], [1, 4, 5].
    chain = [[], [2], [2, 3, 5]]

    expected_vertices = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 1]]
    )
    vertices = sensitivity_space_sampler.get_vertices_from_filters_chain(
        chain, ground_set=ground_set
    )
    for coordinate, value in np.ndenumerate(vertices):
      self.assertEqual(value, expected_vertices[coordinate])

  @parameterized.parameters(3, 4, 5)
  def test_sample_point_from_simplex(self, dimension):
    vertices = np.vstack([np.zeros(dimension), np.eye(dimension)])
    sample = sensitivity_space_sampler.sample_point_from_simplex(
        dimension, vertices
    )
    self.assertEqual(sample.shape, (dimension,))
    for coordinate in sample:
      self.assertGreaterEqual(coordinate, 0.0)
    self.assertLess(np.sum(sample), 1.0)

  def test_sample_poset_balldepth_one(self):
    sample = sensitivity_space_sampler.sample_poset_ball(
        ground_set=self.ground_set_depth_one, order=self.order_depth_one
    )
    self.assertEqual(sample.shape, (len(self.ground_set_depth_one) + 1,))

  @parameterized.parameters(2, 5, 7, 10)
  def test_sample_poset_ball_is_uniform(self, dim):
    """Tests that the sampler is uniform in the path graph order.

    Consider the poset on nodes [dim-1, ..., 2, 1] with order dim->...->1. The
    corresponding sensitivity space `S` is the parallelogram with vertices
    [1,0, ..., 0], [1, 1, 0, ... , 0], [1, 1, 1, ... , 1] and their negatives.
    The linear map `transform` maps space S to the L1-ball with vertices
    in the canonical basis, preserving uniform sampling. We use a chi-square
    test to test this hypothesis of uniform counts over 2**dim orthants.

    Args:
      dim: Dimension of the poset.
    """
    # `ground_set` is listed in reversed order to facilitate `order` and
    # `transform` matrices. Root node is `dim`, appended by the sampler in the
    # first position.
    ground_set = list(range(dim - 1, 0, -1))

    # Use the path graph order.
    order = -np.triu(np.ones((dim - 1, dim - 1)), k=1) + np.tril(
        np.ones((dim - 1, dim - 1))
    )

    # The following matrix maps the sensitivity space to an L1-ball.
    transform = np.eye(dim) - np.diag(np.ones((dim - 1,)), k=1)

    num_samples = 10000

    # We count over all orthants of the cube.
    expected_count = num_samples / 2**dim
    expected_counts = expected_count * np.ones(2**dim)

    rotated_samples = []
    for _ in range(num_samples):
      sample = sensitivity_space_sampler.sample_poset_ball(
          ground_set=ground_set, order=order
      )
      rotated_samples.append(np.matmul(transform, sample))
    rotated_samples = np.array(rotated_samples)

    # Each sequence of `true`/`false` represents an orthant of the cube.
    samples_orthant = rotated_samples > 0

    _, counts = np.unique(samples_orthant, axis=0, return_counts=True)

    l1_norms = np.linalg.norm(rotated_samples, ord=1, axis=1)

    with self.subTest("Samples are inside L1-ball"):
      for norm in l1_norms:
        self.assertLessEqual(norm, 1)

    with self.subTest("Counts include all orthants"):
      self.assertEqual(counts.shape, (2**dim,))

    with self.subTest("Chi-square test p-value is not significant"):
      _, p_value = scipy.stats.chisquare(counts, expected_counts)
      self.assertGreater(p_value, 0.05)


if __name__ == "__main__":
  absltest.main()

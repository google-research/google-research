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

"""Tests for util."""

import collections

from absl.testing import parameterized
import numpy as np
from numpy import linalg
import tensorflow as tf

from multiple_user_representations.synthetic_data import util


class UtilTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(UtilTest, self).setUp()
    np.random.seed(1234)

  def test_run_markov_chain(self):
    """Tests util.run_markov_chain."""

    test_transition_matrix = np.ones((2, 2), dtype=float)
    test_transition_matrix[0, 0] = 0.8
    test_transition_matrix[0, 1] = 0.2
    test_transition_matrix[1, 0] = 0.4
    test_transition_matrix[1, 1] = 0.6
    inital_state_prob = np.ones((2,)) * 0.5
    sequence = util.run_markov_chain(
        test_transition_matrix,
        time_steps=5000,
        initial_state_prob=inital_state_prob)
    estimated_transition_matrix = np.ones((2, 2), dtype=float)
    prev_state = -1
    for state in sequence:
      if prev_state != -1:
        estimated_transition_matrix[prev_state, state] += 1
      prev_state = state

    estimated_transition_matrix = estimated_transition_matrix / np.sum(
        estimated_transition_matrix, axis=1, keepdims=True)

    self.assertLess(
        linalg.norm(
            test_transition_matrix - estimated_transition_matrix, ord=1), 0.05)

  @parameterized.parameters((0.0), (1.0))
  def test_generate_item_sequence_from_interest_sequence(self,
                                                         item_power):
    """Tests util.generate_item_sequence_from_interest_sequence."""

    items_per_interest = 10
    num_interests = 5
    test_interest_sequence = np.random.randint(
        low=0, high=num_interests, size=(100000,), dtype=int)
    expected_probs = np.arange(1, items_per_interest + 1)**(-1.0 * item_power)
    expected_probs = np.tile(np.expand_dims(expected_probs, 1),
                             (1, num_interests)).flatten()
    expected_probs /= np.sum(expected_probs)

    item_sequence = util.generate_item_sequence_from_interest_sequence(
        test_interest_sequence, items_per_interest, item_power=item_power)

    estimated_interest_sequence = item_sequence / items_per_interest
    estimated_interest_sequence = estimated_interest_sequence.astype(int)
    item_counts = np.array(list(collections.Counter(item_sequence).values()))
    item_probs = sorted(item_counts / sum(item_counts), reverse=True)
    self.assertTrue(
        (estimated_interest_sequence == test_interest_sequence).any())
    self.assertLessEqual(np.sum(np.abs(expected_probs - item_probs)), 0.02)

  def test_extract_ground_truth_scores_for_ndcg(self):
    """Tests util.extract_ground_truth_scores_for_ndcg."""

    item_clusters = np.ones((50,))
    item_clusters[:10] = 0
    item_clusters[10:20] = 1
    item_clusters[20:50] = 2
    user_interests = np.array([[0, 2], [1, 2], [0], [0, 1, 2]])

    expected_scores = np.zeros((4, 50))
    expected_scores[0, :10] = 1
    expected_scores[0, 20:50] = 1
    expected_scores[1, 10:20] = 1
    expected_scores[1, 20:50] = 1
    expected_scores[2, :10] = 1
    expected_scores[3, :] = 1

    scores = util.extract_ground_truth_scores_for_ndcg(item_clusters,
                                                       user_interests)
    self.assertSequenceEqual(scores.shape, expected_scores.shape)
    self.assertEqual(np.sum(np.abs(expected_scores - scores)), 0.0)


if __name__ == '__main__':
  tf.test.main()

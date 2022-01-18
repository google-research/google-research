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

"""Tests for generate_synthetic_data."""

import numpy as np
from numpy import linalg
import tensorflow as tf

from multiple_user_representations.synthetic_data import generate_synthetic_data


class GenerateSyntheticDataTest(tf.test.TestCase):

  def test_generate_user_specific_markovian_data(self):
    """Tests the conditional markovian data generating process."""

    alpha = 0.7
    gamma = 0.2
    epsilon = 0.1
    num_clusters = 3
    time_steps = 10000
    data = generate_synthetic_data.generate_user_specific_markovian_data(
        num_clusters=num_clusters,
        items_per_cluster=2,
        clusters_per_user=2,
        num_data_points=2,
        time_steps=time_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon)

    user_item_seq = data['user_item_sequences']
    items = data['items']
    item_labels = np.array(data['item_clusters'], dtype=int)
    user_interests = np.array(data['user_interests'], dtype=int)
    not_interesting = np.setdiff1d(np.arange(3), user_interests[0])[0]

    expected_transition_matrix = np.ones((3, 3)) * 0.2
    np.fill_diagonal(expected_transition_matrix, 0.7)
    expected_transition_matrix[user_interests, not_interesting] = 0.1
    expected_transition_matrix[not_interesting, :] = 0.45
    expected_transition_matrix[not_interesting, not_interesting] = 0.1

    chain = user_item_seq[0]
    prev_cluster = -1
    transition_matrix = np.zeros((3, 3))
    for item in chain:
      curr_cluster = item_labels[item]
      if prev_cluster != -1:
        transition_matrix[prev_cluster, curr_cluster] += 1
      prev_cluster = curr_cluster

    transition_matrix = transition_matrix / np.sum(
        transition_matrix, axis=1, keepdims=True)

    self.assertTrue(np.array_equal(items, np.arange(6)))
    self.assertTrue(np.array_equal(item_labels, np.array([0, 0, 1, 1, 2, 2])))
    self.assertSequenceEqual(user_item_seq.shape, [2, time_steps])
    self.assertContainsSubset(user_item_seq.reshape(-1), items)
    self.assertLess(
        linalg.norm(expected_transition_matrix - transition_matrix, ord=1),
        0.05)

  def test_generate_global_markovian_data(self):
    """Tests the global markovian data generating process."""

    time_steps = 5000
    data = generate_synthetic_data.generate_global_markovian_data(
        num_clusters=3,
        items_per_cluster=2,
        num_data_points=2,
        time_steps=time_steps,
        alpha=0.9)

    user_item_seq = data['user_item_sequences']
    items = data['items']
    item_labels = np.array(data['item_clusters'], dtype=int)
    expected_transition_matrix = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05],
                                           [0.05, 0.05, 0.9]])

    chain = user_item_seq[0]
    prev_cluster = -1
    transition_matrix = np.zeros((3, 3))
    for item in chain:
      curr_cluster = item_labels[item]
      if prev_cluster != -1:
        transition_matrix[prev_cluster, curr_cluster] += 1
      prev_cluster = curr_cluster

    transition_matrix = transition_matrix / np.sum(
        transition_matrix, axis=1, keepdims=True)

    self.assertTrue(np.array_equal(items, np.arange(6)))
    self.assertTrue(np.array_equal(item_labels, np.array([0, 0, 1, 1, 2, 2])))
    self.assertSequenceEqual(user_item_seq.shape, [2, time_steps])
    self.assertContainsSubset(user_item_seq.reshape(-1), items)
    self.assertLess(
        linalg.norm(expected_transition_matrix - transition_matrix, ord=1),
        0.05)

  def test_generate_heterogeneuos_user_slices(self):
    """Tests generate_heterogeneuous_user_slices method."""

    alpha = 0.7
    gamma = 0.2
    epsilon = 0.1
    num_clusters = 5
    time_steps = 20
    items_per_cluster = 2
    data = generate_synthetic_data.generate_heterogeneuos_user_slices(
        num_clusters=num_clusters,
        items_per_cluster=items_per_cluster,
        clusters_per_user_list=[2, 3, 4],
        num_data_points=9,
        time_steps=time_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon)

    self.assertIn('user_interests_slice_1', data)
    self.assertIn('user_interests_slice_2', data)
    self.assertIn('user_interests_slice_3', data)

    user_item_seq = data['user_item_sequences']
    items = data['items']
    item_labels = np.array(data['item_clusters'], dtype=int)
    user_interests_slice_1 = data['user_interests_slice_1']
    user_interests_slice_2 = data['user_interests_slice_2']
    user_interests_slice_3 = data['user_interests_slice_3']

    self.assertLen(user_interests_slice_1, 3)
    self.assertLen(user_interests_slice_1[0], 2)
    self.assertLen(user_interests_slice_2, 3)
    self.assertLen(user_interests_slice_2[0], 3)
    self.assertLen(user_interests_slice_3, 3)
    self.assertLen(user_interests_slice_3[0], 4)
    self.assertTrue(np.array_equal(items, np.arange(10)))
    self.assertTrue(
        np.array_equal(item_labels, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])))
    self.assertSequenceEqual(user_item_seq.shape, [9, time_steps])
    self.assertContainsSubset(user_item_seq.reshape(-1), items)

  def test_generate_mixture_interest_volatility_users(self):
    """Tests generate_mixture_interest_volatility_users method."""

    data = generate_synthetic_data.generate_mixture_interest_volatility_users(
        num_clusters=5,
        clusters_per_user=2,
        items_per_cluster=2,
        num_data_points=12,
        time_steps=10,
        alpha_list=[0.9, 0.8, 0.7, 0.6],
        epsilon=0.1)

    user_item_seq = data['user_item_sequences']
    items = data['items']
    user_interests = data['user_interests']
    item_labels = np.array(data['item_clusters'], dtype=int)

    # Only testing the final size of the dataset after combining the slices. The
    # logic to generate slices is tested in
    # test_generate_user_specific_markovian_data.
    self.assertLen(user_interests, 12)
    self.assertLen(user_interests[0], 2)
    self.assertTrue(np.array_equal(items, np.arange(10)))
    self.assertTrue(
        np.array_equal(item_labels, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])))
    self.assertSequenceEqual(user_item_seq.shape, [12, 10])
    self.assertContainsSubset(user_item_seq.reshape(-1), items)


if __name__ == '__main__':
  tf.test.main()

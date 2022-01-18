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

r"""Provides utility functions for working with synthetic user-item data."""

import math
import os
from typing import Any, Dict, List

import numpy as np

from tensorflow.io import gfile


def run_markov_chain(transition_matrix,
                     initial_state_prob,
                     time_steps):
  """Runs the markov chain for a given transition matrix.

  Args:
    transition_matrix: a 2d matrix specifying the probability of transition
      across states.
    initial_state_prob: A probability vector specifying the probability of state
      at t=0.
    time_steps: Number of steps for which the markov chain will run.

  Returns:
    user_interest_sequence: A list of integers of length time_steps, specifying
      the state at each time_step.

  Raise:
    ValueError:
      - If rows of transition matrix do not sum to 1, or
      - If initial_state_prob doesn't sum to 1.
  """

  transition_matrix = np.array(transition_matrix)
  num_states = transition_matrix.shape[0]
  if not np.allclose(np.sum(transition_matrix, axis=1), np.ones(num_states,)):
    raise ValueError('Invalid transition matrix. Rows do not sum to 1.')
  elif not math.isclose(np.sum(initial_state_prob), 1.0):
    raise ValueError(
        'Initial probability distribution (Sum: {}) does not sum to 1.'.format(
            np.sum(initial_state_prob)))

  interests = np.arange(len(transition_matrix))
  user_interest_sequence = np.ones((time_steps,), dtype=int) * -1
  user_interest_sequence[0] = np.random.choice(interests, p=initial_state_prob)

  for t in range(1, time_steps):
    prev_interest = user_interest_sequence[t - 1]
    user_interest_sequence[t] = np.random.choice(
        interests, p=transition_matrix[prev_interest])

  return user_interest_sequence


def generate_item_sequence_from_interest_sequence(
    user_interest_sequence,
    items_per_interest,
    item_power = 0.0):
  """Generates a user item sequence from a list of user interests.

  Args:
    user_interest_sequence: A sequence of user interests at different timesteps.
    items_per_interest: Number of item in each interest cluster.
    item_power: The exponent for power-law distribution of items in an interest.
      If zero, the distribution will be uniform.

  Returns:
    user_item_sequence: A user item sequence corresponding to user interests.
  """

  user_item_sequence = []
  prob = np.arange(1, items_per_interest + 1)**(-1.0 * item_power)
  prob /= np.sum(prob)
  for interest in user_interest_sequence:
    user_item_sequence.append(np.random.choice(
        np.arange(items_per_interest * interest,
                  items_per_interest * (interest + 1)),
        p=prob))

  return np.array(user_item_sequence)


def load_data(data_path):
  """Loads synthetic dataset given the path to generated dataset.

  Args:
    data_path: Path to the data directory, which contains the dataset. The path
      should be the output directory used when the synthetic dataset was
      generated. When the generated synthetic data has type = `global`,
      `user_interests.npy` doesn't exist. For details see
      synthetic_data/generate_synthetic_data.py.

  Returns:
    dataset: A dictionary containing the synthetic dataset.
  """

  dataset = dict()
  filenames = [
      'items', 'user_item_sequences', 'item_clusters', 'user_interests'
  ]

  for fname in filenames:
    file_path = os.path.join(data_path, fname + '.npy')
    if not gfile.exists(file_path):
      continue

    with gfile.GFile(file_path, 'rb') as f:
      dataset[fname] = np.load(f)

  user_interests = dataset.get('user_interests', None)

  if user_interests is not None:
    item_clusters = dataset['item_clusters']
    dataset['ground_truth_scores'] = extract_ground_truth_scores_for_ndcg(
        item_clusters, user_interests)

  return dataset


def extract_ground_truth_scores_for_ndcg(item_clusters, user_interests):
  """Returns the ground truth relevance scores for synthetic dataset.

  The scores are used as the ground truth for the NDCG@K metric. For synthetic
  data, since all the items are uniformly drawn from a cluster, we set the same
  score for all items in a cluster. For instance, if a user is interested in
  clusters [1, 2], then all items in [1, 2] will have a score of 1.0, whereas
  all other items will have score of 0.0.

  Args:
    item_clusters: The cluster ids for all the items in the dataset.
    user_interests: Array of user interests.

  Returns:
    ground_truth_scores: An array of size [num_users, num_items], where each
    element depicts the relevance score of items for each user.
  """

  ground_truth_scores = np.expand_dims(item_clusters, 0)
  ground_truth_scores = np.tile(ground_truth_scores,
                                (user_interests.shape[0], 1))
  ground_truth_scores = [
      np.isin(item_clusters, user_interest)
      for (item_clusters,
           user_interest) in zip(ground_truth_scores, user_interests)
  ]
  ground_truth_scores = np.array(ground_truth_scores).astype(dtype=float)
  return ground_truth_scores

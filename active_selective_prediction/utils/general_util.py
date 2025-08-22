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

"""General Utils."""

from typing import List, Optional

import numpy as np
import tensorflow as tf


def set_random_seed(seed):
  """Sets random seed.

  Args:
    seed: a random seed
  """
  np.random.seed(seed)
  tf.random.set_seed(seed)


def outer_product_opt(
    c1,
    d1,
    c2,
    d2
):
  """Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products."""
  b1, b2 = c1.shape[0], c2.shape[0]
  t1 = np.matmul(
      np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)),
      np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1))
  )
  t2 = np.matmul(
      np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)),
      np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1))
  )
  t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
  t1 = t1.reshape(b1, 1).repeat(b2, axis=1)
  t2 = t2.reshape(1, b2).repeat(b1, axis=0)
  return t1 + t2 - 2 * t3


def kmeans_plus_plus_opt(
    x1_list,
    x2_list,
    n_clusters,
    init,
    random_state = np.random.RandomState(1234),
    n_local_trials = None
):
  """Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)."""

  idxs = np.empty((n_clusters+len(init)-1,), dtype=np.long)
  # Set the number of local seeding trials if none is given
  if n_local_trials is None:
    # This is what Arthur/Vassilvitskii tried, but did not report
    # specific results for other than mentioning in the conclusion
    # that it helped.
    n_local_trials = 2 + int(np.log(n_clusters))
  # Pick first center randomly
  idxs[:len(init)] = init
  # Initialize list of closest distances and calculate current potential
  distance_to_candidates_list = []
  for x1, x2 in zip(x1_list, x2_list):
    distance_to_candidates_list.append(outer_product_opt(
        x1[init], x2[init], x1, x2
    ).reshape(len(init), -1))
  distance_to_candidates = sum(distance_to_candidates_list)
  candidates_pot = distance_to_candidates.sum(axis=1)
  best_candidate = np.argmin(candidates_pot)
  current_pot = candidates_pot[best_candidate]
  closest_dist_sq = distance_to_candidates[best_candidate]
  # Pick the remaining n_clusters-1 points
  for c in range(len(init), len(init)+n_clusters-1):
    # Choose center candidates by sampling with probability proportional
    # to the squared distance to the closest existing center
    rand_vals = random_state.random_sample(n_local_trials) * current_pot
    candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)
    # XXX: numerical imprecision can result in a candidate_id out of range
    np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
    # Compute distances to center candidates
    distance_to_candidates_list = []
    for x1, x2 in zip(x1_list, x2_list):
      distance_to_candidates_list.append(outer_product_opt(
          x1[candidate_ids], x2[candidate_ids], x1, x2
      ).reshape(len(candidate_ids), -1))
    distance_to_candidates = sum(distance_to_candidates_list)
    # update closest distances squared and potential for each candidate
    np.minimum(
        closest_dist_sq, distance_to_candidates, out=distance_to_candidates
    )
    candidates_pot = distance_to_candidates.sum(axis=1)
    # Decide which candidate is the best
    best_candidate = np.argmin(candidates_pot)
    current_pot = candidates_pot[best_candidate]
    closest_dist_sq = distance_to_candidates[best_candidate]
    best_candidate = candidate_ids[best_candidate]
    idxs[c] = best_candidate
  return idxs[len(init)-1:]

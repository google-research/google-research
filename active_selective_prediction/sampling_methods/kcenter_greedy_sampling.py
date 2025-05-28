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

"""K-center greedy sampling method.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
"""

from typing import List

from active_selective_prediction.sampling_methods import base_sampler
from active_selective_prediction.utils import tf_util
import numpy as np
from sklearn.metrics import pairwise_distances
import tensorflow as tf


class KCenterGreedySampling(base_sampler.SamplingMethod):
  """K-center greedy sampling method."""

  def __init__(
      self,
      ensemble_models,
      n,
      target_test_ds,
      debug_info = False,
  ):
    super().__init__(n=n, debug_info=debug_info)
    self.ds = target_test_ds
    self.ensemble_models = ensemble_models

  def get_scores(
      self, already_selected_indices, label_budget
  ):
    """Gets scores of the test data for sampling."""
    if already_selected_indices.shape[0] + label_budget >= self.n:
      # Scores are useless in this case,
      # since all remaining samples will be selected.
      return np.zeros(self.n, dtype=np.float32)
    features = []
    for batch_x, _ in self.ds:
      batch_feature = tf_util.get_ensemble_model_feature(
          self.ensemble_models, batch_x
      )
      features.extend(batch_feature.numpy())
    features = np.array(features)
    cluster_centers = np.copy(already_selected_indices)
    for _ in range(label_budget):
      if cluster_centers.shape[0] == 0:
        index = np.random.choice(self.all_indices)
        cluster_centers = np.append(cluster_centers, [index])
        continue
      remain_indices = np.setdiff1d(self.all_indices, cluster_centers)
      center_features = features[cluster_centers]
      remain_features = features[remain_indices]
      dist = pairwise_distances(remain_features, center_features)
      min_dist = np.amin(dist, axis=1)
      index = np.argmax(min_dist)
      cluster_centers = np.append(cluster_centers, [remain_indices[index]])
    scores = np.ones(self.n, dtype=np.float32)
    scores[cluster_centers] = 0
    return np.array(scores)

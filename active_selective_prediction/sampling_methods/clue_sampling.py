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

"""CLUE sampling method.

Implements the Clustering Uncertainty-weighted Embeddings
(CLUE) proposed in https://arxiv.org/pdf/2010.08666.pdf.
Follows the official implementation of CLUE
in https://github.com/virajprabhu/CLUE/blob/main/sample.py#L207-L252.
"""

from typing import List

from active_selective_prediction.sampling_methods import base_sampler
from active_selective_prediction.utils import tf_util
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf


class CLUESampling(base_sampler.SamplingMethod):
  """CLUE sampling method."""

  def __init__(
      self,
      ensemble_models,
      n,
      target_test_ds,
      temperature,
      ensemble_method = 'soft',
      debug_info = False,
  ):
    super().__init__(n=n, debug_info=debug_info)
    self.ds = target_test_ds
    self.ensemble_models = ensemble_models
    self.ensemble_method = ensemble_method
    self.temperature = temperature

  def get_scores(
      self,
      already_selected_indices,
      label_budget
  ):
    """Gets scores of the test data for sampling."""
    if already_selected_indices.shape[0] + label_budget >= self.n:
      # Scores are useless in this case,
      # since all remaining samples will be selected.
      return np.zeros(self.n, dtype=np.float32)
    features = []
    outputs = []
    for batch_x, _ in self.ds:
      batch_output, batch_feature = (
          tf_util.get_ensemble_model_output_and_feature(
              self.ensemble_models,
              batch_x,
              self.ensemble_method,
              self.temperature,
          )
      )
      features.extend(batch_feature.numpy())
      outputs.extend(batch_output.numpy())
    features = np.array(features)
    outputs = np.array(outputs)
    entropy_scores = entropy(outputs, axis=1)
    remain_indices = np.setdiff1d(self.all_indices, already_selected_indices)
    remain_features = features[remain_indices]
    remain_entropy_scores = entropy_scores[remain_indices]
    # Runs weighted K-means over embeddings
    km = KMeans(label_budget)
    remain_features = np.nan_to_num(remain_features)
    remain_entropy_scores = np.nan_to_num(remain_entropy_scores)
    km.fit(remain_features, sample_weight=remain_entropy_scores)
    # Finds nearest neighbors to inferred centroids
    dists = euclidean_distances(km.cluster_centers_, remain_features)
    sort_idxs = dists.argsort(axis=1)
    q_idxs = []
    ax, rem = 0, label_budget
    while rem > 0:
      q_idxs.extend(list(sort_idxs[:, ax][:rem]))
      q_idxs = list(set(q_idxs))
      rem = label_budget-len(q_idxs)
      ax += 1
    q_idxs = np.array(q_idxs)
    scores = np.ones(self.n, dtype=np.float32)
    scores[remain_indices[q_idxs]] = 0
    return scores

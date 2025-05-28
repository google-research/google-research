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

"""BADGE sampling method.

Implements the Batch Active learning by Diverse Gradient Embeddings
(BADGE) proposed in https://arxiv.org/pdf/1906.03671.pdf.
"""

from typing import List

from active_selective_prediction.sampling_methods import base_sampler
from active_selective_prediction.utils import general_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


class BADGESampling(base_sampler.SamplingMethod):
  """BADGE sampling method."""

  def __init__(
      self,
      ensemble_models,
      n,
      target_test_ds,
      ensemble_method = 'soft',
      random_seed = 1234,
      debug_info = False,
  ):
    super().__init__(n=n, debug_info=debug_info)
    self.ds = target_test_ds
    self.ensemble_models = ensemble_models
    self.ensemble_method = ensemble_method
    self.random_seed = random_seed

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
    remain_indices = np.setdiff1d(self.all_indices, already_selected_indices)
    remain_size = remain_indices.shape[0]
    num_classes = None
    remain_feature_list = []
    uncertain_score_list = []
    for model in self.ensemble_models:
      outputs = []
      features = []
      for batch_x, _ in self.ds:
        batch_output, batch_feature = tf_util.get_model_output_and_feature(
            model,
            batch_x,
        )
        if num_classes is None:
          num_classes = batch_output.shape[1]
        outputs.extend(batch_output.numpy())
        features.extend(batch_feature.numpy())
      outputs = np.array(outputs)
      features = np.array(features)
      remain_outputs = outputs[remain_indices]
      remain_features = features[remain_indices]
      remain_preds = np.argmax(remain_outputs, axis=1)
      scores_delta = np.zeros((remain_size, num_classes), dtype=np.float32)
      scores_delta[np.arange(remain_size), remain_preds] = 1.0
      uncertain_scores = (remain_outputs - scores_delta)
      remain_feature_list.append(remain_features)
      uncertain_score_list.append(uncertain_scores)
    random_state = np.random.RandomState(self.random_seed)
    init = np.array([random_state.randint(remain_size)])
    q_idxs = general_util.kmeans_plus_plus_opt(
        x1_list=uncertain_score_list,
        x2_list=remain_feature_list,
        n_clusters=label_budget,
        init=init,
        random_state=random_state,
        n_local_trials=None,
    )
    scores = np.ones(self.n, dtype=np.float32)
    scores[remain_indices[q_idxs]] = 0
    return scores

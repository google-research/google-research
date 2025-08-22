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

"""Average KL divergence sampling method."""

from typing import List

from active_selective_prediction.sampling_methods import base_sampler
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


class AverageKLDivergenceSampling(base_sampler.SamplingMethod):
  """Average KL divergence sampling method."""

  def __init__(
      self,
      ensemble_models,
      n,
      target_test_ds,
      ensemble_method = 'soft',
      debug_info = False,
  ):
    super().__init__(n=n, debug_info=debug_info)
    self.ds = target_test_ds
    self.ensemble_models = ensemble_models
    self.ensemble_method = ensemble_method

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
    scores = []
    for batch_x, _ in self.ds:
      batch_output_list = []
      for model in self.ensemble_models:
        batch_output = tf_util.get_model_output(model, batch_x)
        batch_output_list.append(batch_output)
      batch_avg_output = sum(batch_output_list) / len(batch_output_list)
      kl_scores = 0
      for batch_output in batch_output_list:
        kl_scores += tf.keras.metrics.kl_divergence(
            batch_output, batch_avg_output
        )
      kl_scores /= len(batch_output_list)
      scores.extend(-kl_scores.numpy())
    scores = np.array(scores)
    return scores

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

"""Average margin sampling method."""

from active_selective_prediction.sampling_methods import base_sampler
import numpy as np


class AverageMarginSampling(base_sampler.SamplingMethod):
  """Average margin sampling method."""

  def __init__(
      self,
      n,
      debug_info = False,
  ):
    super().__init__(n=n, debug_info=debug_info)
    self.ensemble_outputs = None

  def update_ensemble_outputs(self, outputs):
    """Updates ensemble outputs."""
    self.ensemble_outputs = outputs

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
    if self.ensemble_outputs is None:
      raise ValueError('Must update ensemble outputs first!')
    sorted_outputs = np.sort(self.ensemble_outputs, axis=1)
    scores = sorted_outputs[:, -1] - sorted_outputs[:, -2]
    return scores

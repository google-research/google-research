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

"""Uniform sampling method."""

from active_selective_prediction.sampling_methods import base_sampler
import numpy as np


class UniformSampling(base_sampler.SamplingMethod):
  """Uniform sampling method."""

  def __init__(self, n, debug_info = False):
    super().__init__(n=n, debug_info=debug_info)

  def get_scores(
      self, already_selected_indices, label_budget
  ):
    """Gets scores of the test data for sampling."""
    if already_selected_indices.shape[0] + label_budget >= self.n:
      # Scores are useless in this case,
      # since all remaining samples will be selected.
      return np.zeros(self.n, dtype=np.float32)
    scores = np.random.uniform(low=0.0, high=1.0, size=(self.n,))
    return scores

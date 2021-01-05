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

# python3
"""Defines eviction policies that use the model."""

import torch
from cache_replacement.policy_learning.cache import eviction_policy
from cache_replacement.policy_learning.cache_model import model


class LearnedScorer(eviction_policy.CacheLineScorer):
  """Cache line scorer that uses a learned model under the hood.

  NOTE: Must be called on consecutive memory accesses.
  """

  def __init__(self, scoring_model):
    """Constructs a LearnedScorer from a given model.

    Args:
      scoring_model (EvictionPolicyModel): the model to compute scores with.

    Returns:
      LearnedScorer
    """
    self._model = scoring_model
    self._hidden_state = None

  @classmethod
  def from_model_checkpoint(cls, model_config, model_checkpoint=None):
    """Creates scorer from a model loaded from given checkpoint and config.

    Args:
      model_config (Config): model config to use.
      model_checkpoint (str | None): path to a checkpoint for the model. Model
        uses default random initialization if no checkpoint is provided.

    Returns:
      LearnedScorer
    """
    device = "cpu"
    if torch.cuda.is_available():
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
      device = "cuda:0"

    scoring_model = model.EvictionPolicyModel.from_config(model_config).to(
        torch.device(device))

    if model_checkpoint is not None:
      with open(model_checkpoint, "rb") as f:
        scoring_model.load_state_dict(torch.load(f, map_location=device))
    return cls(scoring_model)

  def __call__(self, cache_access, access_times):
    del access_times

    scores, _, self._hidden_state, _ = self._model(
        [cache_access], self._hidden_state, inference=True)
    return {line: -scores[0, i].item() for i, (line, _) in
            enumerate(cache_access.cache_lines)}

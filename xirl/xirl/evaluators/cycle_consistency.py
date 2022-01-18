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

"""Cycle-consistency evaluator."""

import itertools
from typing import List

from .base import Evaluator
from .base import EvaluatorOutput
import numpy as np
from scipy.spatial.distance import cdist
from xirl.models import SelfSupervisedOutput


class _CycleConsistency(Evaluator):
  """Base class for cycle consistency evaluation."""

  def __init__(self, n_way, stride, distance):
    """Constructor.

    Args:
      n_way: The number of cycle-consistency ways.
      stride: Controls how many frames are skipped in each video sequence. For
        example, if the embedding vector of the first video is (100, 128), a
        stride of 5 reduces it to (20, 128).
      distance: The distance metric to use when calculating nearest-neighbours.

    Raises:
      ValueError: If the distance metric is invalid or the
        mode is invalid.
    """
    super().__init__(inter_class=False)

    assert n_way in [2, 3], "n_way must be 2 or 3."
    assert isinstance(stride, int), "stride must be an integer."
    if distance not in ["sqeuclidean", "cosine"]:
      raise ValueError(
          "{} is not a supported distance metric.".format(distance))

    self.n_way = n_way
    self.stride = stride
    self.distance = distance

  def _evaluate_two_way(self, embs):
    """Two-way cycle consistency."""
    num_embs = len(embs)
    total_combinations = num_embs * (num_embs - 1)
    ccs = np.zeros((total_combinations))
    idx = 0
    for i in range(num_embs):
      query_emb = embs[i][::self.stride]
      ground_truth = np.arange(len(embs[i]))[::self.stride]
      for j in range(num_embs):
        if i == j:
          continue
        candidate_emb = embs[j][::self.stride]
        dists = cdist(query_emb, candidate_emb, self.distance)
        nns = np.argmin(dists[:, np.argmin(dists, axis=1)], axis=0)
        ccs[idx] = np.mean(np.abs(nns - ground_truth) <= 1)
        idx += 1
    ccs = ccs[~np.isnan(ccs)]
    return EvaluatorOutput(scalar=np.mean(ccs))

  def _evaluate_three_way(self, embs):
    """Three-way cycle consistency."""
    num_embs = len(embs)
    cycles = np.stack(list(itertools.permutations(np.arange(num_embs), 3)))
    total_combinations = len(cycles)
    ccs = np.zeros((total_combinations))
    for c_idx, cycle in enumerate(cycles):
      # Forward consistency check. Each cycle will be a length 3
      # permutation, e.g. U - V - W. We compute nearest neighbours across
      # consecutive pairs in the cycle and loop back to the first cycle
      # index to obtain: U - V - W - U.
      query_emb = None
      for i in range(len(cycle)):
        if query_emb is None:
          query_emb = embs[cycle[i]][::self.stride]
        candidate_emb = embs[cycle[(i + 1) % len(cycle)]][::self.stride]
        dists = cdist(query_emb, candidate_emb, self.distance)
        nns_forward = np.argmin(dists, axis=1)
        query_emb = candidate_emb[nns_forward]
      ground_truth_forward = np.arange(len(embs[cycle[0]]))[::self.stride]
      cc_forward = np.abs(nns_forward - ground_truth_forward) <= 1
      # Backward consistency check. A backward check is equivalent to
      # reversing the middle pair V - W and performing a forward check,
      # e.g. U - W - V - U.
      cycle[1:] = cycle[1:][::-1]
      query_emb = None
      for i in range(len(cycle)):
        if query_emb is None:
          query_emb = embs[cycle[i]][::self.stride]
        candidate_emb = embs[cycle[(i + 1) % len(cycle)]][::self.stride]
        dists = cdist(query_emb, candidate_emb, self.distance)
        nns_backward = np.argmin(dists, axis=1)
        query_emb = candidate_emb[nns_backward]
      ground_truth_backward = np.arange(len(embs[cycle[0]]))[::self.stride]
      cc_backward = np.abs(nns_backward - ground_truth_backward) <= 1
      # Require consistency both ways.
      cc = np.logical_and(cc_forward, cc_backward)
      ccs[c_idx] = np.mean(cc)
    ccs = ccs[~np.isnan(ccs)]
    return EvaluatorOutput(scalar=np.mean(ccs))

  def evaluate(self, outs):
    embs = [o.embs for o in outs]
    if self.n_way == 2:
      return self._evaluate_two_way(embs)
    return self._evaluate_three_way(embs)


class TwoWayCycleConsistency(_CycleConsistency):
  """2-way cycle consistency evaluator [1].

  References:
    [1]: https://arxiv.org/abs/1805.11592
  """

  def __init__(self, stride, distance):
    super().__init__(2, stride, distance)


class ThreeWayCycleConsistency(_CycleConsistency):
  """2-way cycle consistency evaluator [1].

  References:
    [1]: https://arxiv.org/abs/1805.11592
  """

  def __init__(self, stride, distance):
    super().__init__(3, stride, distance)

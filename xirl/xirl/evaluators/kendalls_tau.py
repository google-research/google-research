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

"""Kendall rank correlation coefficient evaluator."""

from typing import List

from .base import Evaluator
from .base import EvaluatorOutput
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from xirl.models import SelfSupervisedOutput


def softmax(dists, temp = 1.0):
  dists_ = np.array(dists - np.max(dists))
  exp = np.exp(dists_ / temp)
  return exp / np.sum(exp)


class KendallsTau(Evaluator):
  """Kendall rank correlation coefficient [1].

  References:
    [1]: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
  """

  def __init__(self, stride, distance):
    """Constructor.

    Args:
      stride: Controls how many frames are skipped in each video sequence. For
        example, if the embedding vector of the first video is (100, 128), a
        stride of 5 reduces it to (20, 128).
      distance: The distance metric to use when calculating nearest-neighbours.

    Raises:
      ValueError: If the distance metric is invalid.
    """
    super().__init__(inter_class=False)

    assert isinstance(stride, int), "stride must be an integer."
    if distance not in ["sqeuclidean", "cosine"]:
      raise ValueError(
          "{} is not a supported distance metric.".format(distance))

    self.stride = stride
    self.distance = distance

  def evaluate(self, outs):
    """Get pairwise nearest-neighbours then compute KT."""
    embs = [o.embs for o in outs]
    num_embs = len(embs)
    total_combinations = num_embs * (num_embs - 1)
    taus = np.zeros((total_combinations))
    idx = 0
    img = None
    for i in range(num_embs):
      query_emb = embs[i][::self.stride]
      for j in range(num_embs):
        if i == j:
          continue
        candidate_emb = embs[j][::self.stride]
        dists = cdist(query_emb, candidate_emb, self.distance)
        if i == 0 and j == 1:
          sim_matrix = []
          for k in range(len(query_emb)):
            sim_matrix.append(softmax(-dists[k]))
          img = np.array(sim_matrix, dtype=np.float32)[Ellipsis, None]
        nns = np.argmin(dists, axis=1)
        taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
        idx += 1
    taus = taus[~np.isnan(taus)]
    if taus.size == 0:
      tau = 0.0
    else:
      tau = np.mean(taus)
    return EvaluatorOutput(scalar=tau, image=img)

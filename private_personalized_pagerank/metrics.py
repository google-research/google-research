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

"""Metrics for PPR."""

import random
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


def sort_ppr_index_descending(ppr):
  ppr_with_random = [(x, random.random(), i) for i, x in enumerate(ppr)]
  ppr_with_random.sort(reverse=True)
  return np.array([pos for _, _, pos in ppr_with_random])


def score_approximation(
    ppr_true_param,
    ppr_approx_param,
    ks = (1, 3, 5, 10, 25, 50, 100, 1000),
):
  """Output the approximation statistics."""
  # Make a copy as the function then modifies the rankings.
  ppr_true = ppr_true_param.copy()
  ppr_approx = ppr_approx_param.copy()

  assert len(ppr_true) == len(
      ppr_approx
  ), 'Length of two PPR arrays should match!'
  delta = ppr_true - ppr_approx
  results = {
      'L1': np.linalg.norm(delta, 1),
      'L2': np.linalg.norm(delta, 2),
      'L2_norm': np.linalg.norm(delta, 2) / np.linalg.norm(ppr_true),
      'SpearmanRank': spearmanr(ppr_true, ppr_approx)[0],
  }
  ppr_true_index = sort_ppr_index_descending(ppr_true)
  ppr_approx_index = sort_ppr_index_descending(ppr_approx)
  for k in ks:
    results[f'Recall@{k}'] = (
        len(np.intersect1d(ppr_true_index[:k], ppr_approx_index[:k])) / k
    )
  return results

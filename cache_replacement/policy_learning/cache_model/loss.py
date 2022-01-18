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

"""Defines custom loss functions."""

import torch


def top_1_log_likelihood(probs):
  """Log-likelihood of the top option.

  Args:
    probs (torch.FloatTensor): size (batch_size, num_options), where the options
      are sorted from highest to lowest.

  Returns:
    loss (torch.FloatTensor): tensor of shape (batch_size) [-log p(top option)]
  """
  # 1e-8 for numerical stability
  return -torch.log(probs[:, 0] + 1e-8)


def approx_ndcg(scores, relevances, alpha=10., mask=None):
  """Computes differentiable estimate of NDCG of scores as following.

  Uses the approximation framework from Qin et. al., 2008

    IDCG = sum_i (exp(rel[i]) - 1) / ln(i + 1)
    DCG = sum_i (exp(rel[i]) - 1) / ln(pos(score, i) + 1)
    pos(score, i) =
      1 + sum_{j != i} exp(-alpha s_{i, j}) / (1 + exp(-alpha s_{i, j}))
      (differentiable approximate position function)
    s_{i, j} = scores[i] - scores[j]
    NDCG loss = -DCG / IDCG

  Args:
    scores (torch.FloatTensor): tensor of shape (batch_size, num_elems).
    relevances (torch.FloatTensor): tensor of same shape as scores (rel).
    alpha (float): value to use in the approximate position function. The
      approximation becomes exact as alpha tends toward inf.
    mask (torch.ByteTensor | None): tensor of same shape as scores. Masks out
      elements at index [i][j] if mask[i][j] = 0. Defaults to no masking.

  Returns:
    ndcg (torch.FloatTensor): tensor of shape (batch_size).
  """
  def approx_positions(scores, alpha=10.):
    # s_{i, j} (batch_size, num_elems)
    diff = (scores.unsqueeze(-1).expand(-1, -1, scores.shape[1]) -
            scores.unsqueeze(1).expand(-1, scores.shape[1], -1))
    # Add 0.5 instead of 1, because s_{1, i} = 0.5 is included
    return 0.5 + torch.sigmoid(alpha * diff).sum(1)

  if mask is None:
    mask = torch.ones_like(scores)

  # +1 because indexing starts at 1 in IDCG
  idcg = torch.expm1(relevances) * mask.float() / torch.log1p(
      torch.arange(scores.shape[-1]).float() + 1)
  pos = approx_positions(scores, alpha)
  dcg = torch.expm1(relevances) * mask.float() / torch.log1p(pos)
  return -dcg.sum(-1) / (idcg.sum(-1) + 1e-8)

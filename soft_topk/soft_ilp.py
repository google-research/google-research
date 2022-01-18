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

"""Soft top-k as integer linear programming.

Used in [A Framework For Differentiable Discovery of Graph Algorithms](https://openreview.net/pdf?id=5UvvKsBTDcR).
"""

import numpy as np
import torch
import torch.nn as nn


def argmin_khot(score_mat, k):
  with torch.no_grad():
    if k > score_mat.shape[1]:
      k = score_mat.shape[1]
    _, topk_idx = torch.topk(score_mat, k, largest=False, dim=1)
    out_mask = torch.zeros(score_mat.shape).to(score_mat.device)
    ones = torch.ones(topk_idx.shape).to(score_mat.device)
    out_mask.scatter_(1, topk_idx, ones)
  return out_mask


class ColumnTopkFunc(torch.autograd.Function):

  @staticmethod
  def forward(ctx, score_mat, k, noise):
    ctx.k = k
    ctx.noise = noise
    khot = argmin_khot(-score_mat, k)
    ctx.save_for_backward(score_mat, khot)
    return khot

  @staticmethod
  def backward(ctx, grad_output):
    score_mat, ctx_khot = ctx.saved_tensors
    k, noise = ctx.k, ctx.noise
    with torch.no_grad():
      score_perturb = -score_mat + grad_output * noise
      new_khot = argmin_khot(score_perturb, k)
      grad = (ctx_khot - new_khot) / noise
      return grad, None, None


def soft_topk(score_mat, k, noise):
  if k < 0 or k > score_mat.shape[1]:
    with torch.no_grad():
      return score_mat * 0 + 1.0
  if k == 0:
    with torch.no_grad():
      return score_mat * 0
  return ColumnTopkFunc.apply(score_mat, k, noise)


if __name__ == '__main__':
  score = torch.randn(2, 4)
  print(score)
  arg_topk = soft_topk(score, 2, noise=100)
  print(arg_topk)

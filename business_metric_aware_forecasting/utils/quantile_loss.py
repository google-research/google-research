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

"""Quantile loss module."""

import math
import torch
from torch import nn
from utils.eval_utils import get_ragged_mean


class QuantileLoss(nn.Module):
  """Computes quantile loss in a differentiable manner."""

  def __init__(self, init_quantile, frozen=False):
    super().__init__()
    if frozen:
      self.quantile = torch.Tensor([init_quantile])
    else:
      init_base_param = -math.log((1./init_quantile) - 1.)
      self.base_param = torch.nn.Parameter(torch.Tensor([init_base_param]))
      self.base_param.requires_grad = True
      self.quantile = None

  def get_quantile(self):
    if self.quantile is None:
      q = torch.sigmoid(self.base_param)
    else:
      q = self.quantile
    return q

  def forward(self, preds, targets, forecast_horizon_lengths, time_lengths):
    relu = nn.ReLU()
    q = self.get_quantile()
    error = targets - preds
    loss = (relu(error) * q) + (relu(-error) * (1 - q))
    # get average along forecasting horizon
    loss = get_ragged_mean(loss, lens=forecast_horizon_lengths, axis=-1)
    # get average along time axis
    loss = get_ragged_mean(loss, lens=time_lengths, axis=-1)
    return loss

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

"""Reinversible instance normalization."""
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RevIN(nn.Module):
  """Reinversible instance normalization.

  Attributes:
    num_features: the number of features or channels
    eps: a value added for numerical stability
    axis: axis to be normalized
    affine: if True, RevIN has learnable affine parameters
  """

  def __init__(self, num_features, eps=1e-5, affine=False, axis=1):
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.axis = axis
    self.affine = affine
    if self.affine:
      self.init_params()

  def forward(self, x, mode):
    """Apply normalization or inverse normalization.

    Args:
      x: input ts
      mode: normalize or denormalize

    Returns:
      nomarlized or denormalized x

    Raises:
      NotImplementedError:
    """

    if mode == "norm":
      self.get_statistics(x)
      x = self.normalize(x)
    elif mode == "denorm":
      x = self.denormalize(x)
    else:
      raise NotImplementedError
    return x

  def init_params(self):
    # initialize RevIN params:
    self.affine_weight = nn.Parameter(torch.ones(1, 1,
                                                 self.num_features)).to(device)
    self.affine_bias = nn.Parameter(torch.zeros(1, 1,
                                                self.num_features)).to(device)

  def get_statistics(self, x):
    self.mean = torch.mean(x, dim=self.axis, keepdim=True)
    self.stdev = torch.sqrt(
        torch.std(x, dim=self.axis, keepdim=True) + self.eps)

  def normalize(self, x):
    x = x - self.mean
    x = x / self.stdev
    if self.affine:
      x = x * self.affine_weight
      x = x + self.affine_bias
    return x

  def denormalize(self, x):
    if self.affine:
      x = x - self.affine_bias
      x = x / (self.affine_weight + self.eps * self.eps)
    x = x * self.stdev
    x = x + self.mean
    return x

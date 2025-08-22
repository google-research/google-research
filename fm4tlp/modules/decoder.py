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

"""Decoder modules for dynamic link prediction"""

import torch
from torch import nn
from torch.nn import functional


class LinkPredictor(torch.nn.Module):
  """Reference:

  - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
  """

  def __init__(self, in_channels):
    super().__init__()
    self.lin_src = nn.Linear(in_channels, in_channels)
    self.lin_dst = nn.Linear(in_channels, in_channels)
    self.lin_final = nn.Linear(in_channels, 1)

  def forward(self, z_src, z_dst):
    h = self.lin_src(z_src) + self.lin_dst(z_dst)
    h = h.relu()
    return self.lin_final(h).sigmoid()


class NodePredictor(torch.nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.lin_node = nn.Linear(in_dim, in_dim)
    self.out = nn.Linear(in_dim, out_dim)

  def forward(self, node_embed):
    h = self.lin_node(node_embed)
    h = h.relu()
    h = self.out(h)
    # h = functional.F.log_softmax(h, dim=-1)
    return h

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

"""Time Encoding Module

Reference:
    -
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
from torch import nn


class TimeEncoder(torch.nn.Module):

  def __init__(self, out_channels):
    super().__init__()
    self.out_channels = out_channels
    self.lin = nn.Linear(1, out_channels)

  def reset_parameters(self):
    self.lin.reset_parameters()

  def forward(self, t):
    return self.lin(t.view(-1, 1)).cos()

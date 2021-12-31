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

"""Backbone subnetwork based on ResNet-20 and ResNet-56."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights


class BasicBlock(nn.Module):
  """Basic block of ResNet with filters gated."""

  def __init__(self, in_channels, out_channels, stride, gate_idx):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
    self.conv2 = nn.Conv2d(
        out_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if in_channels != out_channels:
      self.shortcut.add_module(
          'conv',
          nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=1,
              stride=stride,
              padding=0,
              bias=False))
      self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    self.gate_idx = gate_idx

  def forward(self, x_gate):
    x, gate = x_gate
    y = F.relu(self.bn1(self.conv1(x)), inplace=True)
    y = self.bn2(self.conv2(y))

    # Get gate for the current block,
    # shape: [batch size, number of filters to be gated in current block].
    gate = gate[:, self.gate_idx : self.gate_idx+y.size()[1]]

    # Apply gate, which is broadcasted along the spacial axes.
    y = y * gate.unsqueeze(dim=2).unsqueeze(dim=3)

    y += self.shortcut(x)
    y = F.relu(y, inplace=True)
    return [y, x_gate[1]]


class Network(nn.Module):
  """Backbone network based on ResNet."""

  def __init__(self, depth=20, num_classes=10):
    super(Network, self).__init__()
    input_shape = [1, 3, 32, 32]

    base_channels = 16
    num_blocks_per_stage = (depth - 2) // 6

    n_channels = [base_channels, base_channels * 2, base_channels * 4]

    self.conv = nn.Conv2d(
        input_shape[1],
        n_channels[0],
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False)
    self.bn = nn.BatchNorm2d(base_channels)

    # gate_idx = [0, 16, 32, 48, 48+32, 48+32*2, 48+32*3, 144+64, 144+64*2].
    gate_idx = 0
    self.stage1, gate_idx = self._make_stage(
        n_channels[0], n_channels[0],
        num_blocks_per_stage, stride=1, gate_idx=gate_idx)
    self.stage2, gate_idx = self._make_stage(
        n_channels[0], n_channels[1],
        num_blocks_per_stage, stride=2, gate_idx=gate_idx)
    self.stage3, gate_idx = self._make_stage(
        n_channels[1], n_channels[2],
        num_blocks_per_stage, stride=2, gate_idx=gate_idx)
    self.gate_size = gate_idx
    print('Total number of gates needed: {}\n'.format(self.gate_size))

    # Get feature size.
    with torch.no_grad():
      self.feature_size = self._get_conv_features(
          torch.ones(*input_shape),
          gate=torch.ones(1, self.gate_size)).view(-1).shape[0]

    self.fc = nn.Linear(self.feature_size, num_classes)

    # Initialize weights.
    self.apply(init_weights)

  def _make_stage(self, in_channels, out_channels, num_blocks,
                  stride, gate_idx):
    s = nn.Sequential()
    for i in range(num_blocks):
      name = 'block{}'.format(i + 1)
      if i == 0:
        s.add_module(name,
                     BasicBlock(in_channels,
                                out_channels,
                                stride=stride,
                                gate_idx=gate_idx))
      else:
        s.add_module(name,
                     BasicBlock(out_channels,
                                out_channels,
                                stride=1,
                                gate_idx=gate_idx))
      gate_idx += out_channels
    return s, gate_idx

  def _get_conv_features(self, x, gate):
    y = F.relu(self.bn(self.conv(x)), inplace=True)
    y, _ = self.stage1([y, gate])
    y, _ = self.stage2([y, gate])
    y, _ = self.stage3([y, gate])
    y = F.adaptive_avg_pool2d(y, output_size=1)
    return y

  def forward(self, x, gate):
    y = self._get_conv_features(x, gate)
    y = y.view(y.size(0), -1)
    y = self.fc(y)
    return y

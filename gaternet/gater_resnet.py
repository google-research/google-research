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

"""Gater subnetwork based on ResNet-20."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights


class FNN(nn.Module):
  """Fully connected feedforward neural network."""

  def __init__(self, dims):
    super(FNN, self).__init__()
    assert len(dims) >= 2, 'Length of dims is smaller than 2.'

    self.fc = []
    for i in range(len(dims) - 2):
      self.fc.append(nn.Linear(dims[i], dims[i+1]))
      self.fc.append(nn.ReLU(inplace=True))
    self.fc.append(nn.Linear(dims[-2], dims[-1]))

    self.fc = nn.Sequential(*self.fc)

  def forward(self, x):
    return self.fc(x)


class BasicBlock(nn.Module):
  """Basic block of ResNet."""

  def __init__(self, in_channels, out_channels, stride):
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

  def forward(self, x):
    y = F.relu(self.bn1(self.conv1(x)), inplace=True)
    y = self.bn2(self.conv2(y))
    y += self.shortcut(x)
    y = F.relu(y, inplace=True)
    return y


class Network(nn.Module):
  """ResNet model."""

  def __init__(self, bottleneck_size, depth=20):
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

    self.stage1 = self._make_stage(
        n_channels[0], n_channels[0], num_blocks_per_stage, stride=1)
    self.stage2 = self._make_stage(
        n_channels[0], n_channels[1], num_blocks_per_stage, stride=2)
    self.stage3 = self._make_stage(
        n_channels[1], n_channels[2], num_blocks_per_stage, stride=2)

    # Get feature size.
    with torch.no_grad():
      self.feature_size = self._get_conv_features(
          torch.ones(*input_shape)).view(-1).shape[0]

    self.newfc = nn.Linear(self.feature_size, bottleneck_size)

    # Initialize weights.
    self.apply(init_weights)

  def _make_stage(self, in_channels, out_channels, num_blocks, stride):
    s = nn.Sequential()
    for i in range(num_blocks):
      name = 'block{}'.format(i + 1)
      if i == 0:
        s.add_module(name,
                     BasicBlock(in_channels,
                                out_channels,
                                stride=stride))
      else:
        s.add_module(name,
                     BasicBlock(out_channels,
                                out_channels,
                                stride=1))
    return s

  def _get_conv_features(self, x):
    y = F.relu(self.bn(self.conv(x)), inplace=True)
    y = self.stage1(y)
    y = self.stage2(y)
    y = self.stage3(y)
    y = F.adaptive_avg_pool2d(y, output_size=1)
    return y

  def forward(self, x):
    y = self._get_conv_features(x)
    y = y.view(y.size(0), -1)
    y = self.newfc(y)
    return y


class StepFunction(torch.autograd.Function):
  """Step function for gate discretization."""

  @staticmethod
  def forward(ctx, x, theshold=0.49999):
    return (x > theshold).float()

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clone(), None


class Gater(nn.Module):
  """Gater which generates gates from input images."""

  def __init__(self, depth, bottleneck_size, gate_size):
    super(Gater, self).__init__()

    self.resnet = Network(bottleneck_size, depth=depth)
    self.fnn = FNN((bottleneck_size, gate_size))
    # Initialize weights.
    with torch.no_grad():
      self.apply(init_weights)

  def forward(self, x):
    preact = F.relu(self.resnet(x), inplace=True)
    preact = self.fnn(preact)

    if self.training:
      preact += torch.randn_like(preact)
    gate = F.sigmoid(preact)
    gate = torch.clamp(1.2*gate-0.1, min=0, max=1)
    discrete_gate = StepFunction.apply(gate)

    if self.training:
      discrete_prob = 0.5
      mix_mask = gate.new_empty(
          size=[gate.size()[0], 1]).uniform_() < discrete_prob
      gate = torch.where(mix_mask, discrete_gate, gate)
    else:
      gate = discrete_gate

    return gate

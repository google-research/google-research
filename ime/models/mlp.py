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

"""Class that implement MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
  """An implementation  of MLP."""

  def __init__(self, input_sz, output_sz, d_model, n_layers):
    """Initializes a MLP instance.

    Args:
      input_sz: Input features dimension
      output_sz:  Dimension of output
      d_model: Hidden layer dimension
      n_layers: Number of MLP layers.
    Inputs:
      x: A tensor of shape `(batch_size, input_sz)`

    Returns:
      out: Output tensor of shape `(batch_size, output_sz)`
    """
    super(MLP, self).__init__()
    self.linear_first = torch.nn.Linear(input_sz, d_model)
    self.linear_first.bias.data.fill_(0)
    self.linear_second = torch.nn.Linear(d_model, output_sz)
    self.linear_second.bias.data.fill_(0)
    self.linear_second.weight.data.fill_(0)

    layers = []
    for _ in range(n_layers):
      layers.append(nn.Linear(d_model, d_model))
      layers.append(nn.ReLU())
      layers.append(nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1))

    self.layers = nn.Sequential(*layers)

  def forward(self, x):

    out = self.linear_first(x)
    out = self.layers(out)
    out = self.linear_second(F.relu(out))
    return out.squeeze()


class MLPime(nn.Module):
  """An implementation  of MLP.

     This MLP is used by interpretable mixture of expert as assigner module.
  """

  def __init__(self, input_sz, output_sz, num_experts, d_model, n_layers):
    """Initializes a  MLPime instance.

    Args:
      input_sz: Input features dimension
      output_sz:  Dimension of output
      num_experts: Number of experts
      d_model: Hidden layer dimension
      n_layers: Number of MLP layers.
    Inputs:
      x: A tensor of shape `(batch_size, input_sz)`

    Returns:
      output: Output used for expert classification a tensor of shape
      `(batch_size, num_experts)`
      reg_out: Regression output the prediction a tensor of
              `(batch_size, output_sz)`
    """
    super(MLPime, self).__init__()

    self.linear_first = torch.nn.Linear(input_sz, d_model)
    self.linear_first.bias.data.fill_(0)

    self.linear_second = torch.nn.Linear(d_model, num_experts)
    self.linear_second.bias.data.fill_(0)
    self.linear_second.weight.data.fill_(0)

    self.linear_third = torch.nn.Linear(d_model, output_sz)
    self.linear_third.bias.data.fill_(0)
    self.linear_third.weight.data.fill_(0)

    layers = []
    for _ in range(n_layers):
      layers.append(nn.Linear(d_model, d_model))
      layers.append(nn.ReLU())
      layers.append(nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1))

    self.layers = nn.Sequential(*layers)

  def forward(self, x):

    out = self.linear_first(x)
    output = self.layers(out)

    out = self.linear_second(F.relu(output))

    reg_out = self.linear_third(F.relu(output))

    return out, reg_out

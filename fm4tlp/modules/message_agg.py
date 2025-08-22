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

"""Message Aggregator Module

Reference:
    -
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
import torch_geometric
import torch_scatter


class LastAggregator(torch.nn.Module):
  """Returns last message from an input messages."""

  def forward(
      self,
      msg,
      index,
      t,
      dim_size,
  ):
    _, argmax = torch_scatter.scatter_max(t, index, dim=0, dim_size=dim_size)
    out = msg.new_zeros((dim_size, msg.size(-1)))
    mask = argmax < msg.size(0)  # Filter items with at least one entry.
    out[mask] = msg[argmax[mask]]
    return out


class MeanAggregator(torch.nn.Module):
  """Returns mean message from an input messages."""

  def forward(
      self,
      msg,
      index,
      t,
      dim_size,
  ):
    return torch_geometric.utils.scatter(
        msg, index, dim=0, dim_size=dim_size, reduce="mean"
    )

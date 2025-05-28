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

"""GNN-based modules used in the architecture of MP-TG models"""

import math
import torch
import torch_geometric


class GraphAttentionEmbedding(torch.nn.Module):
  """Reference:

  - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
  """

  def __init__(self, in_channels, out_channels, msg_dim, time_enc):
    super().__init__()
    self.time_enc = time_enc
    edge_dim = msg_dim + time_enc.out_channels
    self.conv = torch_geometric.nn.TransformerConv(
        in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
    )

  def forward(self, x, last_update, edge_index, t, msg):
    rel_t = last_update[edge_index[0]] - t
    rel_t_enc = self.time_enc(rel_t.to(x.dtype))
    edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
    return self.conv(x, edge_index, edge_attr)


class TimeEmbedding(torch.nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    class NormalLinear(torch.nn.Linear):
      # From TGN code: From JODIE code
      def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.out_channels)

  def forward(self, x, last_update, t):
    rel_t = t - last_update
    embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

    return embeddings

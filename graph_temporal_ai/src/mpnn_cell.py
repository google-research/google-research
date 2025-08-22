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

"""Message Passing Neural Network cell with message passing propagation rules."""

import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2SeqAttrs:
  """Initialize the hyperparameters from the arguments.

    Attributes:
  """

  def __init__(self, args):
    self.input_dim = args.input_dim
    self.output_dim = args.output_dim
    self.rnn_units = args.hidden_dim
    self.num_nodes = args.num_nodes
    self.input_len = args.input_len
    self.output_len = args.output_len

    self.max_diffusion_step = args.max_diffusion_step
    self.filter_type = args.filter_type
    self.use_gc_ru = args.use_gc_ru
    self.use_gc_c = args.use_gc_c
    self.dropout = args.dropout

    self.batch_size = args.batch_size
    self.num_rnn_layers = args.num_layers
    self.lr = args.learning_rate
    self.activation = args.activation


class MPNNCell(MessagePassing):
  """Message Passing Neural Net cell.

  Attributes:
    edge_index: adjacency list of edges [2 x num_edges]
    edge_attr: attributes [num_edges x num_feature]
  """

  def __init__(self, edge_index, edge_attr, args):
    super().__init__(aggr="add", node_dim=-2)

    Seq2SeqAttrs.__init__(self, args)
    self.edge_index = edge_index
    self.edge_attr = edge_attr
    # compute normalization
    row, col = self.edge_index
    deg = degree(col, self.num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    self.norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    self.activation = torch.tanh

    # concate r and u state together
    input_size = 2 * self.rnn_units

    self.ru_layer = torch.nn.Linear(input_size, 2 * self.rnn_units)
    self.c_layer = torch.nn.Linear(input_size, self.rnn_units)

  def forward(self, inputs, hx, edge_index, edge_attr):
    """Update GRU/LSTM hidden states.

    Args:
      inputs: input tensor
      hx: hidden state tensor
      edge_index: adjacency list of edges [2 x num_edges]
      edge_attr: attributes [num_edges x num_feature]

    Returns:
      out: updated hidden state
    """

    if self.use_gc_ru:
      ru_opt = self._mp
    else:
      ru_opt = self._fc

    value = ru_opt(inputs, hx, is_c_opt=False)
    value = torch.tanh(value)

    # value = torch.reshape(value, (-1, self._num_nodes, output_size))
    r, u = torch.split(
        tensor=value, split_size_or_sections=self.rnn_units, dim=-1)

    del value
    if self.use_gc_c:
      c_opt = self._mp
    else:
      c_opt = self._fc

    c = c_opt(inputs, r * hx, is_c_opt=True)
    new_state = u * hx + (1.0 - u) * c

    del u
    del hx
    del c
    return new_state

  def _mp(self, inputs, state, is_c_opt=False):
    """Perform message passing operaton.

    Args:
        inputs (tensor): [batch, num_nodes, input_dim]
        state (tensor): [batch, num_nodes, hidden_dim]
        is_c_opt:

    Returns:
        gconv (tensor): [batch, num_nodes, output_dim]
    """
    x = torch.cat([inputs, state], dim=-1)  # input_dim

    if is_c_opt:
      x = self.c_layer(x)
    else:
      x = self.ru_layer(x)

    # torch.geometric does not support batch processing

    out = []
    for b in range(self.batch_size):
      out_b = self.propagate(self.edge_index, x=x[b, :], norm=self.norm)
      out.append(out_b)
    out = torch.stack(out, dim=0)
    return out

  def message(self, x_j, norm):
    # x_j has shape [E, out_channels]

    # Step 4: Normalize node features.
    msg = norm.view(-1, 1) * x_j
    return msg

  def _fc(self, inputs, state, is_c_opt=False):
    """Embed with fully connected layer.

    Args:
      inputs(tensor): [batch, num_nodes, input_dim]
      state(tensor): [batch, num_nodes, hidden_dim]
      is_c_opt:

    Returns:
      output(tensor): [batch, num_nodes, output_dim]
    """
    batch_size = self.batch_size
    num_nodes = self.num_nodes

    inputs0 = torch.reshape(inputs, (batch_size * num_nodes, -1))
    state0 = torch.reshape(state, (batch_size * num_nodes, -1))

    x = torch.cat([inputs0, state0], dim=-1)
    del inputs0
    del state0

    # Compute X W  + b as new features
    if is_c_opt:
      x = self.c_layer(x)
    else:
      # print('x device', x.device, 'ru_weight device', self.ru_weights.device)
      x = self.ru_layer(x)

    # reshape into 2d tensor
    return torch.reshape(x, [batch_size, num_nodes, -1])


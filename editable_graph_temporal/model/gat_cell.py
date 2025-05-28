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
"""GATRNN unit cell model.

Defines the unit cell used in the GATRNN model.
"""

from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_geometric.utils import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2SeqAttrs:
  """Stores model-related arguments."""

  def _initialize_arguments(self, args):
    """Initializes model arguments.

    Args:
      args: python argparse.ArgumentParser class, we only use model-related
        arguments here.
    """
    self.input_dim = args.input_dim
    self.output_dim = args.output_dim
    self.rnn_units = args.hidden_dim
    self.num_nodes = args.num_nodes
    self.input_len = args.input_len
    self.output_len = args.output_len
    self.num_relation_types = args.num_relation_types

    self.dropout = args.dropout
    self.negative_slope = args.negative_slope

    self.num_rnn_layers = args.num_layers
    self.lr = args.learning_rate
    self.activation = args.activation
    self.share_attn_weights = args.share_attn_weights


class GATGRUCell(LightningModule, Seq2SeqAttrs):
  """Implements a single unit cell of GATRNN model."""

  def __init__(self, args):
    """Instantiates the GATRNN unit cell model.

    Args:
      args: python argparse.ArgumentParser class, we only use model-related
        arguments here.
    """
    super().__init__()
    self._initialize_arguments(args)

    self.activation = torch.tanh
    input_size = 2 * self.rnn_units

    # gconv
    weight_dim = input_size if self.share_attn_weights else 2 * input_size
    biases_dim = self.rnn_units if self.share_attn_weights else 2 * self.rnn_units

    self.r_weights = nn.Parameter(
        torch.empty((weight_dim, self.rnn_units, self.num_relation_types - 1),
                    device=device))
    self.r_biases = nn.Parameter(
        torch.zeros((biases_dim, self.num_relation_types - 1), device=device))
    self.u_weights = nn.Parameter(
        torch.empty((weight_dim, self.rnn_units, self.num_relation_types - 1),
                    device=device))
    self.u_biases = nn.Parameter(
        torch.zeros((biases_dim, self.num_relation_types - 1), device=device))
    self.c_weights = nn.Parameter(
        torch.empty((weight_dim, self.rnn_units, self.num_relation_types - 1),
                    device=device))
    self.c_biases = nn.Parameter(
        torch.zeros((biases_dim, self.num_relation_types - 1), device=device))

    torch.nn.init.xavier_normal_(self.r_weights)
    torch.nn.init.xavier_normal_(self.u_weights)
    torch.nn.init.xavier_normal_(self.c_weights)

  def forward(self, inputs, hx, adj, global_embs):
    r"""Forward computation of a single unit cell of GATRNN model.

    The forward computation is generally the same as
    that of a GRU cell of sequence model, but gate vectors and candidate
    hidden vectors are computed by graph attention
    network based convolutions.

    Args:
      inputs: input one-step time series, with shape (batch_size,
        self.num_nodes, self.rnn_units).
      hx: hidden vectors from the last unit, with shape(batch_size,
        self.num_nodes, self.rnn_units). If this is the first unit, usually hx
        is supposed to be a zero vector.
      adj: adjacency matrix, with shape (self.num_nodes, self.num_nodes).
      global_embs: global embedding matrix, with shape (self.num_nodes,
        self.rnn_units).

    Returns:
      hx: new hidden vector.
    """
    r = torch.tanh(self._gconv(inputs, adj, global_embs, hx, 'r'))
    u = torch.tanh(self._gconv(inputs, adj, global_embs, hx, 'u'))
    c = self._gconv(inputs, adj, global_embs, r * hx,
                    'c')  # element-wise multiplication
    if self.activation is not None:
      c = self.activation(c)

    hx = u * hx + (1.0 - u) * c

    del r
    del u
    del c

    return hx

  @staticmethod
  def _concat(x, x_):
    r"""Concatenates two tensors along the first dimension.

    Args:
      x: first input tensor.
      x_: second input tensor.

    Returns:
      concatenation tensor of x and x_.
    """
    x_ = x_.unsqueeze(0)
    return torch.cat([x, x_], dim=0)

  def _gconv(self, inputs, adj_mx, global_embs, state, option='r'):
    r"""Graph attention network based convolution computation.

    Args:
      inputs: input vector, with shape (batch_size, self.num_nodes,
        self.rnn_units).
      adj_mx: adjacency matrix, with shape (self.num_nodes, self.num_nodes).
      global_embs: global embedding matrix, with shape (self.num_nodes,
        self.rnn_units).
      state: hidden vectors from the last unit, with shape(batch_size,
        self.num_nodes, self.rnn_units). If this is the first unit, usually hx
        is supposed to be a zero vector.
      option: indicate whether the output is reset gate vector ('r'), update
        gate vector ('u'), or candidate hidden vector ('c').

    Returns:
      out: output, can be reset gate vector (option is 'r'), update gate
      vector (option is 'u'), or
        candidate hidden vector (option is 'c').
    """
    batch_size = inputs.shape[0]
    num_nodes = self.num_nodes

    x = torch.cat([inputs, state], dim=-1)  # input_dim
    out = torch.zeros(
        size=(batch_size, num_nodes, self.rnn_units), device=device)

    for relation_id in range(self.num_relation_types - 1):
      if option == 'r':
        r_weights_left = self.r_weights[:2 * self.rnn_units, :, relation_id]
        r_biases_left = self.r_biases[:self.rnn_units, relation_id]
        r_weights_right = r_weights_left if self.share_attn_weights else self.r_weights[
            2 * self.rnn_units:, :, relation_id]
        r_biases_right = r_biases_left if self.share_attn_weights else self.r_biases[
            self.rnn_units:, relation_id]
        x_left = torch.matmul(x, r_weights_left) + r_biases_left
        x_right = torch.matmul(x, r_weights_right) + r_biases_right
      elif option == 'u':
        u_weights_left = self.u_weights[:2 * self.rnn_units, :, relation_id]
        u_biases_left = self.u_biases[:self.rnn_units, relation_id]
        u_weights_right = u_weights_left if self.share_attn_weights else self.u_weights[
            2 * self.rnn_units:, :, relation_id]
        u_biases_right = u_biases_left if self.share_attn_weights else self.u_biases[
            self.rnn_units:, relation_id]
        x_left = torch.matmul(x, u_weights_left) + u_biases_left
        x_right = torch.matmul(x, u_weights_right) + u_biases_right
      elif option == 'c':
        c_weights_left = self.c_weights[:2 * self.rnn_units, :, relation_id]
        c_biases_left = self.c_biases[:self.rnn_units, relation_id]
        c_weights_right = c_weights_left if self.share_attn_weights else self.c_weights[
            2 * self.rnn_units:, :, relation_id]
        c_biases_right = c_biases_left if self.share_attn_weights else self.c_biases[
            self.rnn_units:, relation_id]
        x_left = torch.matmul(x, c_weights_left) + c_biases_left
        x_right = torch.matmul(x, c_weights_right) + c_biases_right

      i, j = torch.nonzero(adj_mx[:, :, relation_id], as_tuple=True)
      i, j = i.to(device), j.to(device)
      x_left_per_edge = x_left.index_select(1, i)
      x_right_per_edge = x_right.index_select(1, j)
      x_per_edge = x_left_per_edge + x_right_per_edge
      x_per_edge = nn.functional.leaky_relu(x_per_edge, self.negative_slope)

      alpha = (x_per_edge * global_embs[i]).sum(dim=2)
      alpha = softmax(alpha, index=i, num_nodes=num_nodes, dim=1)

      attns = torch.zeros([batch_size, num_nodes, num_nodes], device=device)
      batch_idxs = torch.arange(batch_size, device=device)
      batch_expand = torch.repeat_interleave(batch_idxs, len(i), dim=0)
      i_expand = torch.repeat_interleave(
          i.view(1, -1), batch_size, dim=0).view(-1)
      j_expand = torch.repeat_interleave(
          j.view(1, -1), batch_size, dim=0).view(-1)
      indices = (batch_expand, i_expand, j_expand)
      attns.index_put_(indices, alpha.view(-1))

      zero_mask = (adj_mx[:, :,
                          relation_id] == 0).unsqueeze(0).repeat_interleave(
                              batch_size, dim=0)
      zero_coeffs = torch.ones([batch_size, num_nodes, num_nodes],
                               device=device) / zero_mask.float().sum(
                                   dim=-1, keepdim=True)
      attns[zero_mask] = zero_coeffs[zero_mask]

      out += torch.bmm(adj_mx[:, :, relation_id] * attns, x_right) + x_left

    return out

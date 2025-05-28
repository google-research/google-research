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

"""Latent graph forecaster cell with different graph convolution filter rules."""
import argparse
from typing import Sequence

import numpy as np
from pytorch_lightning import LightningModule
from src import utils
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2SeqAttrs:
  """Initialize the hyperparameters from the arguments.

  Attributes:
    input_dim: input feature dimension
    output_dim: output feature dimension
    rnn_units: hidden dimension of the rnn layer
    num_nodes: graph size
    input_len: input sequence length
    output_len: output sequence length
    max_diffusion_step: graph convolution filter order
    filter_type: 'learned', 'laplacian', 'random_walk', 'dual_random_rank'
    use_gc_ru: flag for using graph convolution (gc) for r and u gate
    use_gc_c: flag for using gc for the memory c cell
    use_curriculum_learning: flag for using curriculum learning
    cl_decay_steps:
    dropout:
    batch_size:
    num_rnn_layers:
    lr:
    activation:
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
    self.use_curriculum_learning = args.use_curriculum_learning
    self.cl_decay_steps = args.cl_decay_steps
    self.dropout = args.dropout

    self.batch_size = args.batch_size
    self.num_rnn_layers = args.num_layers
    self.lr = args.learning_rate
    self.activation = args.activation

    # Overwrite with tunable params
    # self.lr = config["lr"]
    # self.rnn_units = config["rnn_units"]


class LayerParams:
  """Allocate trainable parameters for one layer."""

  def __init__(self, rnn_network, layer_type):
    self._rnn_network = rnn_network
    self._params_dict = {}
    self._biases_dict = {}
    self._type = layer_type

  def get_weights(self, shape):
    if shape not in self._params_dict:
      nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
      torch.nn.init.xavier_normal_(nn_param)
      self._params_dict[shape] = nn_param
      self._rnn_network.register_parameter(
          "{}_weight_{}".format(self._type, str(shape)), nn_param)
    return self._params_dict[shape]

  def get_biases(self,
                 length,
                 bias_start = 0.0):
    """Obtain bias tensor parameters.

    Args:
      length: dimension
      bias_start: initial value

    Returns:

    """
    if length not in self._biases_dict:
      biases = torch.nn.Parameter(torch.empty(length, device=device))
      torch.nn.init.constant_(biases, bias_start)
      self._biases_dict[length] = biases
      self._rnn_network.register_parameter(
          "{}_biases_{}".format(self._type, str(length)), biases)

    return self._biases_dict[length]


class LGFCell(LightningModule):
  """Latent graph forecasting customized cell.

  Attributes:
    activation:
    supports: list of graph convolution filters
    ru_weights: r and u gate weights are concatenated
    ru_biases:
    c_weights: memory cell weights
    c_biases:
  """

  def __init__(self, adj_mx, args):

    super().__init__()
    Seq2SeqAttrs.__init__(self, args)

    self.activation = torch.tanh
    # support other nonlinearities up here?
    self.supports = []

    supports = []
    if self.filter_type == "laplacian":
      supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
    elif self.filter_type == "random_walk":
      supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
    elif self.filter_type == "dual_random_walk":
      supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
      supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
    elif self.filter_type == "residual":
      supports.append(utils.calculate_residual_random_walk_matrix(adj_mx).T)

    for support in supports:
      self.supports.append(self._build_sparse_matrix(support))

    if self.filter_type == "learned":
      # return supports as a list of variables with gradients
      # self.adj_mx = nn.Parameter(torch.eye(self.num_nodes, device=device))
      self.supports.append(adj_mx.to_sparse())

    # parameter registration issue
    num_matrices = len(self.supports) * self.max_diffusion_step + 1
    input_size = 2 * self.rnn_units

    if self.use_gc_ru:
      # gconv
      self.ru_weights = nn.Parameter(
          torch.empty((input_size * num_matrices, 2 * self.rnn_units),
                      device=device))
      self.ru_biases = nn.Parameter(
          torch.zeros((2 * self.rnn_units), device=device))
    else:  # fc
      self.ru_weights = nn.Parameter(
          torch.empty((input_size, 2 * self.rnn_units)))
      self.ru_biases = nn.Parameter(
          torch.zeros((2 * self.rnn_units), device=device))
    if self.use_gc_c:
      # gconv
      self.c_weights = nn.Parameter(
          torch.empty((input_size * num_matrices, self.rnn_units)))
      self.c_biases = nn.Parameter(torch.zeros((self.rnn_units), device=device))
    else:
      # fc
      self.c_weights = nn.Parameter(torch.empty((input_size, self.rnn_units)))
      self.c_biases = nn.Parameter(torch.zeros((self.rnn_units), device=device))

    torch.nn.init.xavier_normal_(self.ru_weights)
    torch.nn.init.xavier_normal_(self.c_weights)
    # self._fc_params = LayerParams(self, 'fc')
    # self._gconv_params = LayerParams(self, 'gconv')

  @classmethod
  def _build_sparse_matrix(cls, mat):
    mat = mat.tocoo()
    indices = np.column_stack((mat.row, mat.col))
    # ensure row-major ordering equals torch.sparse.sparse_reorder(L)
    indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
    mat = torch.sparse_coo_tensor(indices.T, mat.data, mat.shape, device=device)
    return mat

  def forward(self, inputs, hx):
    """Gated recurrent unit (GRU) with Graph Convolution.

    Args:
        inputs (tensor): [batch, num_nodes, input_dim]
        hx (tensor): [batch, num_nodes, rnn_units)

    Returns:
        new_state (tensor): [batch, num_nodes,  rnn_units].
    """
    # print('dcrnn cell')
    # for name, param in self.named_parameters():
    #     if param.requires_grad:
    #         print(name, )
    if self.use_gc_ru:
      ru_opt = self._gconv
    else:
      ru_opt = self._fc

    output_size = 2 * self.rnn_units

    # with autograd.detect_anomaly():

    value = ru_opt(inputs, hx, output_size, is_c_opt=False)
    value = torch.tanh(value)
    r, u = torch.split(
        tensor=value, split_size_or_sections=self.rnn_units, dim=-1)
    del value

    # value = torch.reshape(value, (-1, self._num_nodes, output_size))
    # r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
    # u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

    if self.use_gc_c:
      c_opt = self._gconv
    else:
      c_opt = self._fc

    c = c_opt(
        inputs, r * hx, self.rnn_units,
        is_c_opt=True)  # element-wise multiplication
    if self.activation is not None:
      c = self.activation(c)

    hx = u * hx + (1.0 - u) * c

    del r
    del u
    del c

    return hx

  @classmethod
  def _concat(cls, x, x_):
    x_ = x_.unsqueeze(0)
    return torch.cat([x, x_], dim=0)

  def _gconv(self, inputs, state, output_size, is_c_opt=False):
    """Perform graph convolution operaton.

    Args:
        inputs: [batch, num_nodes, input_dim]
        state: [batch, num_nodes, hidden_dim]
        output_size:
        is_c_opt:

    Returns:
        gconv: [batch, num_nodes, output_dim]
    """
    batch_size = self.batch_size
    num_nodes = self.num_nodes

    x = torch.cat([inputs, state], dim=-1)  # input_dim
    input_size = x.size(2)

    x0 = x.permute(1, 2, 0)  # (num_nodes, input_dim, batch_size)
    x0 = torch.reshape(x0, shape=[num_nodes, input_size * batch_size])
    x = torch.unsqueeze(x0, 0)

    # Compute g(A)X as graph features
    if self.max_diffusion_step == 0:
      pass
    else:
      for support in self.supports:
        # adj = support.get_weights((num_nodes, num_nodes))
        # support = torch.tanh(adj) #limit the range
        support = support.to(device)
        x1 = torch.sparse.mm(support, x0)
        # x1 = torch.mm(support, x0)
        x = self._concat(x, x1)

        for k in range(2, self.max_diffusion_step + 1):  # pylint: disable=unused-variable
          x2 = 2 * torch.sparse.mm(support, x1) - x0
          # x2 = 2 * torch.mm(support, x1) - x0
          x = self._concat(x, x2)
          x1, x0 = x2, x1

    del x0

    num_matrices = len(
        self.supports) * self.max_diffusion_step + 1  # Adds for x itself.
    x = torch.reshape(
        x, shape=[num_matrices, num_nodes, input_size, batch_size])
    x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
    x = torch.reshape(
        x, shape=[batch_size * num_nodes, input_size * num_matrices])

    # Compute g(A)X W + b as new features
    if is_c_opt:
      x = torch.matmul(x, self.c_weights)
      x += self.c_biases
    else:
      x = torch.matmul(x, self.ru_weights)
      x += self.ru_biases

    # Reshape res back to 3D: (batch_size * num_node, state_dim)
    # -> (batch_size, num_node, state_dim)
    return torch.reshape(x, [batch_size, num_nodes, output_size])

  def _fc(self, inputs, state, output_size, is_c_opt=False):
    """Embed with fully connected layer.

    Args:
        inputs: [batch, num_nodes, input_dim]
        state: [batch, num_nodes, hidden_dim]
        output_size:
        is_c_opt:

    Returns:
        output: [batch, num_nodes, output_dim]
    """
    batch_size = self.batch_size
    num_nodes = self.num_nodes

    inputs0 = torch.reshape(inputs, (batch_size * num_nodes, -1))
    state0 = torch.reshape(state, (batch_size * num_nodes, -1))

    # inputs0 = torch.reshape(inputs, (batch_size, -1))
    # state0 = torch.reshape(state, (batch_size, -1))

    # print('input device', inputs0.device, 'states device', state0.device)
    x = torch.cat([inputs0, state0], dim=-1)
    del inputs0
    del state0

    # Compute X W  + b as new features
    if is_c_opt:
      x = torch.matmul(x, self.c_weights)
      x += self.c_biases
    else:
      # print('x device', x.device, 'ru_weight device', self.ru_weights.device)
      x = torch.matmul(x, self.ru_weights)
      x += self.ru_biases

    # reshape into 2d tensor
    return torch.reshape(x, [batch_size, num_nodes, output_size])

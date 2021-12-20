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

"""Class that implement SDT."""
import torch
import torch.nn as nn


class SDT(nn.Module):
  """Fast implementation of soft decision tree in PyTorch.

     Adapted from https://github.com/xuyxu/Soft-Decision-Tree

    Args:
      input_dim : int
        The number of input dimensions.
      output_dim : int
        The number of output dimensions. For example, for a multi-class
        classification problem with `K` classes, it is set to `K`.
      depth : int, default=5
        The depth of the soft decision tree. Since the soft decision tree is
        a full binary tree, setting `depth` to a large value will drastically
        increases the training and evaluating cost.
      lamda : float, default=1e-3
        The coefficient of the regularization term in the training loss. Please
        refer to the paper on the formulation of the regularization term.
      use_cuda : bool, default=False
        When set to `True`, use GPU to fit the model. Training a soft decision
        tree using CPU could be faster considering the inherent data forwarding
        process.
    other Attributes:
      internal_node_num_ : int
        The number of internal nodes in the tree. Given the tree depth `d`, it
        equals to :math:`2^d - 1`.
      leaf_node_num_ : int The number of leaf nodes in the tree. Given the tree
        depth `d`, it equals to :math:`2^d`.
      penalty_list : list
        A list storing the layer-wise coefficients of the regularization term.
      inner_nodes : torch.nn.Sequential
        A container that simulates all internal nodes in the soft decision tree.
        The sigmoid activation function is concatenated to simulate the
        probabilistic routing mechanism.
      leaf_nodes : torch.nn.Linear
        A `nn.Linear` module that simulates all leaf nodes in the tree.
  """

  def __init__(self, input_dim, output_dim, depth=5, lamda=1e-3, device="cpu"):
    super(SDT, self).__init__()

    self.input_dim = input_dim
    self.output_dim = output_dim

    self.depth = depth
    self.lamda = lamda
    self.device = device

    self._validate_parameters()

    self.internal_node_num_ = 2 ** self.depth - 1
    self.leaf_node_num_ = 2 ** self.depth

    # Different penalty coefficients for nodes in different layers
    self.penalty_list = [
        self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
    ]

    # Initialize internal nodes and leaf nodes, the input dimension on
    # internal nodes is added by 1, serving as the bias.
    self.inner_nodes = nn.Sequential(
        nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
        nn.Sigmoid(),
    )

    self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                self.output_dim,
                                bias=False)

  def forward(self, x, is_training_data=False):
    mu, penalty_ = self._forward(x)
    y_pred = self.leaf_nodes(mu)

    # When `x` is the training data, the model also returns the penalty
    # to compute the training loss.
    if is_training_data:
      return y_pred, penalty_
    else:
      return y_pred

  def _forward(self, x):
    """Implementation on the data forwarding process."""

    batch_size = x.size()[0]
    x = self._data_augment(x)

    path_prob = self.inner_nodes(x)
    path_prob = torch.unsqueeze(path_prob, dim=2)
    path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

    mu = x.data.new(batch_size, 1, 1).fill_(1.0)
    penalty_ = torch.tensor(0.0).to(self.device)

    # Iterate through internal odes in each layer to compute the final path
    # probabilities and the regularization term.
    begin_idx = 0
    end_idx = 1

    for layer_idx in range(0, self.depth):
      path_prob_ = path_prob[:, begin_idx:end_idx, :]

      # Extract internal nodes in the current layer to compute the
      # regularization term
      penalty_ = penalty_ + self._cal_penalty(layer_idx, mu, path_prob_)
      mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)

      mu = mu * path_prob_  # update path probabilities

      begin_idx = end_idx
      end_idx = begin_idx + 2 ** (layer_idx + 1)

    mu = mu.view(batch_size, self.leaf_node_num_)

    return mu, penalty_

  def _cal_penalty(self, layer_idx, mu, path_prob_):
    penalty = torch.tensor(0.0).to(self.device)

    batch_size = mu.size()[0]
    mu = mu.view(batch_size, 2 ** layer_idx)
    path_prob_ = path_prob_.view(batch_size, 2 ** (layer_idx + 1))

    for node in range(0, 2 ** (layer_idx + 1)):
      alpha = torch.sum(
          path_prob_[:, node] * mu[:, node // 2], dim=0
      ) / torch.sum(mu[:, node // 2], dim=0)

      coeff = self.penalty_list[layer_idx]

      penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

    return penalty

  def _data_augment(self, x):
    """Add a constant input `1` onto the front of each sample."""
    batch_size = x.size()[0]
    x = x.view(batch_size, -1)
    bias = torch.ones(batch_size, 1).to(self.device)
    x = torch.cat((bias, x), 1)

    return x

  def _validate_parameters(self):

    if not self.depth > 0:
      msg = ("The tree depth should be strictly positive, but got {}"
             "instead.")
      raise ValueError(msg.format(self.depth))

    if not self.lamda >= 0:
      msg = (
          "The coefficient of the regularization term should not be"
          " negative, but got {} instead."
      )
      raise ValueError(msg.format(self.lamda))

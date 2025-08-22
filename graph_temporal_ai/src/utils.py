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

"""Utility functions."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

import torch
from torch.nn.modules.loss import _Loss


def calculate_normalized_laplacian(adj):
  """Compute L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2, D = diag(A 1).

  Args:
    adj: original adjacency matrix

  Returns:
    filter rule
  """
  adj = sp.coo_matrix(adj)
  d = np.array(adj.sum(1))
  d_inv_sqrt = np.power(d, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
  normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
      d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
  return normalized_laplacian.astype(np.float32)


def calculate_residual_random_walk_matrix(adj_mx,
                                          eps = 0.1):
  """Calculate Graph Isomophism ( I*epsilon + g(A) ).

  Args:
    adj_mx: input graph adjancy matrix
    eps: random walk background parameter

  Returns:
    residual random walk matrix
  """

  adj_mx = sp.coo_matrix(adj_mx)
  d = np.array(adj_mx.sum(1))
  d_inv = np.power(d, -1).flatten()
  d_inv[np.isinf(d_inv)] = 0.
  d_mat_inv = sp.diags(d_inv)
  random_walk_mx = sp.coo_matrix(
      np.eye(adj_mx.shape[0]) * eps) + d_mat_inv.dot(adj_mx).tocoo()
  return random_walk_mx.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
  adj_mx = sp.coo_matrix(adj_mx)
  d = np.array(adj_mx.sum(1))
  d_inv = np.power(d, -1).flatten()
  d_inv[np.isinf(d_inv)] = 0.
  d_mat_inv = sp.diags(d_inv)
  random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
  return random_walk_mx.astype(np.float32)


def calculate_reverse_random_walk_matrix(adj_mx):
  return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max = 2,
                               undirected = True):
  """Calculating scaled Laplacian matrix.

  Args:
    adj_mx:
    lambda_max:
    undirected:

  Returns:

  """
  if undirected:
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
  mat = calculate_normalized_laplacian(adj_mx)
  if lambda_max is None:
    lambda_max, _ = linalg.eigsh(mat, 1, which='LM')
    lambda_max = lambda_max[0]
  mat = sp.csr_matrix(mat)
  sz, _ = mat.shape
  i_mat = sp.identity(sz, format='csr', dtype=mat.dtype)
  mat = (2 / lambda_max * mat) - i_mat
  return mat.astype(np.float32)


def assert_shape(x, shape):
  """ex:assert_shape(conv_input_array,[8,3,None,None])."""
  assert len(x.shape) == len(shape), (x.shape, shape)
  for a, b in zip(x.shape, shape):
    if isinstance(b, int):
      assert a == b, (x.shape, shape)


class MAPELoss(_Loss):
  """Mean Absolute Percentange Error.

  """

  __constants__ = ['reduction']

  # def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
  #   super().__init__(size_average, reduce, reduction)

  def forward(self, inputs, target):
    entries = torch.nonzero(target, as_tuple=True)
    loss = torch.mean(
        (inputs[entries] - target[entries]).abs() / target[entries].abs())
    return loss

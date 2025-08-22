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

"""This file contains necessary functions for SEM simulation.

We visulize k means clustering for environment building, data loading and
preprocessing, saving, and selecting W matrix.
"""

import argparse
import copy
import glob
import os
import random

import igraph as ig
from imblearn.over_sampling import RandomOverSampler
from metrics import count_accuracy
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


# pylint: disable=invalid-name
# pylint: disable=f-string-without-interpolation
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
# pylint: disable=dangerous-default-value
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin
# pylint: disable=undefined-variable
# pylint: disable=g-bare-generic
# pylint: disable=g-builtin-op
# pylint: disable=anomalous-backslash-in-string
def is_dag(W):
  """Checks if the given matrix W encodes a DAG.

  Args:
    W: an adjacency matrix

  Returns:
    True if W encodes a DAG and false otherwise
  """
  G = ig.Graph.Weighted_Adjacency(W.tolist())
  return G.is_dag()


def save_dag(
    W,
    exp,
    vertex_label=['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
    vertex_size=80,
    vs_label_size=50):
  """Saves the W to a file in the current directory."""

  G = ig.Graph.Weighted_Adjacency(W.tolist())
  # G.vs['vertex_label'] =
  #  ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1','y2', 'y3']
  G.vs['vertex_label'] = vertex_label
  ig.plot(
      G,
      f'{exp}.png',
      vertex_size=vertex_size,
      vertex_label=G.vs['vertex_label'],
      vertex_label_size=vs_label_size,
      edge_arrow_size=2,
      edge_arrow_width=2,
  )


def simulate_dag(d, s0, graph_type):
  """Simulates random DAG with some expected number of edges.

  Args:
    d: number of nodes
    s0: expected number of edges
    graph_type: ER, SF, BP

  Returns:
    B: [d, d] binary adj matrix of DAG
  """

  def _random_permutation(M):
    # np.random.permutation permutes first axis only
    P = np.random.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P

  def _random_acyclic_orientation(B_und):
    return np.tril(_random_permutation(B_und), k=-1)

  def _graph_to_adjmat(G):
    return np.array(G.get_adjacency().data)

  if graph_type == 'ER':
    # Erdos-Renyi
    G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
    B_und = _graph_to_adjmat(G_und)
    B = _random_acyclic_orientation(B_und)
  elif graph_type == 'SF':
    # Scale-free, Barabasi-Albert
    G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
    B = _graph_to_adjmat(G)
  elif graph_type == 'BP':
    # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
    top = int(0.2 * d)
    G = ig.Graph.Random_Bipartite(
        top, d - top, m=s0, directed=True, neimode=ig.OUT)
    B = _graph_to_adjmat(G)
  else:
    raise ValueError('unknown graph type')
  B_perm = _random_permutation(B)
  assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
  return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
  """Simulates SEM parameters for a DAG.

  Args:
    B: [d, d] binary adj matrix of DAG
    w_ranges: disjoint weight ranges

  Returns:
    W: [d, d] weighted adj matrix of DAG
  """
  W = np.zeros(B.shape)
  S = np.random.randint(len(w_ranges), size=B.shape)
  for i, (low, high) in enumerate(w_ranges):
    U = np.random.uniform(low=low, high=high, size=B.shape)
    W += B * (S == i) * U
  return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
  """Simulates samples from linear SEM with specified type of noise.

  For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

  Args:
    W: [d, d] weighted adj matrix of DAG
    n: number of samples, n=inf mimics population risk
    sem_type: Type of the SEM, either gauss, exp, gumbel, uniform, logistic, or
      poisson
    noise_scale: scale parameter of additive noise, default all ones

  Returns:
    X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
  """

  def _simulate_single_equation(X, w, scale):

    if sem_type == 'gauss':
      z = np.random.normal(scale=scale, size=n)
      x = X @ w + z
    elif sem_type == 'exp':
      z = np.random.exponential(scale=scale, size=n)
      x = X @ w + z
    elif sem_type == 'gumbel':
      z = np.random.gumbel(scale=scale, size=n)
      x = X @ w + z
    elif sem_type == 'uniform':
      z = np.random.uniform(low=-scale, high=scale, size=n)
      x = X @ w + z
    elif sem_type == 'logistic':
      x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
    elif sem_type == 'poisson':
      x = np.random.poisson(np.exp(X @ w)) * 1.0
    else:
      raise ValueError('unknown sem type')
    return x

  d = W.shape[0]
  if noise_scale is None:
    scale_vec = np.ones(d)
  elif np.isscalar(noise_scale):
    scale_vec = noise_scale * np.ones(d)
  else:
    if len(noise_scale) != d:
      raise ValueError('noise scale must be a scalar or has length d')
    scale_vec = noise_scale
  if not is_dag(W):
    raise ValueError('W must be a DAG')
  if np.isinf(n):  # population risk for linear gauss SEM
    if sem_type == 'gauss':
      # make 1/d X'X = true cov
      X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
      return X
    else:
      raise ValueError('population risk not available')
  # empirical risk
  G = ig.Graph.Weighted_Adjacency(W.tolist())
  ordered_vertices = G.topological_sorting()
  assert len(ordered_vertices) == d
  X = np.zeros([n, d])
  for j in ordered_vertices:
    parents = G.neighbors(j, mode=ig.IN)
    X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j],
                                        scale_vec[j])
  return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
  """Simulates samples from nonlinear SEM.

  Args:
    B: [d, d] binary adj matrix of DAG
    n: num of samples
    sem_type: The type of the SEM: mlp, mim, gp, gp-add
    noise_scale: scale parameter of additive noise.

  Returns:
    X (np.ndarray): [n, d] sample matrix
  """

  def _simulate_single_equation(X, scale, hidden=100):
    z = np.random.normal(scale=scale, size=n)
    pa_size = X.shape[1]
    if pa_size == 0:
      return z
    if sem_type == 'mlp':
      W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
      W1[np.random.rand(*W1.shape) < 0.5] *= -1
      W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
      W2[np.random.rand(hidden) < 0.5] *= -1
      x = sigmoid(X @ W1) @ W2 + z
    elif sem_type == 'mim':
      w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w1[np.random.rand(pa_size) < 0.5] *= -1
      w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w2[np.random.rand(pa_size) < 0.5] *= -1
      w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w3[np.random.rand(pa_size) < 0.5] *= -1
      x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
    elif sem_type == 'gp':
      gp = GaussianProcessRegressor()
      x = gp.sample_y(X, random_state=None).flatten() + z
    elif sem_type == 'gp-add':
      gp = GaussianProcessRegressor()
      x = sum([
          gp.sample_y(X[:, i, None], random_state=None).flatten()
          for i in range(X.shape[1])
      ]) + z
    else:
      raise ValueError('unknown sem type')
    return x

  d = B.shape[0]
  scale_vec = noise_scale if noise_scale else np.ones(d)
  X = np.zeros([n, d])
  G = ig.Graph.Adjacency(B.tolist())
  ordered_vertices = G.topological_sorting()
  assert len(ordered_vertices) == d
  for j in ordered_vertices:
    parents = G.neighbors(j, mode=ig.IN)
    X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
  return X


def simulate_nonlinear_sem_with_counterfactual(B,
                                               n,
                                               irm_param,
                                               counter_scale=2,
                                               noise_scale=None,
                                               hidden=100):
  """Simulates samples from nonlinear SEM."""

  scale = 1
  d = irm_param['num_xy']
  z = np.random.normal(scale=scale, size=n)
  X = np.zeros([n, d])

  # the X's are random Gaussian noise
  for i in range(1, d, 1):
    X[:, i] = np.random.normal(scale=scale, size=n)

  W1 = np.random.uniform(low=0.5, high=2.0, size=[d - 1, hidden])
  W1[np.random.rand(*W1.shape) < 0.5] *= -1
  W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
  W2[np.random.rand(hidden) < 0.5] *= -1
  z = np.random.normal(scale=scale, size=n)
  X[:, 0] = sigmoid(X @ W1) @ W2 + z


def customized_graph(graph_type='xyz_complex',
                     vertex_label=['Y', 'X1', 'X2', 'S1']):
  """Creates customized graphs based on data generation process.

  Args:
    graph_type: the type of graph to generate
    vertex_label: the names of the vertices of the graph Returns a adjacency
      matrix of the genereated graph

  Returns:
    Processed graph
  """

  if graph_type == 'binary':
    g = ig.Graph(
        n=4,
        edges=[[1, 0], [2, 0], [0, 3]],
        vertex_attrs={'name': ['Y', 'X1', 'X2', 'S1']},
        directed=True)

  if graph_type == 'xyz_complex':
    g = ig.Graph(
        n=10,
        edges=[[1, 7], [2, 7], [3, 7], [3, 8], [4, 8], [5, 8], [6, 9], [7, 0],
               [8, 0], [9, 0]],
        vertex_attrs={
            'name': ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1', 'y2', 'y3']
        },
        directed=True)

  elif graph_type == 'castle_complex':
    g = ig.Graph(
        n=10,
        edges=[[1, 2], [1, 3], [1, 4], [2, 5], [2, 0], [3, 0], [3, 6], [3, 7],
               [0, 8], [0, 9], [6, 9]],
        vertex_attrs={
            'vertex_label': [
                'y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'
            ]
        },
        directed=True)

  elif graph_type == 'IRM' or graph_type == 'IRM_counter':
    edges = list()
    for _, vertex in enumerate(vertex_label):
      if 'X' in vertex:
        edges.append((int(vertex[1]), 0))
    g = ig.Graph(
        n=len(vertex_label),
        edges=edges,
        vertex_attrs={'name': vertex_label},
        directed=True)

  else:
    edges = list()
    for _, vertex in enumerate(vertex_label):
      if 'X' in vertex:
        edges.append((int(vertex[1]), 0))
    g = ig.Graph(
        n=len(vertex_label),
        edges=edges,
        vertex_attrs={'name': vertex_label},
        directed=True)

  g_adj_m = np.array(g.get_adjacency().data)

  return g_adj_m


def simulate_linear_daring(W,
                           n,
                           sem_type,
                           noise_scale=1,
                           setting='mixed_noise'):
  """Simulates linear DARING.

  Args:
    W: [d, d] binary adjacency matrix of DAG
    n: num of samples
    sem_type: The type of the SEM: mlp, mim, gp, gp-add
    noise_scale: scale parameter of additive noise, default all ones
    setting: noise setting.

  Returns:
    X: [n, d] sample matrix
  """

  def _simulate_single_equation(X, w, scale):
    # Tensor sizes: X: [n, num of parents], w: [num of parents], x: [n]
    if sem_type == 'gauss':
      z = np.random.normal(scale=scale, size=n)
      x = X @ w + z
    elif sem_type == 'exp':
      z = np.random.exponential(scale=scale, size=n)
      x = X @ w + z
    elif sem_type == 'gumbel':
      z = np.random.gumbel(scale=scale, size=n)
      x = X @ w + z
    elif sem_type == 'uniform':
      z = np.random.uniform(low=-scale, high=scale, size=n)
      x = X @ w + z
    elif sem_type == 'logistic':
      x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
    elif sem_type == 'poisson':
      x = np.random.poisson(np.exp(X @ w)) * 1.0
    else:
      raise ValueError('unknown sem type')
    return x

  d = W.shape[0]
  if noise_scale is None:
    scale_vec = np.ones(d)
  elif np.isscalar(noise_scale):
    scale_vec = noise_scale * np.ones(d)
  else:
    if len(noise_scale) != d:
      raise ValueError('noise scale must be a scalar or has length d')
    scale_vec = noise_scale
  if not is_dag(W):
    raise ValueError('W must be a DAG')
  if np.isinf(n):  # population risk for linear gauss SEM
    if sem_type == 'gauss':
      # make 1/d X'X = true cov
      X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
      return X
    else:
      raise ValueError('population risk not available')

  # empirical risk
  G = ig.Graph.Weighted_Adjacency(W.tolist())
  ordered_vertices = G.topological_sorting()
  assert len(ordered_vertices) == d

  X = np.zeros([n, d])

  if setting == 'mixed_noise':
    noise_types = ['gauss', 'uniform', 'exp', 'gumbel']

    for j in ordered_vertices:
      parents = G.neighbors(j, mode=ig.IN)
      sem_type = random.choice(noise_types)
      X[:, j] = (
          _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j]))

  elif setting == 'id_mixed_noise':
    scale_choices = [1, 2, 3, 4]

    for j in ordered_vertices:
      parents = G.neighbors(j, mode=ig.IN)
      scale_vec = random.choice(scale_choices) * np.ones(d)
      sem_type = 'uniform'
      X[:, j] = (
          _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j]))

  elif setting == 'non_id_mixed_noise':
    scale_choices = [1, 2, 3, 4]

    for j in ordered_vertices:
      parents = G.neighbors(j, mode=ig.IN)
      sem_type = 'gauss'
      scale_vec = random.choice(scale_choices) * np.ones(d)
      X[:, j] = (
          _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j]))
  else:
    ValueError('Unrecognizable setting')

  return X


def simulate_nonlinear_daring(B,
                              n,
                              sem_type,
                              noise_scale=None,
                              setting='mixed_noise'):
  """Simulates samples from nonlinear SEM.

  Args:
    B: [d, d] binary adj matrix of DAG
    n: num of samples
    sem_type: The type of the SEM: mlp, mim, gp, gp-add
    noise_scale: scale parameter of additive noise, default all ones
    setting: noise setting.

  Returns:
    X: [n, d] sample matrix
  """

  def _simulate_single_equation(X, scale, hidden=100):
    # X: [n, num of parents], x: [n]
    z = np.random.normal(scale=scale, size=n)
    pa_size = X.shape[1]
    if pa_size == 0:
      return z
    if sem_type == 'mlp':
      W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
      W1[np.random.rand(*W1.shape) < 0.5] *= -1
      W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
      W2[np.random.rand(hidden) < 0.5] *= -1
      x = sigmoid(X @ W1) @ W2 + z
    elif sem_type == 'mim':
      w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w1[np.random.rand(pa_size) < 0.5] *= -1
      w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w2[np.random.rand(pa_size) < 0.5] *= -1
      w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w3[np.random.rand(pa_size) < 0.5] *= -1
      x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
    elif sem_type == 'gp':
      gp = GaussianProcessRegressor()
      x = gp.sample_y(X, random_state=None).flatten() + z
    elif sem_type == 'gp-add':
      gp = GaussianProcessRegressor()
      x = sum([
          gp.sample_y(X[:, i, None], random_state=None).flatten()
          for i in range(X.shape[1])
      ]) + z
    else:
      raise ValueError('unknown sem type')
    return x

  def _simulate_single_equaiton_daring(X, scale, hidden=100):
    pa_size = X.shape[1]

    z = _simulate_noise(scale)
    W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
    W1[np.random.rand(*W1.shape) < 0.5] *= -1
    W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
    W2[np.random.rand(hidden) < 0.5] *= -1
    x = sigmoid(X @ W1) @ W2 + z

    return x

  def _simulate_noise(scale):
    if setting == 'mixed_noise':
      scale = 1
      noise_types = ['gauss', 'uniform', 'exp', 'gumbel']
      noise = random.choice(noise_types)
      if noise == 'gauss':
        z = np.random.normal(scale=scale, size=n)
      elif noise == 'uniform':
        z = np.random.uniform(low=-scale, high=scale, size=n)
      elif noise == 'exp':
        z = np.random.exponential(scale=scale, size=n)
      elif noise == 'gumbel':
        z = np.random.gumbel(scale=scale, size=n)
      else:
        raise ValueError('unknown noise type')

    elif setting == 'id_mixed_noise':
      scale = random.choice([1, 2, 3, 4])
      z = np.random.uniform(low=-scale, high=scale, size=n)

    elif setting == 'id_large_noise_var':
      scale = 4
      z = np.random.uniform(low=-scale, high=scale, size=n)

    elif setting == 'non_id_small_noise_var':
      scale = 1
      z = np.random.normal(scale=scale, size=n)

    else:
      z = np.random.normal(scale=scale, size=n)
    return z

  d = B.shape[0]
  scale_vec = noise_scale if noise_scale else np.ones(d)
  X = np.zeros([n, d])
  G = ig.Graph.Adjacency(B.tolist())
  ordered_vertices = G.topological_sorting()
  assert len(ordered_vertices) == d

  for j in ordered_vertices:
    parents = G.neighbors(j, mode=ig.IN)
    X[:, j] = _simulate_single_equaiton_daring(X[:, parents], scale_vec[j])

  return X


def simulate_nonlinear_IRM(B, n, sem_type, noise_scale=None):
  """Simulates samples from nonlinear SEM.

  Args:
    B: [d, d] binary adj matrix of DAG
    n: num of samples
    sem_type: The type of the SEM: mlp, mim, gp, gp-add
    noise_scale: scale parameter of additive noise, default all ones

  Returns:
    X (np.ndarray): [n, d] sample matrix
  """

  def _simulate_single_equation(X, scale, hidden=100):
    z = np.random.normal(scale=scale, size=n)
    pa_size = X.shape[1]
    if pa_size == 0:
      return z
    if sem_type == 'mlp':
      W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
      W1[np.random.rand(*W1.shape) < 0.5] *= -1
      W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
      W2[np.random.rand(hidden) < 0.5] *= -1
      x = sigmoid(X @ W1) @ W2 + z
    elif sem_type == 'mim':
      w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w1[np.random.rand(pa_size) < 0.5] *= -1
      w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w2[np.random.rand(pa_size) < 0.5] *= -1
      w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
      w3[np.random.rand(pa_size) < 0.5] *= -1
      x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
    elif sem_type == 'gp':
      gp = GaussianProcessRegressor()
      x = gp.sample_y(X, random_state=None).flatten() + z
    elif sem_type == 'gp-add':
      gp = GaussianProcessRegressor()
      x = sum([
          gp.sample_y(X[:, i, None], random_state=None).flatten()
          for i in range(X.shape[1])
      ]) + z
    else:
      raise ValueError('unknown sem type')
    return x

  d = B.shape[0]
  scale_vec = noise_scale if noise_scale else np.ones(d)
  X = np.zeros([n, d])
  G = ig.Graph.Adjacency(B.tolist())
  ordered_vertices = G.topological_sorting()
  assert len(ordered_vertices) == d
  for j in ordered_vertices:
    parents = G.neighbors(j, mode=ig.IN)
    X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
  return X


def W_true_Sachs(vertex_label=[
    'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38',
    'Jnk'
]):
  """Returns the ground true dag of Sachs protein data.

  Args:
    vertex_label: the names of the vertices

  Returns:
        g: ig.Graph object, ground truth DAG for the sachs dataset
        W_true: np.array, the ground truth DAG for the sachs dataset in
          adjacency matrix format
  """
  g = ig.Graph(n=11, vertex_attrs={'name': vertex_label}, directed=True)

  # 17 edges in total
  g.add_edges([('Raf', 'Mek'), ('Mek', 'Erk'), ('Plcg', 'PIP2'),
               ('Plcg', 'PIP3'), ('PKC', 'Raf'), ('PKC', 'Mek'), ('PKC', 'Jnk'),
               ('PKC', 'PKA'), ('PKC', 'P38'), ('PIP3', 'PIP2'), ('PKA', 'Raf'),
               ('PKA', 'Mek'), ('PKA', 'Erk'), ('PKA', 'Jnk'), ('PKA', 'P38'),
               ('PKA', 'Akt'), ('Erk', 'Akt')],)

  return g, np.array(g.get_adjacency().data)


def preprocess_sachs():
  """Preprocesses the sachs_txt.data by adding columns names to its data."""

  sachs_data = pd.read_csv(
      '/notears/sachs.data.txt', header=None, delim_whitespace=True)
  sachs_data.columns = [
      'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38',
      'Jnk'
  ]
  sachs_data.to_csv('sachs.csv', index=None)


def read_sachs(folder_path, format='np'):
  """Reads all Sachs source data.

  Args:
    folder_path: path of the folder
    format: 'np': return in np array 'df': return in panda dataframe format

  Returns:
    A tensor of size [number of all sample, num of features] ex: (11672, 11)
  """
  sachs_data = list()
  for file in glob.glob(f'{folder_path}*.xls'):
    sachs_df = pd.read_excel(file)
    if format == 'np':
      sachs_data.append(sachs_df.to_numpy())

  return np.vstack(sachs_data)


def filter_edge_by_weight(W, low, high):
  """Filters W to abtain the dag with number of edges in [low, high].

  Args:
    W: the dag in the adjacency matrix format
    low: minimin number of edges the dag should have
    high: max number of edges the dag can have

  Returns:
    W: the dag in the adjacency matrix format with  low < num_edges < high
  """
  W_range = np.arange(np.min(W), np.max(W), np.max(W) / 10)
  w_threshold = 0

  if np.count_nonzero(W) < low:
    print(f'W has less than {low} edges')
    return W

  for w_step in W_range:
    w_threshold = w_step
    W_est = copy.deepcopy(W)
    W_est[np.abs(W_est) < w_threshold] = 0
    num_edges = np.count_nonzero(W_est)

    if num_edges < low:
      filter_edge_by_weight(W_est, low, w_threshold)
    elif num_edges >= low and num_edges <= high:
      return W_est

    else:
      continue

  return W_est


def filter_W(W, low, high, W_threshold_init=0.5, step_size=0.05):
  """Filters W to abtain the dag with number of edges in [low, high]."""

  w_threshold = W_threshold_init

  num_edges = np.count_nonzero(W)
  if num_edges < low:
    print(f'W has less than {low} edges')
    return W

  if num_edges < high:
    return W

  while num_edges > high:
    w_threshold += step_size
    W_est = copy.deepcopy(W)
    W_est[np.abs(W_est) < w_threshold] = 0
    num_edges = np.count_nonzero(W_est)

    if w_threshold > 30:
      print(f'Could not find proper DAG with step_size {step_size}')
      return W

  return w_threshold, W_est


def rank_W(W,
           low,
           high,
           B_true,
           W_threshold_low=0,
           W_threshold_high=10,
           step_size=0.05):
  """Filters W to abtain the dag with number of edges in [low, high].

  Args:
    W: the dag in the adjacency matrix format
    low: minimin number of edges the dag should have
    high: max number of edges the dag can have
    B_true: ground truth of DAG
    W_threshold_low: lower threshold for W
    W_threshold_high: higher threshold for W
    step_size: increase w_threshold by step_size at each iteration

  Returns:
    w_threshold: the threhold found used to filter W
    W: the dag in the adjacency matrix format with  low < num_edges < high
  """

  wthresh_W_shd_dict = {}
  for w_threshold in np.arange(W_threshold_low, W_threshold_high, step_size):

    W_est = copy.deepcopy(W)
    W_est[np.abs(W_est) < w_threshold] = 0
    num_edges = np.count_nonzero(W_est)

    if num_edges < low:
      print(f'W has less than {low} edges')
      # wthresh_W_shd_dict[w_threshold] = (
      #  W_est, count_accuracy(B_true.T, W_est !=0)['shd'])
      break

    if num_edges <= high:
      if is_dag(W_est):
        wthresh_W_shd_dict[w_threshold] = (W_est,
                                           count_accuracy(B_true,
                                                          W_est != 0)['shd'])

  wthresh_W_shd_dict = {
      k: v for k, v in sorted(
          wthresh_W_shd_dict.items(), key=lambda item: item[1][1])
  }
  return wthresh_W_shd_dict


def rank_W_noGT(W,
                low,
                high,
                W_threshold_low=0,
                W_threshold_high=10,
                step_size=0.05):
  """Filters W to abtain the dag with number of edges in [low, high].

  Args:
    W: the dag in the adjacency matrix format
    low: minimin number of edges the dag should have
    high: max number of edges the dag can have
    W_threshold_low: lower threshold for W
    W_threshold_high: higher threshold for W
    step_size: increase w_threshold by step_size at each iteration

  Returns:
    w_threshold: the threhold found used to filter W
    W: the dag in the adjacency matrix format with  low < num_edges < high
  """

  wthresh_W_shd_dict = {}
  for w_threshold in np.arange(W_threshold_low, W_threshold_high, step_size):

    W_est = copy.deepcopy(W)
    W_est[np.abs(W_est) < w_threshold] = 0
    num_edges = np.count_nonzero(W_est)

    if num_edges < low:
      print(f'W has less than {low} edges')
      break

    if num_edges <= high:
      if is_dag(W_est):
        wthresh_W_shd_dict[w_threshold] = (W_est,
                                           count_accuracy(B_true,
                                                          W_est != 0)['shd'])

  wthresh_W_shd_dict = {
      k: v for k, v in sorted(
          wthresh_W_shd_dict.items(), key=lambda item: item[1][1])
  }
  return wthresh_W_shd_dict


def remove_outliers(X, fill=False):
  """Removes samples that contain features identified as outliers.

  Args:
    X: the dag in the adjacency matrix format
    fill: boolean suggesting whether to interpolate the removed outliers

  Returns:
    numpy array of the samples with outliers removed
  """
  columns = [
      'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38',
      'Jnk'
  ]
  # X = pd.DataFrame(X, columns=columns)
  scaler = preprocessing.StandardScaler().fit(X)
  for index, mean in enumerate(scaler.mean_):  # go over each variable
    std = scaler.scale_[index]
    upper_bound = mean + 3 * std
    lower_bound = mean - 3 * std
    mask = [(x >= lower_bound and x <= upper_bound) for x in X[:, index]]
    X = X[mask, :]

  return X


def preprocess_labeled_data(X, Y, normalization):
  """Different approaches for preprocessing the data."""

  if normalization:
    if normalization == 'standard':
      scaler = preprocessing.StandardScaler().fit(X)
      X = scaler.fit_transform(X)

    elif normalization == 'minmax':
      X = remove_outliers(X)
      scaler = preprocessing.MinMaxScaler()
      X = scaler.fit_transform(X)

    elif normalization == 'removeOutlier':
      scaler = preprocessing.StandardScaler().fit(X)
      for index, mean in enumerate(scaler.mean_):  # go over each variable
        std = scaler.scale_[index]
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std
        mask = [(x >= lower_bound and x <= upper_bound) for x in X[:, index]]
        X = X[mask, :]
        Y = Y[mask]
      scaler = preprocessing.StandardScaler().fit(X)
      X = scaler.fit_transform(X)

  return X, Y


def preprocess(X, normalization):
  """Different approaches for preprocessing the data."""

  if normalization:
    if normalization == 'standard':
      scaler = preprocessing.StandardScaler().fit(X)
      X = scaler.fit_transform(X)

    elif normalization == 'minmax':
      X = remove_outliers(X)
      scaler = preprocessing.MinMaxScaler()
      X = scaler.fit_transform(X)

    elif normalization == 'removeOutlier':
      X = remove_outliers(X)
      scaler = preprocessing.StandardScaler().fit(X)
      X = scaler.fit_transform(X)

    else:
      return X
  return X


def distribute_to_envs(X_cluster):
  """Aggregates sample data into different clusters.

  Args:
    X_cluster: a dict containing the key as the cluster_index and value

  Returns:
    A tensor of shape [envs, sampels, features]
  """
  X_envs = list()
  for i, X in X_cluster.items():
    X_envs.append(X)

  return np.asanyarray(X_envs)


def read_sachs_to_envs(folder_path,
                       num_envs,
                       normalization,
                       format='np'):
  """Reads all Sachs source datasets.

  Args:
    folder_path: path of the folder.
    num_envs: the number of envs to read the data
    normalization: normalization type
    format: 'np': return in np array 'df': return in panda dataframe format

  Returns:
    A tensor of shape [num_envs, number of sample in each envs, num of
    features]
  """
  sachs_data = list()
  if num_envs == 14:
    Y_label = []
    for i, file in enumerate(glob.glob(f'{folder_path}*.xls')):
      sachs_df = pd.read_excel(file)
      sachs_array = sachs_df.to_numpy()
      sachs_array = preprocess(sachs_array, normalization=args.normalization)
      sachs_data.append(sachs_array)
      Y_label.append(np.ones(sachs_array.shape[0]) * i)

    sachs_data_envs = np.vstack(sachs_data)
    sachs_data_labels = np.hstack(Y_label)
    X_res, Y_res = over_sampling(sachs_data_envs, sachs_data_labels)
    X_cluster = classify_x(X_res, Y_res)
    X_envs = distribute_to_envs(X_cluster)

  elif num_envs == 2:

    X_envs = [None] * 2
    Y_label = [None] * 2
    for i, file in enumerate(glob.glob(f'{folder_path}*.xls')):
      start_index = file.index('sachs_data/') + 11
      end_index = file.index(' ') - 1
      file_index = int(file[start_index:end_index])
      label = 0 if file_index <= 9 else 1

      sachs_df = pd.read_excel(file)
      sachs_array = sachs_df.to_numpy()
      if X_envs[label] is None:
        X_envs[label] = sachs_array
        Y_label[label] = np.ones(sachs_array.shape[0]) * label
      else:
        X_envs[label] = np.concatenate((X_envs[label], sachs_array), axis=0)
        Y_label[label] = np.concatenate(
            (Y_label[label], (np.ones(sachs_array.shape[0]) * label)), axis=0)

    for i in range(num_envs):
      X_envs[i], Y_label[i] = preprocess_labeled_data(X_envs[i], Y_label[i],
                                                      normalization)

    X = np.vstack(X_envs)
    Y = np.hstack(Y_label)
    X_res, Y_res = over_sampling(X, Y)
    X_cluster = classify_x(X_res, Y_res)
    X_envs = distribute_to_envs(X_cluster)

  elif num_envs == 3:
    for file in glob.glob(f'./data/cluster/*3.csv'):
      sachs_array = np.loadtxt(file, delimiter=',')
      sachs_data.append(sachs_array)
    X_envs = np.stack(sachs_data)

  elif num_envs == 6:
    for file in glob.glob(f'./data/cluster/*6.csv'):
      sachs_array = np.loadtxt(file, delimiter=',')
      sachs_array = preprocess(sachs_array, normalization)
      sachs_data.append(sachs_array)
    X_envs = np.stack(sachs_data)

  elif num_envs == 7:

    X_envs = [None] * num_envs
    Y_label = [None] * num_envs
    for i, file in enumerate(glob.glob(f'{folder_path}*.xls')):
      start_index = file.index('sachs_data/') + 11
      end_index = file.index(' ') - 1
      file_index = int(file[start_index:end_index])
      label = file_index % num_envs

      sachs_df = pd.read_excel(file)
      sachs_array = sachs_df.to_numpy()

      if X_envs[label] is None:
        X_envs[label] = sachs_array
        Y_label[label] = np.ones(sachs_array.shape[0]) * label
      else:
        X_envs[label] = np.concatenate((X_envs[label], sachs_array), axis=0)
        Y_label[label] = np.concatenate(
            (Y_label[label], (np.ones(sachs_array.shape[0]) * label)), axis=0)

    for i in range(num_envs):
      # X_envs[i] = np.vstack(X_envs[i])
      X_envs[i], Y_label[i] = preprocess_labeled_data(X_envs[i], Y_label[i],
                                                      normalization)

    X = np.vstack(X_envs)
    Y = np.hstack(Y_label)
    X_res, Y_res = over_sampling(X, Y)
    X_cluster = classify_x(X_res, Y_res)
    X_envs = distribute_to_envs(X_cluster)

  return np.vstack(X_envs)


def BH_norm_test():
  """BH normalization test."""

  # evaluate model on training dataset with outliers removed
  # load the dataset
  url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
  df = pd.read_csv(url, header=None)
  # retrieve the array
  data = df.values
  # split into inpiut and output elements
  X, y = data[:, :-1], data[:, -1]
  # split into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.33, random_state=1)
  # identify outliers in the training dataset
  lof = LocalOutlierFactor()
  yhat = lof.fit_predict(X_train)
  # select all rows that are not outliers
  mask = yhat != -1
  X_train, y_train = X_train[mask, :], y_train[mask]
  # summarize the shape of the updated training dataset
  print(X_train.shape, y_train.shape)
  # fit the model
  model = LinearRegression()
  model.fit(X_train, y_train)
  # evaluate the model
  yhat = model.predict(X_test)
  # evaluate predictions
  mae = mean_absolute_error(y_test, yhat)
  print('MAE: %.3f' % mae)


def W_true_insurance():
  """Returns the ground true DAG for Insurance data."""

  # The output s are ig.graph object of ground truth dag of insurance dataset
  # and grouth truth dag of the insurance dataset in adjacency matrix format.

  vertex_label = [
      'PropCost', 'GoodStudent', 'Age', 'SocioEcon', 'RiskAversion',
      'VehicleYear', 'ThisCarDam', 'RuggedAuto', 'Accident', 'MakeModel',
      'DrivQuality', 'Mileage', 'Antilock', 'DrivingSkill', 'SeniorTrain',
      'ThisCarCost', 'Theft', 'CarValue', 'HomeBase', 'AntiTheft',
      'OtherCarCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost',
      'DrivHist'
  ]

  g = ig.Graph(n=27, vertex_attrs={'name': vertex_label}, directed=True)

  # 52 edges in total
  g.add_edges([
      ('Age', 'SocioEcon'),
      ('Age', 'GoodStudent'),
      ('Age', 'SeniorTrain'),
      ('Age', 'DrivingSkill'),
      ('Age', 'MedCost'),
      ('SocioEcon', 'AntiTheft'),
      ('SocioEcon', 'HomeBase'),
      ('SocioEcon', 'OtherCar'),
      ('SocioEcon', 'RiskAversion'),
      ('SocioEcon', 'GoodStudent'),
      ('SocioEcon', 'MakeModel'),
      ('SocioEcon', 'VehicleYear'),
      ('RiskAversion', 'AntiTheft'),
      ('RiskAversion', 'HomeBase'),
      ('RiskAversion', 'VehicleYear'),
      ('RiskAversion', 'MakeModel'),
      ('RiskAversion', 'DrivQuality'),
      ('RiskAversion', 'DrivHist'),
      ('RiskAversion', 'SeniorTrain'),
      ('SeniorTrain', 'DrivingSkill'),
      ('DrivingSkill', 'DrivHist'),
      ('DrivingSkill', 'DrivQuality'),
      ('AntiTheft', 'Theft'),
      ('HomeBase', 'Theft'),
      ('VehicleYear', 'CarValue'),
      ('VehicleYear', 'RuggedAuto'),
      ('VehicleYear', 'Antilock'),
      ('VehicleYear', 'Airbag'),
      ('MakeModel', 'CarValue'),
      ('MakeModel', 'RuggedAuto'),
      ('MakeModel', 'Antilock'),
      ('MakeModel', 'Airbag'),
      ('DrivQuality', 'Accident'),
      ('CarValue', 'Theft'),
      ('CarValue', 'ThisCarCost'),
      ('RuggedAuto', 'ThisCarDam'),
      ('RuggedAuto', 'ThisCarCost'),
      ('RuggedAuto', 'OtherCarCost'),
      ('RuggedAuto', 'Cushioning'),
      ('Mileage', 'CarValue'),
      ('Mileage', 'Accident'),
      ('Antilock', 'Accident'),
      ('Airbag', 'Cushioning'),
      ('Accident', 'MedCost'),
      ('Accident', 'ILiCost'),
      ('Accident', 'OtherCarCost'),
      ('Accident', 'ThisCarDam'),
      ('Cushioning', 'MedCost'),
      ('Theft', 'ThisCarCost'),
      ('ThisCarDam', 'ThisCarCost'),
      ('ThisCarCost', 'PropCost'),
      ('OtherCarCost', 'PropCost'),
  ])
  assert g.vcount() == 27 and g.ecount() == 52
  return g, np.array(g.get_adjacency().data)


def swap(arr, start_index, last_index):
  """Switches column of  start_index and last_index."""
  arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]


def swap_row_column(arr, start_index, last_index):
  """Switches both row and column start_index and last_index."""
  arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]
  arr[[start_index, last_index], :] = arr[[last_index, start_index], :]


def classify_x(X, labeled_X, labels=None):
  """Aggregates sample data into different clusters.

  Args:
    X: sample data in numpy array
    labeled_X: the label for each sample in X
    labels: all labels

  Returns:
    A dict with key as cluster index and value as the numpy array of the
    samples classified key
  """

  X_cluster = dict()
  for i in range(X.shape[0]):
    if labeled_X[i] in X_cluster:
      X_cluster[labeled_X[i]].append(X[i])
    else:
      X_cluster[labeled_X[i]] = [X[i]]

  for k in X_cluster.keys():
    X_cluster[k] = np.array(X_cluster[k])

  return X_cluster


def over_sampling(X, y):
  """Oversampling the input data and its label using random sampling.

  Args:
    X: input data as numpy array
    y: the label for each input data X

  Returns:
    Upsampled input data and its label
  """

  ros = RandomOverSampler(random_state=42)
  X_res, y_res = ros.fit_resample(X, y)

  return X_res, y_res


def load_BH():
  """Loads boston housing dataset and preprocess the data."""

  data_url = 'http://lib.stat.cmu.edu/datasets/boston'
  raw_df = pd.read_csv(data_url, sep='\s+', skiprows=22, header=None)
  data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  target = raw_df.values[1::2, 2]
  return np.concatenate([target[:, np.newaxis], data], axis=1)


def cluster_dataset(save_path, dataset, cluster):
  """Clusters the data and prerpocess after.

  Args:
    save_path: the path to save the proprocessed data
    dataset: str, specifying the type of dataset to use
    cluster: number of clusters to cluster the dataset
  """

  np.set_printoptions(precision=3)

  if dataset == 'sachs':
    X_raw = read_sachs('./data/sachs_data/')
  elif dataset == 'BH':
    X_raw = load_BH()

  label = np.ones(len(X_raw))
  X, X_test, _, _ = train_test_split(
      X_raw, label, test_size=0.1, random_state=42)
  # X_train_val, X_test, _, _ = train_test_split(
  #   X_raw, LABEL, test_size=0.1, random_state=42)
  # X, X_val, _, _ = train_test_split(
  #   X_train_val, np.ones(len(X_train_val)), test_size=0.1, random_state=1)

  _, d = X.shape

  kmeans = KMeans(n_clusters=cluster, max_iter=1000).fit(X)
  labeled_X = kmeans.fit_predict(X)
  print(labeled_X.shape)
  X_cluster = classify_x(X, labeled_X)  # X_cluster is a dict

  os.makedirs(save_path, exist_ok=True)
  # save the BH preprocesssed by standardization
  X_upsampled, y_upsampled = over_sampling(X, labeled_X)
  X_envs = classify_x(X_upsampled, y_upsampled)
  standard_scaler = preprocessing.StandardScaler()
  minmax_scaler = preprocessing.MinMaxScaler()

  i = 1
  for X_env in X_envs.items():
    standardX = standard_scaler.fit_transform(X_env[1])
    exp = save_path + f'standard_BH_env_{i}_{cluster}.csv'
    np.savetxt(exp, standardX, fmt='%.3f', delimiter=',')
    i += 1

  # for train
  standard_train_param = standard_scaler.fit(X)
  standardXtrain = standard_scaler.transform(X)
  X_trainexp = save_path + f'standard_BH_train.csv'
  np.savetxt(X_trainexp, standardXtrain, fmt='%.3f', delimiter=',')

  # for test
  standardXtest = standard_scaler.transform(X_test)
  X_testexp = save_path + f'standard_BH_test.csv'
  np.savetxt(X_testexp, standardXtest, fmt='%.3f', delimiter=',')

  # if val exist, for val:
  # standardX_val = standard_scaler.transform(X_val)
  # X_valexp = args.save_path + f'standard_BH_val.csv'
  # np.savetxt(X_valexp, standardX_val, fmt='%.3f', delimiter=',')


def load_data(dataset_name, preprocess, num_envs):
  """Loads data based on name and normalization."""

  data_root = './data/{}'.format(dataset_name)

  # load X_envs (an np.array with size [num_envs, num_samples, data_dimension])
  X_envs_list = []
  for i in range(num_envs):
    # load data for each environment
    env_path = os.path.join(
        data_root, '{}_{}_env_{}_{}.csv'.format(preprocess, dataset_name, i + 1,
                                                num_envs))
    env_data = np.loadtxt(env_path, delimiter=',')
    X_envs_list.append(env_data)
  X_envs = np.stack(X_envs_list, axis=0)
  # load X_test (an np.array with size [num_samples, data_dimension])
  X_test_path = os.path.join(data_root,
                             '{}_{}_test.csv'.format(preprocess, dataset_name))
  X_test = np.loadtxt(X_test_path, delimiter=',')

  return X_envs, X_test


def load_self_supervised_data(dataset_name, preprocess):
  """Loads data based on name and normalization."""

  data_root = './data/{}'.format(dataset_name)

  X_test_path = os.path.join(data_root,
                             '{}_{}_test.csv'.format(preprocess, dataset_name))
  X_test = np.loadtxt(X_test_path, delimiter=',')
  X_train_path = os.path.join(
      data_root, '{}_{}_train.csv'.format(preprocess, dataset_name))
  X_train = np.loadtxt(X_train_path, delimiter=',')
  X = np.vstack([X_train, X_test])

  return X


def load_vertex_label(dataset_name):
  """Returns vertex label for each dataset."""

  if dataset_name == 'BH':
    vertex_label = [
        'Y_MEDV', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
  elif dataset_name == 'Insurance':
    vertex_label = [
        'PropCost', 'GoodStudent', 'Age', 'SocioEcon', 'RiskAversion',
        'VehicleYear', 'ThisCarDam', 'RuggedAuto', 'Accident', 'MakeModel',
        'DrivQuality', 'Mileage', 'Antilock', 'DrivingSkill', 'SeniorTrain',
        'ThisCarCost', 'Theft', 'CarValue', 'HomeBase', 'AntiTheft',
        'OtherCarCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag',
        'ILiCost', 'DrivHist'
    ]
  elif dataset_name == 'Sachs':
    vertex_label = [
        'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38',
        'Jnk'
    ]
  else:
    print('unrecognized dataset!')

  return vertex_label


def load_W_true(dataset_name):
  """Returns ground truth DAG for each dataset."""

  if dataset_name == 'BH':
    W_true = np.zeros(W_orig.shape)
    W_true[0][1, 6, 8, 13] = 1
    g = None
  elif dataset_name == 'Insurance':
    g, W_true = utils.W_true_insurance()
  elif dataset_name == 'Sachs':
    g, W_true = utils.W_true_Sachs()
  else:
    print('unrecognized dataset!')

  return vertex_label


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(description='cluster dataset algorithm')
  parser.add_argument(
      '--normalization', type=str, default='removeOutlier', help='')
  parser.add_argument(
      '--folder_path',
      type=str,
      default='sachs',
      help='which data set to cluster')
  parser.add_argument(
      '--cluster', type=int, default=3, help='number of clusters')

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  g, g_list = W_true_insurance()

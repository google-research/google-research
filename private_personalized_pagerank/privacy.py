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

"""Differentially Private PPR library."""

from typing import Collection

import numpy as np
import scipy.sparse as sps

from private_personalized_pagerank import ppr


def ppr_pushflow_unbounded_noise(
    size, alpha, epsilon
):
  """DP noise generation for the PushFlow PPR algorithm."""
  return np.random.laplace(0, 2.0 * (1 - alpha) / (epsilon * alpha), size=size)


def ppr_capping_noise(*, size, epsilon, sigma):
  """DP noise generation for the PushFlow PPR algorithm with capping."""
  return np.random.laplace(
      0,
      sigma / epsilon,
      size=size,
  )


def dp_ppr_total_capping(
    adjacency,
    source_node,
    *,
    sigma,
    num_iter = 10,
    alpha = 0.15,
    is_joint_dp,
    epsilon,
):
  """DP PPR computation using the modified push flow method with flow capping.

  Args:
    adjacency: sps.matrix encoding the adjacency matrix of a undirected graph.
    source_node: the source node from which PPR walks are started.
    sigma: the target sensitivty desired.
    num_iter: max number of iterations for the power iteration method.
    alpha: the jump (a.k.a. stop) probability. I.e., the probability the walk
      restarts at the source.
    is_joint_dp: If `True`, use joint DP definition from the paper otherwise it
      is standard DP.
    epsilon: DP parameter.

  Returns:
    A post-processed DP PPR vector.
  """
  capped_ppr = ppr.ppr_total_capping(
      adjacency,
      source_node,
      sigma=sigma,
      num_iter=num_iter,
      alpha=alpha,
      is_joint_dp=is_joint_dp,
  )
  dp_ppr = capped_ppr + ppr_capping_noise(
      size=capped_ppr.shape, epsilon=epsilon, sigma=sigma
  )
  dp_ppr = ppr.postprocess_dp_ppr(dp_ppr)
  return dp_ppr


def dp_edge_flipping(adjacency, epsilon):
  """DP random edge flipping algorithm."""
  # Creasting a u.a.r. symmetric matrix of {0,1}
  n_nodes = len(adjacency)
  random_flip = np.random.uniform(0, 1, (n_nodes, n_nodes))
  random_flip = (random_flip - np.tril(random_flip)).T + (
      random_flip - np.tril(random_flip)
  )
  random_flip[random_flip > 0.5] = 1
  random_flip[random_flip <= 0.5] = 0

  # Probability of edge flipping.
  p = 2 / (1 + np.exp(epsilon))
  random_positions = np.random.uniform(0, 1, (n_nodes, n_nodes))
  random_positions = (random_positions - np.tril(random_positions)).T + (
      random_positions - np.tril(random_positions)
  )
  # Positions for flipping.
  random_positions = random_positions <= p
  dp_adjacency = adjacency.copy()
  dp_adjacency[random_positions] = random_flip[random_positions]

  return dp_adjacency


def compute_dp_ppr(
    adjacency,
    source_node,
    *,
    sigma,
    num_iter = 10,
    alpha = 0.15,
    is_joint_dp,
    epsilon,
):
  """DP PPR computation using the modified push flow method and noise.

  Args:
    adjacency: sps.matrix encoding the adjacency matrix of a undirected graph.
    source_node: the source node from which PPR walks are started.
    sigma: the target sensitivty desired.
    num_iter: max number of iterations for the power iteration method.
    alpha: the jump (a.k.a. stop) probability. I.e., the robability the walk
      restarts at the source.
    is_joint_dp: If `True`, use joint DP definition from the paper otherwise it
      is standard DP.
    epsilon: DP parameter.

  Returns:
    The DP PPR vector.
  """
  clean_ppr = ppr.ppr_total_capping(
      adjacency,
      source_node,
      sigma=sigma,
      alpha=alpha,
      is_joint_dp=is_joint_dp,
      num_iter=num_iter,
  )
  dp_ppr = clean_ppr + ppr_capping_noise(
      size=clean_ppr.shape, epsilon=epsilon, sigma=sigma
  )
  dp_ppr = ppr.postprocess_dp_ppr(dp_ppr)
  return dp_ppr

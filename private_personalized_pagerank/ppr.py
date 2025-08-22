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

"""Implementation of the Personalized PageRank algorithm."""

from absl import logging
import numpy as np
import scipy.sparse as sps


def ppr_power_iteration(
    adjacency,
    source_node,
    *,
    num_iter = 100,
    alpha = 0.15,
    tolerance = 1e-6,
):
  """Implementation of the (non-private) power iteration method for PPR.

  Args:
    adjacency: sps.matrix encoding the adjacency matrix of a undirected graph.
    source_node: the source node from which PPR walks are started.
    num_iter: max number of iterations for the power iteration method.
    alpha: the jump (a.k.a. stop) probability. I.e., the probability the walk
      restarts at the source.
    tolerance: the accuracy required.

  Returns:
    PPR vector as an n-dimensional vector.
  """
  n = adjacency.shape[0]
  degrees = adjacency.sum(axis=1).A1
  if all(degrees > 0):
    logging.warning('Graph should not contain isolated nodes.')
  degrees[degrees == 0] = 1
  ppr = np.ones(n) / n
  for _ in range(num_iter):
    ppr_tmp = ppr
    ppr = (
        (1 - alpha) * ppr / degrees @ adjacency
    )  # @ stands for vector-matrix multiplication.
    ppr[source_node] += alpha
    error = np.linalg.norm(ppr - ppr_tmp, 1)
    if error < tolerance:
      break
  return ppr


def ppr_pushflow_unbounded(
    adjacency,
    source_node,
    *,
    num_iter = 10,
    alpha = 0.15,
    return_residual = False,
):
  """Implementation of the (non-private) push flow method for PPR.

  Args:
    adjacency: sps.matrix encoding the adjacency matrix of a undirected graph.
    source_node: the source node from which PPR walks are started.
    num_iter: number of iterations the push flow method is applied on all nodes.
    alpha: the jump (a.k.a. stop) probability. I.e., the probability the walk
      restarts at the source.
    return_residual: If `True`, returns both the PPR vector and the residual.

  Returns:
    PPR vector as an n-dimensional vector and (if return_residual is true) also
    the residual vector.
  """
  alpha = convert_alpha(alpha)
  n = adjacency.shape[0]
  ppr = np.zeros(n)
  residual = np.zeros(n)
  residual[source_node] = 1
  degrees = adjacency.sum(
      axis=1
  ).A1  # Compute node degrees and store them contiguously.

  for _ in range(num_iter):
    ppr_tmp = ppr + alpha * residual  # `ppr_tmp` is a copy.
    residual_tmp = (1 - alpha) / 2.0 * residual  # `residual_tmp` is a copy.
    for node in range(n):
      neighors = adjacency[node].nonzero()[1]
      assert len(neighors) == degrees[node], 'Internal degree mismatch!'
      # Update the residual of neighboring nodes.
      residual_tmp[neighors] += (
          (1 - alpha) / 2.0 * residual[node] / degrees[node]
      )
    ppr = ppr_tmp
    residual = residual_tmp
  if return_residual:
    return ppr, residual
  return ppr


def ppr_total_capping(
    adjacency,
    source_node,
    *,
    sigma,
    num_iter = 10,
    alpha = 0.15,
    is_joint_dp,
    return_residual = False,
    heuristic_prepush = False,
):
  """PPR computation using the modified push flow method with flow capping.

  Args:
    adjacency: sps.matrix encoding the adjacency matrix of a undirected graph.
    source_node: the source node from which PPR walks are started.
    sigma: the target sensitivty desired.
    num_iter: max number of iterations for the power iteration method.
    alpha: the jump (a.k.a. stop) probability. I.e., the probability the walk
      restarts at the source.
    is_joint_dp: If `True`, use joint DP definition from the paper otherwise it
      is standard DP.
    return_residual: If `True`, returns both the PPR vector and the residual.
    heuristic_prepush: If `True`, use the heuristic described in the paper.

  Returns:
    A post-processed approximate PPR vector.
  """
  n = adjacency.shape[0]
  ppr = np.zeros(n)
  residual = np.zeros(n)
  residual[source_node] = 1

  if heuristic_prepush:
    residual[source_node] = 0
    ppr[source_node] = alpha
    neighbors = adjacency[source_node].nonzero()[1]
    seed_degree = len(neighbors)
    for neighbor in neighbors:
      ppr[neighbor] = alpha * (1 - alpha) / seed_degree
      residual[neighbor] = (1 - alpha) * (1 - alpha) / seed_degree

  alpha = convert_alpha(alpha)
  pushed_residual = np.zeros(n)
  degrees = adjacency.sum(1).A1

  threshold = np.ones(n) * sigma / ((3 - alpha) * (1 - (1 - alpha) ** num_iter))

  if is_joint_dp:
    threshold[source_node] = 1.0 / alpha

  for _ in range(num_iter):
    flow_to_push = residual.copy()
    cap = degrees * threshold - pushed_residual
    flow_to_push[cap < flow_to_push] = cap[cap < flow_to_push]
    pushed_residual += flow_to_push
    residual -= flow_to_push

    ppr += alpha * flow_to_push
    residual += (1 - alpha) / 2.0 * flow_to_push

    for node in range(n):
      if flow_to_push[node] > 0:
        neighors = adjacency[node].nonzero()[1]
        assert len(neighors) == degrees[node], 'Internal degree mismatch!'
        residual[neighors] += (
            (1 - alpha) / 2.0 * flow_to_push[node] / degrees[node]
        )

  if return_residual:
    return ppr, residual
  return ppr


def convert_alpha(alpha):
  """Convert the alpha parameter from normal to lazy random walk.

  Args:
    alpha: the jump (a.k.a. stop) probability. I.e., the probability the walk
      restarts at the source in the normal random walk.

  Returns:
    The alpha in the lazy probability that remains with 50% probability,
    at each step, at the current node.
  """
  return alpha / (2 - alpha)


def postprocess_dp_ppr(
    pprs, *, renormalize = False
):
  """Truncate the noisy values of PPR to make it a probability vector.

  This is can be used on DP outputs for the PPR vector as DP noise can make it
  incompatible with a probability distribution (e.g., negative values, not
  summing to 1).

  Args:
    pprs: a noisy PPR vector.
    renormalize: If `True`, normalize the vector to sum to 1. Not recommended.

  Returns:
    A post-processed PPR vector. This post processing on DP ouputs does not
    affect DP.
  """
  pprs[pprs < 0] = 0
  pprs[pprs > 1] = 1
  if renormalize:
    pprs /= pprs.sum()
  return pprs

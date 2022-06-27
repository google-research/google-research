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

"""Classes for extracting a subgraph of a larger graph_api.GraphAPI.

This module implements a differentiable L1-regularized PageRank.
It follows the method in Fountoulakis et al., 2016.
`https://arxiv.org/abs/1602.01886`.

The main problem can be written as a L1 penalized quadratic problem (when the
graph is undirected), which we solve using proximal algorithms.
"""

from typing import Optional, Tuple

from flax import struct
import flax.linen as nn

import jax
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

from jaxsel._src import agents
from jaxsel._src import graph_api


@struct.dataclass
class ExtractorConfig:
  """Config for subgraph extractors.

  Attributes:
    max_graph_size: Maximum size of the handled graphs.
    max_subgraph_size: Maximum size of the extracted subgraph.
    rho: L1 penalty strength
    alpha: probability to teleport back to the start node during the walk.
    num_steps: max number of steps in the ISTA algorithm/in the random walk.
    ridge: scale of ridge penalty for the backwards linear problem.
    agent_config: Configuration for the underlying agent model.
  """
  # TODO(gnegiar): Take agent class/name as argument here.
  max_graph_size: int
  max_subgraph_size: int
  rho: float
  alpha: float
  num_steps: int
  ridge: float
  agent_config: agents.AgentConfig


class SparseISTAExtractor(nn.Module):
  """Performs sparse PageRank.

  PageRank can be seen as a random walk on the graph, where the transition
  probabilities are given by the adjacency matrix on the graph.
  Here, the adjacency matrix is parametrized by an `agent` model. PageRank
  yields a vector of weights on the nodes of the graph.

  Reference: https://arxiv.org/abs/1602.01886

  When rho is 0., we still perform topk thresholding on the weight vector.

  Attributes:
    config: Configuration for the extractor layer and the underlying
      agent model.
  """
  config: ExtractorConfig

  def setup(self):
    self.agent = agents.SimpleFiLMedAgentModel(self.config.agent_config)

  def _s(self, start_node_id):
    """Encode the start node."""
    nse = self.config.max_subgraph_size
    return jsparse.BCOO(
        (jnp.zeros(nse).at[0].set(1.), jnp.zeros(
            (nse, 1), dtype=int).at[0, 0].set(start_node_id)),
        shape=(self.config.max_graph_size,))

  def _q_minus_grad(self, q, adjacency_matrix,
                    s):
    return _sum_with_nse(
        (1 - self.config.alpha) * adjacency_matrix.T @ q,
        -self.config.alpha * s,
        nse=self.config.max_subgraph_size)

  def _sparse_softthresh(self, x):
    return jsparse.BCOO(
        (_softthresh(x.data, self.config.alpha, self.config.rho), x.indices),
        shape=x.shape)

  def _ista_step(self, q, adjacency_matrix,
                 s):
    q_minus_grad = self._q_minus_grad(q, adjacency_matrix, s)
    return self._sparse_softthresh(q_minus_grad)

  def _error(self, q, dense_adjacency_matrix,
             s):
    return (_dense_fixed_point(q, dense_adjacency_matrix, s, self.config.alpha,
                               self.config.rho)**2).sum()

  def _extract_dense_submatrix(self, sp_mat,
                               indices):
    """Extracts a dense submatrix of a sparse square matrix at given indices.

    Args:
      sp_mat: A sparse matrix.
      indices: A 1D array of indices into M_sp, or `M_sp.shape[0]` to indicate
        an empty row/column. Assumed to be deduplicated using
        `BCOO.sum_duplicates` or similar beforehand.

    Returns:
      dense_submat:
        A dense submatrix of M_sp, with a subset of rows and columns from M_sp
        or with zero rows and columns in place of -1 indices.
    """
    if sp_mat.ndim != 2:
      raise ValueError(
          f"The first argument must be a 2d matrix. Got {sp_mat.ndim}.")
    if sp_mat.shape[0] != sp_mat.shape[1]:
      raise ValueError(f"sp_mat should be square. Got shape {sp_mat.shape}.")
    if indices.ndim != 1:
      raise ValueError(
          f"indices should be a 1-d array. Got shape {indices.shape}.")

    n_indices = indices.shape[0]
    if n_indices > self.config.max_subgraph_size:
      raise ValueError(f"indices should be smaller than the max_subgraph_size."
                       f"Got shape {indices.shape}.")

    submat_indices = jnp.arange(n_indices)
    i_j = _dstack_product(submat_indices, submat_indices)
    values_to_extract = jax.vmap(_subscript, (None, 0))(sp_mat, indices[i_j])
    dense_submat = values_to_extract.reshape((n_indices, n_indices))
    return dense_submat

  def _make_dense_vector(self, q, indices):
    """Extracts a dense subvector from a sparse vector at given indices."""
    return jax.vmap(_subscript, (None, 0))(q, indices)

  def _ista_solve(self, s,
                  graph):
    """Runs the ISTA solver."""

    def body_fun(mdl, carry):
      step, q = carry
      adjacency_matrix = mdl.agent.fill_sparse_adjacency_matrix(q, graph)
      q = self._ista_step(q, adjacency_matrix, s)
      return step + 1, q

    def cond_fun(mdl, c):
      del mdl
      step, q = c
      del q
      return step < self.config.num_steps

    # Make sure the agent is initialized
    if self.is_mutable_collection("params"):
      _, q = body_fun(self, (0, s))
    else:
      # Things are initialized
      _, q = nn.while_loop(cond_fun, body_fun, self, (0, s))
    return q

  def __call__(
      self, start_node_id, graph
  ):
    """Performs differentiable subgraph extraction.

    Args:
      start_node_id: initial start node id
      graph: underlying graph

    Returns:
      q_star: dense weights over nodes in the sparse subgraph
      node_features: features associated with the extracted subgraph
      dense_submat: the adjacency matrix of the extracted subgraph
      q: sparse weights over nodes. Used for debug purposes.
      adjacency_matrix: the sparse adjacency matrix. Used for debug purposes.
      error: L2 norm of q_t+1 - q_t. Should be 0 at convergence.
    """
    # TODO(gnegiar): remove unnecessary return values
    s = self._s(start_node_id)
    # TODO(gnegiar): Do we need to add a stop_gradient here
    q = self._ista_solve(s, graph)
    q = jax.lax.stop_gradient(q)
    # TODO(gnegiar): Find a way to avoid re-filling adjacency_matrix
    # For now, this allows to propagate gradients back to the `agent` model
    adjacency_matrix = self.agent.fill_sparse_adjacency_matrix(q, graph)
    # Extract dense submatrix
    dense_q = self._make_dense_vector(q, q.indices.flatten())
    dense_s = self._make_dense_vector(s, q.indices.flatten())
    dense_submat = self._extract_dense_submatrix(adjacency_matrix,
                                                 q.indices.flatten())

    def _fixed_point(q):
      return _dense_fixed_point(q, dense_submat, dense_s, self.config.alpha,
                                self.config.rho)

    def _tangent_solve(g, y):
      """Solve implicit function theorem linear system.

      Optionally, use ridge regularization on the normal equation.

      Args:
        g: the linearized zero function in the implicit function theorem. This
          is required by `custom_root`.
        y: the target, here the jvp for what comes after this layer. This is
          required by `custom_root`.

      Returns:
        jvp: the jvp for the subgraph extraction layer.
      """
      linearization = jax.jacobian(g)(y)
      if self.config.ridge != 0.:
        normal_mat_regularized, normal_target = _make_normal_system(
            linearization, y, self.config.ridge)
        jvp = jnp.linalg.solve(normal_mat_regularized, normal_target)
      else:
        jvp = jnp.linalg.solve(linearization, y)
      return jvp

    q_star = jax.lax.custom_root(
        f=_fixed_point,
        initial_guess=dense_q,
        solve=lambda _, q: dense_q,
        tangent_solve=_tangent_solve)

    node_features = jax.vmap(graph.node_features)(q.indices.flatten())

    node_ids = q.indices.flatten()

    error = self._error(q_star, dense_submat, dense_s)
    return q_star, node_features, node_ids, dense_submat, q, adjacency_matrix, error


# Utility functions


def _abs_top_k(u,
               k,
               nse = None):
  """Returns a sparse vector zeroing all but the top k values of `u` in magnitude.

  Args:
    u: BCOO 1d vector to threshold.
    k: number of elements to keep.
    nse: [Optional] Maximal allowed nse. If None passed, use `u.nse`.
  FYI: `nse` means the number of nonzero elements in the matrix. This number
    must be fixed, due to XLA requiring fixed shaped arrays.

  Returns:
    thresholded_u: BCOO 1d vector, where the top k values of `u` in magnitude
      were kept. Has the specified `nse`.
  """
  if nse is None:
    nse = u.nse
  if nse < k:
    raise ValueError(
        f"nse should be larger than the number of elements to keep. "
        f"Got nse={nse} and k={k}.")
  k = min(k, u.nse)
  # TODO(gnegiar): Benchmark speedups using jax.lax.approx_max_k
  _, idx = jax.lax.top_k(abs(u.data), k)
  # Pad to wanted nse
  pad_length = nse - len(idx)
  # Pad data with zeros
  data = jnp.concatenate((u.data[idx][:nse], jnp.zeros(pad_length)))
  # Pad indices with u.shape[0]
  indices = jnp.concatenate(
      (u.indices[idx][:nse], jnp.full((pad_length, 1), u.shape[0])))

  return jsparse.BCOO((data, indices), shape=u.shape)


def _sum_with_nse(mat, other_mat,
                  nse):
  """Returns the sum of two sparse arrays, with fixed nse.

  If `mat` has nse `a`, and `other_mat` has nse `b`, the nse of the sum is
  `a+b` at most. To satisfy `jax`'s fixed shape desiderata,
  we impose a fixed `nse` on the result.

  This may cause unexpected behavior when the true `nse` of `a+b` is more than
  `nse`.

  FYI: `nse` means the number of nonzero elements in the matrix. This number
    must be fixed, due to XLA requiring fixed shaped arrays.

  Args:
    mat: first array to add
    other_mat: second array to add
    nse: max nse of the result

  Returns:
    sum: array with nse=`nse`
  """
  result = mat + other_mat
  # Remove duplicate indices in result.
  result = result.sum_duplicates(nse=result.nse)
  # Return the topk `nse` items in magnitude.
  # TODO(gnegiar): print a warning when the clipping removes elements?
  return _abs_top_k(result, k=nse, nse=nse)


def _dense_q_minus_grad(q, dense_adjacency_matrix,
                        s, alpha):
  """Computes q-grad on dense arguments."""
  return (1. - alpha) * dense_adjacency_matrix.T @ q - alpha * s


def _softthresh(x, alpha, rho):
  """Performs soft-thresholding with alpha*rho threshold."""
  return jnp.sign(x) * jnp.maximum(jnp.abs(x) - alpha * rho, 0.)


def _dense_ista_step(q, dense_adjacency_matrix,
                     s, alpha, rho):
  """Performs a single step of ISTA for dense arguments."""
  q_minus_grad = _dense_q_minus_grad(q, dense_adjacency_matrix, s, alpha)
  return _softthresh(q_minus_grad, alpha, rho)


def _dense_fixed_point(q, dense_adjacency_matrix,
                       s, alpha, rho):
  """Returns the equation to be used in implicit differentiation."""
  q_ = _dense_ista_step(q, dense_adjacency_matrix, s, alpha, rho)
  return q - q_


# TODO(gnegiar): use binary search. Look at jnp.searchsorted.
# https://github.com/google/jax/pull/9108/files
# This would greatly lower memory requirements, at the possible cost of speed.
def _subscript(bcoo_mat, idx):
  """Returns a single element from a sparse matrix at a given index.

  Args:
    bcoo_mat: the sparse matrix to extract from.
    idx: indices to extract the element. The length of idx should match
      bcoo_mat.ndim.

  Returns:
    bcoo_mat[idx]
  """
  # Handle negative indices
  idx = jnp.where(idx >= 0, idx, jnp.array(bcoo_mat.shape) - (-idx))
  data, indices = bcoo_mat.data, bcoo_mat.indices
  # If indices are sorted, the mask should be findable via binary search.
  mask = jnp.all(indices == idx, axis=-1)
  return jnp.vdot(mask, data)  # Sum duplicate indices


def _dstack_product(x, y):
  """Returns the cartesian product of the elements of x and y vectors.

  Args:
    x: 1d array
    y: 1d array of the same dtype as x.

  Returns:
    a 2D array containing the elements of [x]x[y].
  Example:
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5]

    _dstack_product(x,y)
    >>> [[1, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5]]
  """
  return jnp.dstack(jnp.meshgrid(x, y, indexing="ij")).reshape(-1, 2)


def _make_normal_system(mat, b, ridge):
  """Makes regularized normal linear system.

  The normal system to `A x = b` is `(A.T A + ridge * Id) x = A.T b`.

  Args:
    mat: the A in above
    b: the target in the linear system
    ridge: amount of regularization to add.

  Returns:
    normal_mat_regularized: corresponds to `(A.T A + ridge * Id)`
    normal_target: corresponds to `A.T b`
  """
  normal_mat = mat.T @ mat
  normal_target = mat.T @ b
  # Add regularization to diagonal
  normal_mat_regularized = normal_mat.at[jnp.arange(normal_mat.shape[0]),
                                         jnp.arange(normal_mat.shape[1])].add(
                                             ridge)
  return normal_mat_regularized, normal_target

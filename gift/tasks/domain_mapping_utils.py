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

"""Utility functions for LTC method."""

import functools

import jax
from jax import jit

import jax.numpy as jnp


@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=(0, 0))
def align_examples(rng, x, x_index, y):
  """Random alignment based on labels.

  Randomly aligns items in x with items in y (and it's not a one-to-one map).
  The only costraint is that it tries to align items that have the same value.
  In the LTC Task, we pass labels to this function so that the alignment is
  based on labels.

  Here x and y are matrices where the rows are features to be compared
  elementwise for the alignment, and x_indice is the index of the batch
  position of x, needed for vmap.

  Args:
    rng: an array of jax PRNG keys.
    x: jnp.array; Matrix of shape `[N, M]`.
    x_index: jnp.array; Vector of shape `[N,]`.
    y: jnp.array; Matrix of shape `[N, M]`.

  Returns:
    indices of aligned pairs.
  """
  x = jnp.array(x)
  x_index = jnp.array(x_index)
  y = jnp.array(y)

  y_indices = jnp.arange(len(y))
  shuffled_y_idx = jax.random.permutation(rng, y_indices)
  equalities = jnp.float32(x == y[shuffled_y_idx])
  aligned_idx = jnp.argmax(equalities)

  return x_index, shuffled_y_idx[aligned_idx]


def pairwise_equality_1d(x, y):
  """Computes the pairwise boolean matrix out[i,j] = x[i, :] == y[j, :].

  Assumes x and y are integer-valued tensors.

  Args:
    x: Tensor of shape (Nx,) of integer type.
    y: Tensor of shape (Ny,) of integer type.

  Returns:
    Boolean matrix of shape (Nx, Ny) such that out[i,j] = x[i] == y[j]
  """

  assert x.dtype == y.dtype
  assert x.dtype in ['int32', 'int64'], x.dtype

  eq = x[:, jnp.newaxis] == y[jnp.newaxis, :]
  return jnp.float32(eq)


def pairwise_equality_2d(x, y):
  """Computes the pairwise boolean matrix out[i,j] = x[i, :] == y[j, :].

  Assumes x and y are integer-valued tensors.

  Args:
    x: tensor of shape (Nx, D) of integer type.
    y: tensor of shape (Ny, D) of integer type.

  Returns:
    Boolean matrix of shape (Nx, Ny) such that out[i,j] = x[i, :] == y[j, :]
  """
  assert x.dtype == y.dtype
  assert x.dtype in ['int32', 'int64'], x.dtype

  elem_eq = (x[:, jnp.newaxis, :] == y[jnp.newaxis, :, :])
  all_eq = jnp.min(elem_eq, axis=-1)
  return jnp.float32(all_eq)


def pairwise_l2(x, y):
  """Computes the cost matrix C_{ij} = ||x_{i,:}-y_{j,:}||^2_2.

  Args:
    x: tensor of shape (Nx, D).
    y: tensor of shape (Ny, D).

  Returns:
    Cost matrix of shape (Nx, Ny).
  """
  delta = x[:, jnp.newaxis, :] - y[jnp.newaxis, :, :]
  cost = jnp.sum(delta * delta, axis=-1)
  return cost


@functools.partial(jit, static_argnums=(4))
def sinkhorn_dual_solver(a,
                         b,
                         cost,
                         epsilon = 1e-2,
                         num_iters = 1000):
  """Runs Sinkhorn algorithm in log space for stability at small epsilon.

  Uses a fixed number of iterations to enable compatibility with jit.
  TODO(riannevdberg): implement dymanic num_iterations with jax control flows.

  Returns the regularized transport cost as in dual formaul 4.30 of
  https://arxiv.org/abs/1803.00567

  Note that the unregularized transport cost for a finite iteration solution of
  the potentials f and g is a lower bound on the regularized is a lower bound
  on the regurized transport cost with a converged solution for f and g (see
  proposition 4.8 of arxiv paper).

  If you need to compute the coupling P^L from the potentials
  after a finite number of iterations L, you should put
  the coupling through a rounding operation (round_coupling) to ensure that
  the coupling is a doubly stochastic. Otherwise <C, P^L> is not a valid
  approximation of L_C(a, b). After applying the rounding_coupling function
  to P^L <-- round(P^L) you can then use < C, P^L> as a valid approximation.

  Args:
   a: np.ndarray<float>[n]: discrete probability distribution. The rows of the
     coupling matrix P must sum up to this vector. If a represents an empirical
     distribution with n samples, all entries should be equal to 1/n.
   b: np.ndarray<float>[m]: discrete probability distribution. The columns of
     the coupling matrix P must sum up to this vector. If b represents an
     empirical distribution with m samples, all entries should be equal to 1/m.
   cost: np.ndarray<float>[n, m]: the cost matrix cost[i,j] = c(x_i, y_j) where
     x_i and y_j are samples from distributions a and b respectively.
   epsilon: (float) the level of entropic regularization wanted.
   num_iters: (int32) the number of Sinkhorn iterations.

  Returns:
   transportation cost (eq. 4.48 of paper), coupling (which needs to be rounded
     still with round_coupling method if used to compute a loss or if you want
     to ensure it has the correct marginals a and b, error in column marginal.
  """
  loga = jnp.expand_dims(jnp.log(a), axis=1)
  logb = jnp.expand_dims(jnp.log(b), axis=0)
  f = jnp.zeros_like(loga)  # epsilon * log_u
  g = jnp.zeros_like(logb)  # epsilon * log_v

  for _ in range(num_iters):
    # Note: If the update order is g before f, then check the error in b,
    # as this will be the largest error. If using the reverse error, then
    # check the error in a.

    # To carry out the logsumexp in a stable way we use the fact that
    # the matrix f + g - cost has all negative entries. We therefore use this
    # to add and subtract f and g in the respective updates in and outside the
    # logsumexp.

    g = epsilon * logb - epsilon * jax.scipy.special.logsumexp(
        (f + g - cost) / epsilon, axis=0, keepdims=True) + g
    f = epsilon * loga - epsilon * jax.scipy.special.logsumexp(
        (f + g - cost) / epsilon, axis=1, keepdims=True) + f

  # Compute error
  coupling = jnp.exp((f + g - cost) / epsilon)
  b_target = jnp.sum(coupling, axis=0)
  err = jnp.max(jnp.abs(b_target - b) / b, axis=None)

  # Compute unregularized cost according to eq. 4.48 of paper.
  # Note that if you want to compute the regularized cost of eq. 4.30
  # this only requires subtracting epsilon, as the double sum
  # < e^f/eps, K e^g/eps > = 1 for updates like in this sinkhorn algorithm.
  transport_cost = jnp.sum(f * a) + jnp.sum(g * b)
  return transport_cost, coupling, err


def idx2permutation(row_ind, col_ind):
  """Constructs a permutation matrix from the column and row indices of ones."""

  dim = row_ind.shape[0]
  perm = jnp.zeros(shape=(dim, dim), dtype='float32')
  perm = perm.at[(row_ind, col_ind)].set(1.)
  return perm


def sample_permutation(key, coupling):
  """Samples a permutation matrix from a doubly stochastic coupling matrix.

  CAREFUL: the couplings that come out of the sinkhorn solver
  are not doubly stochastic but 1/dim * doubly_stochastic.

  See **Convex Relaxations for Permutation Problems** paper for rough
  explanation of the algorithm.
  Best to use by drawing multiple samples and picking the permutation with
  lowest cost as sometimes permutations seem to be drawn with high cost.
  the sample_best_permutation method does this.

  Args:
    key: jnp.ndarray that functions as a PRNG key.
    coupling: jnp.ndarray of shape [N, N] which must have marginals such that
      coupling.sum(0) == 1. and coupling.sum(1) == 1. Note that in sinkhorn we
      usually output couplings with marginals that sum to 1/N.

  Returns:
    permutation matrix: jnp.ndarray of shape [N, N] of floating dtype.
  """
  dim = coupling.shape[0]

  # random monotonic vector v without duplicates.
  v = jax.random.choice(key, 10 * dim, shape=(dim,), replace=False)
  v = jnp.sort(v) * 10.

  w = jnp.dot(coupling, v)
  # Sorting w will give the row indices of the permutation matrix.
  row_ind = jnp.argsort(w)
  col_ind = jnp.arange(0, dim)

  # Compute permutation matrix from row and column indices
  perm = idx2permutation(row_ind, col_ind)
  return perm


def sample_best_permutation(key, coupling, cost, num_trials=10):
  """Samples permutation matrices and returns the one with lowest cost.

  See **Convex Relaxations for Permutation Problems** paper for rough
  explanation of the algorithm.

  Args:
    key: jnp.ndarray that functions as a PRNG key.
    coupling: jnp.ndarray of shape [N, N]
    cost: jnp.ndarray of shape [N, N].
    num_trials: int, determins the amount of times we sample a permutation.

  Returns:
    permutation matrix: jnp.ndarray of shape [N, N] of floating point type.
      this is the permutation matrix with lowest optimal transport cost.
  """
  vec_sample_permutation = jax.vmap(
      sample_permutation, in_axes=(0, None), out_axes=0)
  key = jax.random.split(key, num_trials)
  perms = vec_sample_permutation(key, coupling)

  # Pick the permutation with minimal ot cost
  ot = jnp.sum(perms * cost[jnp.newaxis, :, :], axis=(1, 2))
  min_idx = jnp.argmin(ot)
  out_perm = perms[min_idx]
  return out_perm


def round_coupling(coupling, a, b):
  """Projects a coupling matrix to the nearest matrix with marginals a and b.

  A finite number of sinkhorn iterations will always lead to a coupling P
  which does not satisfy the constraints that sum_j P[:, j] = a
  or sum_i P[i, :] = b.
  This differential rounding operation from algorithm 2 from Altschuler et al.
  ensures that you map tot he nearest matrix that does satisfy the constraints.

  Note: some implementations convert coupling, a and b to double precision
  before performing the algorithm. In case of instability, try that.

  Args:
    coupling: jnp.ndarray of shape [N, M]. Approximate coupling that results
      from for instance a Sinkhorn solver.
    a: jnp.ndarray of shape [N,]. The desired marginal of the rows of the
      coupling matrix.
    b: jnp.ndarray of shape [M,]. The desired marginal of the columns of the
      coupling matrix.

  Returns:
    r_coupling: jnp.ndarray of shape [N, M] such that
      r_coupling.sum(0) == b and r_coupling.sum(1) == a.
  """

  a_div_coupling = jnp.divide(a, coupling.sum(1))
  x = 1. - jax.nn.relu(1. - a_div_coupling)
  pp = x.reshape((-1, 1)) * coupling

  b_div_coupling = jnp.divide(b, coupling.sum(0))
  y = 1. - jax.nn.relu(1. - b_div_coupling)
  pp = pp * y.reshape((1, -1))

  err_a = a - pp.sum(1)
  err_b = b - pp.sum(0)

  kron_ab = err_a[:, jnp.newaxis] * err_b[jnp.newaxis, :]
  r_coupling = pp + kron_ab / jnp.sum(jnp.abs(err_a))

  return r_coupling

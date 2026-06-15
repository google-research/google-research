# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Ripple count mechanism."""

from functools import lru_cache
import numpy as np
from scipy import special
from dp_ripple import ripple_sum

# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name

comb = special.comb


@lru_cache(maxsize=None)
def _get_comb(n, k):
  """Memoized combination function to avoid redundant factorial math."""
  return special.comb(n, k)


def _compute_normalizing_constant_for_count(d, k, eps):
  """Returns Z(K_{n}(e^{-eps})). See Lemma 4.13 for details.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
  """
  if k == 0:
    return 1
  A = ripple_sum.compute_joint_descent_coefficients(d)
  z = np.exp(-eps)

  def compute_inner_sum(support_size, k, z):
    if support_size == 0:
      return 1
    result = 0
    for i in range(support_size - k + 1, support_size + 1):
      for w in range(support_size + 1):
        result += A[support_size][i][w] * (z**w)
    return (1 - z) ** (-support_size) * result

  result = 0
  for p in range(d + 1):
    for m in range(d - p + 1):
      summand = (
          _get_comb(d, p)
          * _get_comb(d - p, m)
          * compute_inner_sum(p, k, z)
          * compute_inner_sum(m, k, z)
      )
      result += summand
  return result


def _compute_L_layerindex_supportsize(layer_index, support_size, k, C):
  """Returns |L_{layer_index, support_size, X}| for any X \subset [d].

  In the above statement, |X| = support_size.

  Args:
    layer_index: Integer layer index.
    support_size: Integer support size.
    k: Integer l_0 bound.
    C: Numpy array output of compute_ordered_restricted_partitions_table(d,
      layer_index*k, layer_index-2)
  """
  X_J_weight, X_J_complement_weight = ripple_sum.compute_XJ_XJc_sizes(
      support_size, k, layer_index, C
  )
  return X_J_weight + X_J_complement_weight


def _compute_K_n_p_m_weights(d, n, k, C_per_n):
  """Returns the 2-d array of weights K_{n,p,m}_{0<=p+m<=d}.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    k: Integer l_0 bound.
    C_per_n: Numpy array output of
      compute_ordered_restricted_partitions_table(d, n*k, n-1) including all
      cell indices (r, c, m) where r <= d, c <= n*k, m <= n-1.
  """
  # 1. Pre-calculate L values for all possible support sizes p at this n
  # L_table shape: (n+1, d+1) -> rows are layer 'a', cols are support size 'p'
  L_table = np.zeros((n + 1, d + 1))
  for a in range(n + 1):
    for p in range(d + 1):
      L_table[a, p] = _compute_L_layerindex_supportsize(
          a, p, k, C_per_n[:, :, a]
      )

  # 2. Compute the sum over 'a' for all pairs of (p, m)
  # The sum is: weight_sum_across_a[p, m] = sum_{a=0}^{n} L(a, p) * L(n-a, m)
  # This is equivalent to a matrix multiplication of L_table transpose
  # with its row-reversed self.
  weight_sum_across_a = np.dot(L_table.T, L_table[::-1, :])

  # 3. Apply the binomial coefficients Comb(d, p) * Comb(d-p, m)
  # We can pre-calculate this or compute it via broadcasting
  p_indices = np.arange(d + 1)[:, None]  # Column vector
  m_indices = np.arange(d + 1)  # Row vector

  # Mask to ensure m <= d - p
  mask = p_indices + m_indices <= d

  # Vectorized binomials
  comb_p = special.comb(d, p_indices)
  comb_m = special.comb(d - p_indices, m_indices)

  return weight_sum_across_a * comb_p * comb_m * mask


def _sample_layer_index_for_K_n_J_decomposition(n, p, m, k, C_per_n):
  """Returns a layer index for the K_n_J decomposition.

  Recall that K_n_J decomposes as a disjoint union of cartesian products each of
  which is indexed by a layer index.

  Args:
    n: Integer layer index.
    p: Integer support size for positive coordinates.
    m: Integer support size for negative coordinates.
    k: Integer l_0 bound.
    C_per_n: Numpy array output of
      compute_ordered_restricted_partitions_table(d, n*k, n-1) including all
      cell indices (r, c, m) where r <= d, c <= n*k, m <= n-1.
  """
  # Vectorize the retrieval of L values for all 'a'
  L_a_p = np.array([
      _compute_L_layerindex_supportsize(a, p, k, C_per_n[:, :, a])
      for a in range(n + 1)
  ])
  L_na_m = np.array([
      _compute_L_layerindex_supportsize(n - a, m, k, C_per_n[:, :, n - a])
      for a in range(n + 1)
  ])

  layer_index_weights = L_a_p * L_na_m
  weight_sum = np.sum(layer_index_weights)

  return np.random.choice(np.arange(n + 1), p=layer_index_weights / weight_sum)


def sample_ripple_count_point(d, k, eps, num_samples):
  """Returns points sampled uniformly from the ripple sum mechanism.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
    num_samples: Integer number of samples to return.
  """
  normalizing_constant = _compute_normalizing_constant_for_count(d, k, eps)
  max_n = 10 * d
  C_per_n = np.zeros((d + 1, max_n * k + 1, max_n + 1))

  w_n_dist = []
  weights_cache = {}  # Cache weights calculated during the distribution phase

  for n in range(max_n + 1):
    C_table = ripple_sum.compute_ordered_restricted_partitions_table(
        d, k * n, n - 1 if n >= 1 else -1
    )
    C_per_n[: C_table.shape[0], : C_table.shape[1], n] = C_table

    K_weights = _compute_K_n_p_m_weights(d, n, k, C_per_n)
    weights_cache[n] = K_weights  # Store for the sampling phase

    n_weight = np.sum(K_weights) * np.exp(-n * eps) / normalizing_constant
    w_n_dist.append(n_weight)

    # Convergence check to avoid unnecessary iterations
    if n > d and n_weight < 1e-16:
      break

  probs = np.array(w_n_dist) / np.sum(w_n_dist)
  all_ns = np.random.choice(len(probs), size=num_samples, p=probs)

  # Group ns to batch sample (p, m) pairs
  unique_ns, counts = np.unique(all_ns, return_counts=True)
  samples = []

  for n, count in zip(unique_ns, counts):
    W = weights_cache[n]
    W_flat = W.flatten()
    W_sum = np.sum(W_flat)

    # Batch sample p and m indices
    flat_indices = np.random.choice(W.size, size=count, p=W_flat / W_sum)
    ps, ms = np.unravel_index(flat_indices, W.shape)

    for p, m in zip(ps, ms):
      samples.append(_construct_point(d, n, p, m, k, C_per_n))

  return samples


def _construct_point(d, n, p, m, k, C_per_n):
  """Returns a single point constructed from sampled signature and layer index.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    p: Integer positive support size.
    m: Integer negative support size.
    k: Integer l_0 bound.
    C_per_n: 3D numpy array of restricted partition tables.
  """
  random_permutation = np.random.permutation(d)
  P = random_permutation[:p]
  M = random_permutation[p : p + m]

  a = _sample_layer_index_for_K_n_J_decomposition(n, p, m, k, C_per_n)

  v_p = np.zeros(d)
  if p > 0:
    v_p = ripple_sum.sample_L_n_s_J_point(d, p, P, k, a, C_per_n[:, :, a])

  v_m = np.zeros(d)
  if m > 0:
    v_m = -1 * ripple_sum.sample_L_n_s_J_point(
        d, m, M, k, n - a, C_per_n[:, :, n - a]
    )

  return v_p + v_m


def _compute_w_n_for_count(d, k, n, eps, normalizing_constant, C_per_n):
  """Returns $w_{n} = K_{n}e^{-n\eps}/N$ as defined in Lemma 4.15.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    n: Integer layer index.
    eps: Float privacy parameter.
    normalizing_constant: Float normalizing constant for the ripple sum.
    C_per_n: Numpy array output of
      compute_ordered_restricted_partitions_table(d, n*k, n-1) including all
      cell indices (r, c, m) where r <= d, c <= n*k, m <= n-1.
  """
  K_n_p_m_weights = _compute_K_n_p_m_weights(d, n, k, C_per_n)
  K_n = np.sum(K_n_p_m_weights)
  return K_n * np.exp(-n * eps) / normalizing_constant

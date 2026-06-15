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

"""Ripple sum mechanism."""

from functools import lru_cache
import numpy as np
from scipy import special

# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name


# Memoize combinations for a significant speed boost
@lru_cache(maxsize=None)
def _get_comb(n, k):
  """Memoized combination function to avoid redundant math."""
  return special.comb(n, k)


def compute_joint_descent_coefficients(d):
  """Returns the generating function coefficients of A_{d}(y,z) from Lemma 10.4.

  Args:
    d: Integer dimension.
  """
  joint_descent_coefficients = np.zeros((d + 1, d + 1, d + 1))
  joint_descent_coefficients[1][1][1] = 1
  for n in range(2, d + 1):
    prev_A = joint_descent_coefficients[n - 1]
    for i in range(1, n + 1):
      for j in range(1, n + 1):
        recurrence = (
            (i * j + n - 1) * prev_A[i][j]
            + (1 - n + j * (n + 1 - i)) * prev_A[i - 1][j]
            + (1 - n + i * (n + 1 - j)) * prev_A[i][j - 1]
            + (n - 1 + (n + 1 - i) * (n + 1 - j)) * prev_A[i - 1][j - 1]
        )
        joint_descent_coefficients[n][i][j] = recurrence / n
  return joint_descent_coefficients


def _compute_normalizing_constant_for_sum(d, k, eps):
  """Returns Z(L_{n}(e^{-eps})). See Corollary 10.6 for details.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
  """
  if k == 0:
    return 1
  A = compute_joint_descent_coefficients(d)
  z = np.exp(-eps)
  summation = 0
  for s in range(0, d + 1):
    subsummation = 0
    for i in range(s - k + 1, s + 1):
      if i == 0:
        if s == 0:
          subsummation += 1
      else:
        subsummation += np.dot(
            A[s][i][1 : s + 1], np.power(z, np.arange(1, s + 1))
        )
    subsummation *= _get_comb(d, s) * ((1 - z) / 2) ** (-s)
    summation += subsummation
  return summation


def compute_ordered_restricted_partitions_table(d, target_sum, max_summand):
  """Returns the ordered restricted partitions table up to cell index [d][target_sum].

  See definition 4.10 for details
  Args:
    d: Integer dimension.
    target_sum: Non-negative integer target sum.
    max_summand: Positive integer max value of a_i.
  """

  C = np.zeros((d + 1, target_sum + 1))
  if max_summand < 1 or target_sum < 0:
    return C
  for i in range(min(d, target_sum)):
    C[i, i] = 1
  C[1][1 : min(target_sum, max_summand) + 1] = 1
  for i in range(2, d + 1):
    row_cumsum = np.cumsum(C[i - 1][i - 1 :])
    C[i][i:] = row_cumsum[:-1]
    if target_sum - (i - 1) > max_summand:
      C[i][max_summand + (i - 1) + 1 :] -= row_cumsum[
          : target_sum - max_summand - (i - 1)
      ]
  return C


def _sample_ordered_restricted_partition(d, t, m, C):
  """Returns a uniformly sampled partition of t into d positive parts each of size <= m.

  Args:
    d: Integer dimension.
    t: Positive integer target sum.
    m: Positive integer max value of a_i.
    C: Numpy array output of compute_ordered_restricted_partitions_table(d, t,
      m).
  """
  if t == d:
    return np.ones(d, dtype=int)
  result = np.zeros(d, dtype=int)
  current_t = t
  for current_d in range(d, 1, -1):
    max_coord = min(m, current_t)
    # Probabilities for the first coordinate in this step
    weights = C[current_d - 1][current_t - max_coord : current_t][::-1]
    probs = weights / np.sum(weights)
    chosen_summand = np.random.choice(range(1, max_coord + 1), p=probs)
    result[d - current_d] = chosen_summand
    current_t -= chosen_summand
  result[-1] = current_t
  return result


def _compute_X_J_i_weights(s, k, n, i, C):
  """Returns the summands of the binomial expression for X_{J,i} for each J.

  Args:
    s: Integer support size.
    k: Integer l_0 bound.
    n: Integer layer index.
    i: Integer number of non-maximum coordinates.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  return C[s - i][: n * k - n * i + 1]


def _compute_X_J_size(s, k, n, C):
  """Returns |X_J| for any index set J of size s.

  Args:
    s: Integer support size.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  if n == 1:
    return 1 if s <= k else 0
  total = 0
  for i in range(1, k + 1):
    total += _get_comb(s, i) * np.sum(_compute_X_J_i_weights(s, k, n, i, C))
  return total


def _sample_X_J_point(d, s, J, k, n, C):
  """Return a point sampled uniformly at random from X_{J}.

  Args:
    d: Integer dimension.
    s: Integer support size.
    J: Numpy array of size s which is the support of the sampled point.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  if n == 1:
    assert len(J) <= k
    result = np.zeros(d)
    result[J] = 1
  else:
    i_weights = np.array([
        _get_comb(s, i) * np.sum(_compute_X_J_i_weights(s, k, n, i, C))
        for i in range(1, k + 1)
    ])
    chosen_i = np.random.choice(range(1, k + 1), p=i_weights / i_weights.sum())

    t_weights = _compute_X_J_i_weights(s, k, n, chosen_i, C)
    chosen_t = np.random.choice(len(t_weights), p=t_weights / t_weights.sum())

    non_maximum_coordinates = _sample_ordered_restricted_partition(
        s - chosen_i, chosen_t, n - 1, C
    )
    J_vector = np.zeros(s)
    perm = np.random.permutation(s)
    J_vector[perm[:chosen_i]] = n
    J_vector[perm[chosen_i:]] = non_maximum_coordinates
    result = np.zeros(d)
    result[J] = J_vector

  # flip signs of each coordinate randomly
  return result * (2 * np.random.randint(0, 2, size=d) - 1)


def _sample_X_J_complement_point(d, s, J, k, n, C):
  """Return a point sampled uniformly at random from L_n - X.

  Args:
    d: Integer dimension.
    s: Integer support size.
    J: Numpy array of size s which is the support of the sampled point.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  weights = C[s][max(0, n * k - k + 1) : n * k + 1]
  chosen_t = np.random.choice(
      range(max(0, n * k - k + 1), n * k + 1),
      p=weights / np.sum(weights),
  )
  J_vector = _sample_ordered_restricted_partition(s, chosen_t, n - 1, C)
  result = np.zeros(d)
  result[J] = J_vector
  # flip signs of each coordinate randomly
  return result * (2 * np.random.randint(0, 2, size=d) - 1)


def compute_XJ_XJc_sizes(s, k, n, C):
  """Returns |X_{J}|, |X_{J}^{c}| as defined in definition 4.1.

  Args:
    s: Integer support size.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  if n == 0:
    return (1, 0) if s == 0 else (0, 0)
  if k <= 0 or s == 0 or s > n * k:
    return 0, 0
  if n == 1:
    return 1, 0

  xj_size = _compute_X_J_size(s, k, n, C)
  xjc_size = np.sum(C[s][max(0, n * k - k + 1) : n * k + 1])
  return xj_size, xjc_size


def _compute_L_n_s_weights(d, k, n, C):
  """Returns the weights of \{2^{s}|L_{n,s}| : s = 0,...,d\}.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  L_n_s_weights = np.zeros(d + 1)
  for s in range(0, d + 1):
    xj_size, xjc_size = compute_XJ_XJc_sizes(s, k, n, C)
    L_n_s_weights[s] = 2**s * _get_comb(d, s) * (xj_size + xjc_size)
  return L_n_s_weights


def sample_L_n_s_J_point(d, s, J, k, n, C):
  """Return a point sampled uniformly at random from L_{n,s,J}.

  Args:
    d: Integer dimension.
    s: Integer support size.
    J: Numpy array of support indices.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  assert s > 0
  xj_size, xjc_size = compute_XJ_XJc_sizes(s, k, n, C)
  threshold = xj_size / (xj_size + xjc_size)
  if np.random.rand() < threshold:
    return _sample_X_J_point(d, s, J, k, n, C)
  return _sample_X_J_complement_point(d, s, J, k, n, C)


def _sample_point_from_L_n(d, k, n, C):
  """Return a point sampled uniformly at random from L_n.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    n: Integer layer index.
    C: Partition table from compute_ordered_restricted_partitions_table.
  """
  if n == 0:
    return np.zeros(d)
  weights = _compute_L_n_s_weights(d, k, n, C)
  chosen_s = np.random.choice(range(d + 1), p=weights / np.sum(weights))
  chosen_J = np.random.permutation(d)[:chosen_s]
  return sample_L_n_s_J_point(d, chosen_s, chosen_J, k, n, C)


def sample_ripple_sum_point(d, k, eps, num_samples):
  """Returns points sampled uniformly from the ripple sum mechanism.

  Args:
    d: Integer dimension.
    k: Integer l_0 bound.
    eps: Float privacy parameter.
    num_samples: Integer number of samples to return.
  """
  # 1. Pre-calculate the normalizing constant once
  normalizing_const = _compute_normalizing_constant_for_sum(d, k, eps)

  # 2. Determine a safe upper bound for n (where probability decays to ~0)
  max_n = int(np.ceil(15 * d / eps))

  # 3. Pre-fill the Partition Tables (C) and compute Layer Weights (w_n)
  C_per_n = np.zeros((d + 1, max_n * k + 1, max_n + 1))
  w_n_distribution = []

  for n in range(max_n + 1):
    # Build table for this layer
    table = compute_ordered_restricted_partitions_table(
        d, k * n, n - 1 if n >= 1 else -1
    )
    C_per_n[: table.shape[0], : table.shape[1], n] = table

    # Calculate the weight of this entire layer n
    L_n_weight = np.sum(_compute_L_n_s_weights(d, k, n, C_per_n[:, :, n]))
    w_n_distribution.append(L_n_weight * np.exp(-n * eps) / normalizing_const)

  # 4. Batch sample all layer indices at once
  probs = np.array(w_n_distribution)
  probs /= probs.sum()  # Ensure it sums to 1
  sampled_ns = np.random.choice(len(probs), size=num_samples, p=probs)

  # 5. Generate the point samples
  samples = []
  for n in sampled_ns:
    samples.append(_sample_point_from_L_n(d, k, n, C_per_n[:, :, n]))

  return samples

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

"""Induced norm sum mechanism."""

import numpy as np
from scipy import stats

from k_norm import lp_mechanism


def compute_eulerian_numbers(d):
  """Returns A where A[i, j] = Eulerian number A(i, j).

  Args:
    d: Integer such that the returned matrix has d+1 rows and columns.
  """
  eulerian_numbers = np.zeros((d + 1, d + 1))
  eulerian_numbers[:, 0] = np.ones(d + 1)
  for row in range(2, d + 1):
    for k in range(1, row + 1):
      eulerian_numbers[row, k] = (row - k) * eulerian_numbers[
          row - 1, k - 1
      ] + (k + 1) * eulerian_numbers[row - 1, k]
  return eulerian_numbers


def compute_add_ascent_indices(eulerian_numbers, k):
  """Returns array where array[i] = 1 <==> i + 1 adds an ascent when inserted.

  Args:
    eulerian_numbers: Pre-computed ndarray of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer number of ascents.

  Returns:
    An array specifying the values which add an ascent upon insertion when
    iteratively constructing a permutation in S_d. For example, [0, 1, 1, 0]
    means that 2 and 3 add ascents when inserted, and this corresponds to
    permutations (4123), (1423), and (1243).
  """
  d = len(eulerian_numbers) - 1
  add_ascents = np.zeros(d)
  current_k = k
  for current_d in range(1, d + 1)[::-1]:
    left_mass = (current_d - current_k) * eulerian_numbers[
        current_d - 1, current_k - 1
    ]
    right_mass = (current_k + 1) * eulerian_numbers[current_d - 1, current_k]
    if np.random.uniform() < left_mass / (left_mass + right_mass):
      add_ascents[current_d - 1] = 1
      current_k = current_k - 1
    else:
      add_ascents[current_d - 1] = 0
  return add_ascents


def sample_permutation_with_ascents(eulerian_numbers, k):
  """Returns a uniform random permutation on [d] with k ascents.

  Args:
    eulerian_numbers: Pre-computed ndarray of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer number of ascents.
  """
  add_ascents = compute_add_ascent_indices(eulerian_numbers, k)
  d = len(add_ascents)
  permutation = np.ones(1)
  # i ranges over numbers to insert
  for i in range(2, d + 1):
    if add_ascents[i - 1]:
      insertion_options = list(
          np.where(permutation[1:] < permutation[:-1])[0] + 1
      ) + [i - 1]
    else:
      insertion_options = list(
          np.where(permutation[1:] > permutation[:-1])[0] + 1
      ) + [0]
    insert_idx = np.random.choice(insertion_options, 1)[0]
    permutation = np.insert(permutation, min(len(permutation), insert_idx), i)
  return permutation.astype(int)


def phi(x):
  """Returns phi(x), where phi is Stanley's bijection.

  Args:
    x: Float in (0, 1).

  Returns:
    phi(x), where phi is the function described on the first page of
    https://math.mit.edu/~rstan/pubs/pubfiles/34a.pdf.
  """
  d = len(x)
  new_x = np.zeros(d+1)
  new_x[1:] = x
  x_diffs = new_x[:-1] - new_x[1:]
  y = x_diffs + (x_diffs < 0)
  return y


def sample_slice_index(eulerian_numbers, k):
  """Returns a slice index of the cube, sampled proportionally to its volume.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer number of ascents.
  """
  slices = eulerian_numbers[-1, :k]
  weights = slices / np.sum(slices)
  return np.random.choice(len(slices), 1, p=weights)[0]


def sample_fundamental_simplex(d):
  """Returns a uniform random sample from the fundamental simplex.

  Args:
    d: Integer dimension of the fundamental simplex.
  """
  convex_combination_weights = stats.dirichlet.rvs([1] * (d + 1))
  fundamental_simplex_vertices = np.zeros((d + 1, d))
  for i in range(1, d + 1):
    fundamental_simplex_vertices[i - 1, -i:] = 1
  return np.sum(
      np.transpose(convex_combination_weights) * fundamental_simplex_vertices,
      axis=0,
  )


def sample_sum_ball(eulerian_numbers, k):
  """Returns a uniform sample from the induced norm ball for sum.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer l_0 bound.
  """
  d = len(eulerian_numbers) - 1
  slice_index = sample_slice_index(eulerian_numbers, k)
  sigma = sample_permutation_with_ascents(eulerian_numbers, slice_index)
  fundamental_simplex_sample = sample_fundamental_simplex(d)
  permuted_fundamental_simplex_sample = fundamental_simplex_sample[sigma - 1]
  return lp_mechanism.random_signs(phi(permuted_fundamental_simplex_sample))


def sum_mechanism(vector, eulerian_numbers, k, epsilon):
  """Returns a sample from the epsilon-DP induced norm mechanism for sum.

  Args:
    vector: The output will be a noisy version of Numpy array vector.
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: l_0 bound.
    epsilon: The output will be epsilon-DP for float epsilon. Assumes vector is
      1-sensitive with respect to the sum ball.

  Returns:
    A sample from the K-norm mechanism, as described in Section 4 of
    https://arxiv.org/abs/0907.3754, instantiated with the norm induced by sum.
  """
  d = len(vector)
  radius = np.random.gamma(shape=d + 1, scale=1 / epsilon)
  ball_sample = sample_sum_ball(eulerian_numbers, k)
  noise = radius * ball_sample
  return vector + noise
